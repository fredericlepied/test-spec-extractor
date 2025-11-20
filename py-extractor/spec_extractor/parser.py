"""AST parser for Python test files, building Container/TestCase tree structure."""

import ast
import os
import sys
from typing import Dict, List, Optional

from .types import Container, TestCase, TestStep, FileSpec

# Python version compatibility
PYTHON_38_PLUS = sys.version_info >= (3, 8)


def _get_string_value(expr: ast.expr) -> Optional[str]:
    """Extract string value from AST expression, compatible with Python 3.7 and 3.8+."""
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return expr.value
    elif not PYTHON_38_PLUS and hasattr(ast, "Str") and isinstance(expr, ast.Str):
        return expr.s
    return None


def parse_file(file_path: str) -> FileSpec:
    """Parse a Python file and build FileSpec."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            # Return empty spec for unparseable files
            return FileSpec(file_path=file_path)

    spec = FileSpec(file_path=file_path)
    visitor = TestVisitor(file_path)
    visitor.visit(tree)
    visitor.build_spec(spec)
    return spec


def build_file_spec(file_path: str) -> FileSpec:
    """Build FileSpec from a Python file (alias for parse_file for compatibility)."""
    return parse_file(file_path)


class TestVisitor(ast.NodeVisitor):
    """AST visitor that builds Container/TestCase structure from Python tests."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.root = Container(kind="Root")
        self.container_stack: List[Container] = [self.root]
        self.fixtures: Dict[str, List[TestStep]] = {}  # fixture_name -> steps
        self.current_class: Optional[str] = None
        self.class_setup_ops: Dict[str, List[TestStep]] = {}  # class_name -> setup steps
        self.class_teardown_ops: Dict[str, List[TestStep]] = {}  # class_name -> teardown steps

    def current_container(self) -> Container:
        """Get the current container from the stack."""
        return self.container_stack[-1]

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit a class definition."""
        # Check if this is a test class
        is_test_class = (
            node.name.startswith("Test")
            or any(
                base.id == "TestCase"
                for base in node.bases
                if isinstance(base, ast.Name) and hasattr(base, "id")
            )
        )

        if is_test_class:
            # Create a container for this test class
            container = Container(kind="Describe", description=node.name)
            container.labels = self._extract_labels_from_decorators(node.decorator_list)

            # Extract setup/teardown methods
            setup_steps, teardown_steps = self._extract_class_setup_teardown(node)
            container.prep_steps.extend(setup_steps)
            container.cleanup_steps.extend(teardown_steps)
            self.class_setup_ops[node.name] = setup_steps
            self.class_teardown_ops[node.name] = teardown_steps

            # Add to current container
            self.current_container().children.append(container)
            self.container_stack.append(container)
            self.current_class = node.name

            # Visit class body
            self.generic_visit(node)

            # Pop container
            self.container_stack.pop()
            self.current_class = None
        else:
            # Not a test class, visit normally
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit a function definition."""
        # Check if this is a pytest fixture
        if self._is_pytest_fixture(node):
            fixture_name = node.name
            steps = self._extract_steps_from_function(node)
            self.fixtures[fixture_name] = steps
            return

        # Check if this is a test function
        is_test_function = (
            node.name.startswith("test_")
            or node.name.startswith("Test")
            or self._has_pytest_parametrize(node)
        )

        if not is_test_function:
            # Not a test function, visit normally
            self.generic_visit(node)
            return

        # If we're at root level (no test class), create a file-level container
        if len(self.container_stack) == 1:  # Only root in stack
            file_container = Container(kind="Describe", description=os.path.basename(self.file_path))
            self.root.children.append(file_container)
            self.container_stack.append(file_container)

        # Create a test case
        test_case = TestCase()
        test_case.description = self._extract_test_description(node)
        test_case.labels = self._extract_labels_from_decorators(node.decorator_list)

        # Extract skip conditions
        skip_reason = self._extract_skip_reason(node.decorator_list)
        if skip_reason:
            test_case.prep_steps.append(TestStep(text=f"SKIP: {skip_reason}"))

        # Extract steps from function body
        test_case.steps = self._extract_steps_from_function(node)

        # Extract fixture dependencies and add as prep steps
        fixture_prep_steps = self._extract_fixture_prep_steps(node)
        test_case.prep_steps.extend(fixture_prep_steps)

        # Add class-level setup/teardown if in a class
        if self.current_class:
            class_setup = self.class_setup_ops.get(self.current_class, [])
            class_teardown = self.class_teardown_ops.get(self.current_class, [])
            # Note: class setup/teardown are added to container, not test case
            # But we track them for reference

        # Handle parametrized tests
        if self._has_pytest_parametrize(node):
            # Extract parametrize values and create multiple test cases
            param_values = self._extract_parametrize_values(node)
            if param_values:
                for param_value in param_values:
                    param_test_case = TestCase()
                    param_test_case.description = f"{test_case.description} [{param_value}]"
                    param_test_case.labels = test_case.labels.copy()
                    param_test_case.prep_steps = test_case.prep_steps.copy()
                    param_test_case.steps = test_case.steps.copy()
                    param_test_case.cleanup_steps = test_case.cleanup_steps.copy()
                    self.current_container().cases.append(param_test_case)
            else:
                # Single parametrized test case
                self.current_container().cases.append(test_case)
        else:
            self.current_container().cases.append(test_case)

        self.generic_visit(node)

    def build_spec(self, spec: FileSpec):
        """Build the final FileSpec from collected data."""
        # If we have test cases but no containers (no test classes),
        # create a file-level container
        if not self.root.children and self.root.cases:
            # Move cases to a file-level container
            file_container = Container(kind="Describe", description=os.path.basename(spec.file_path))
            file_container.cases = self.root.cases
            self.root.cases = []
            self.root.children.append(file_container)
        spec.root = self.root

    def _is_pytest_fixture(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a pytest fixture."""
        for decorator in node.decorator_list:
            # Check for @pytest.fixture
            if isinstance(decorator, ast.Attribute):
                if decorator.attr == "fixture":
                    if isinstance(decorator.value, ast.Name) and decorator.value.id == "pytest":
                        return True
            # Check for @pytest.fixture(...)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr == "fixture":
                        if (
                            isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == "pytest"
                        ):
                            return True
        return False

    def _has_pytest_parametrize(self, node: ast.FunctionDef) -> bool:
        """Check if function has @pytest.mark.parametrize decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute):
                    if func.attr == "parametrize":
                        if isinstance(func.value, ast.Attribute) and func.value.attr == "mark":
                            if isinstance(func.value.value, ast.Name) and func.value.value.id == "pytest":
                                return True
        return False

    def _extract_parametrize_values(self, node: ast.FunctionDef) -> List[str]:
        """Extract parametrize values from decorator."""
        values = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute) and func.attr == "parametrize":
                    # Look for values in decorator args
                    # This is simplified - real parametrize can be complex
                    if len(decorator.args) >= 2:
                        # Second arg is usually the values
                        values_arg = decorator.args[1]
                        if isinstance(values_arg, (ast.List, ast.Tuple)):
                            for item in values_arg.elts:
                                str_val = _get_string_value(item)
                                if str_val is not None:
                                    values.append(str_val)
        return values

    def _extract_test_description(self, node: ast.FunctionDef) -> str:
        """Extract test description from function name and docstring."""
        # First try docstring
        docstring = ast.get_docstring(node)
        if docstring:
            # Use first line of docstring
            first_line = docstring.split("\n")[0].strip()
            if first_line:
                return first_line

        # Fall back to function name, converting to readable format
        name = node.name
        # Remove test_ prefix
        if name.startswith("test_"):
            name = name[5:]
        # Convert underscores to spaces
        name = name.replace("_", " ")
        return name or "unnamed test"

    def _extract_labels_from_decorators(self, decorator_list: List[ast.expr]) -> List[str]:
        """Extract labels from pytest.mark decorators."""
        labels = []
        for decorator in decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute):
                    # @pytest.mark.label_name
                    if isinstance(func.value, ast.Attribute) and func.value.attr == "mark":
                        if isinstance(func.value.value, ast.Name) and func.value.value.id == "pytest":
                            label = func.attr
                            if label not in ["skip", "skipif", "parametrize", "usefixtures"]:
                                labels.append(label)
            elif isinstance(decorator, ast.Attribute):
                # @pytest.mark.label_name (without parentheses)
                if decorator.attr == "mark":
                    # This is just @pytest.mark, skip
                    pass
        return labels

    def _extract_skip_reason(self, decorator_list: List[ast.expr]) -> Optional[str]:
        """Extract skip reason from @pytest.mark.skip or @pytest.mark.skipif."""
        for decorator in decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute):
                    if func.attr in ["skip", "skipif"]:
                        if isinstance(func.value, ast.Attribute) and func.value.attr == "mark":
                            if isinstance(func.value.value, ast.Name) and func.value.value.id == "pytest":
                                # Look for reason in keyword arguments
                                for keyword in decorator.keywords:
                                    if keyword.arg == "reason":
                                        str_val = _get_string_value(keyword.value)
                                        if str_val is not None:
                                            return str_val
                                # Or in args
                                if decorator.args:
                                    str_val = _get_string_value(decorator.args[0])
                                    if str_val is not None:
                                        return str_val
        return None

    def _extract_fixture_prep_steps(self, node: ast.FunctionDef) -> List[TestStep]:
        """Extract preparation steps from fixtures used by this test."""
        prep_steps = []

        # Check function parameters for fixture names
        for arg in node.args.args:
            if arg.arg in self.fixtures:
                prep_steps.extend(self.fixtures[arg.arg])

        # Check for @pytest.mark.usefixtures decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute) and func.attr == "usefixtures":
                    if isinstance(func.value, ast.Attribute) and func.value.attr == "mark":
                        if isinstance(func.value.value, ast.Name) and func.value.value.id == "pytest":
                            # Extract fixture names from args
                            for arg in decorator.args:
                                fixture_name = _get_string_value(arg)
                                if fixture_name and fixture_name in self.fixtures:
                                    prep_steps.extend(self.fixtures[fixture_name])

        return prep_steps

    def _extract_class_setup_teardown(self, node: ast.ClassDef) -> tuple[List[TestStep], List[TestStep]]:
        """Extract setup and teardown steps from class methods."""
        setup_steps = []
        teardown_steps = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "setup_class" or item.name == "setup_method":
                    setup_steps.extend(self._extract_steps_from_function(item))
                elif item.name == "teardown_class" or item.name == "teardown_method":
                    teardown_steps.extend(self._extract_steps_from_function(item))

        return setup_steps, teardown_steps

    def _extract_steps_from_function(self, node: ast.FunctionDef) -> List[TestStep]:
        """Extract steps from a function body."""
        steps = []
        if not node.body:
            return steps

        for stmt in node.body:
            step_text = self._extract_step_from_stmt(stmt)
            if step_text:
                steps.append(TestStep(text=step_text))

        return steps

    def _extract_step_from_stmt(self, stmt: ast.stmt) -> Optional[str]:
        """Extract a human-readable step description from a statement."""
        if isinstance(stmt, ast.Expr):
            # Expression statement (function call, etc.)
            return self._extract_step_from_expr(stmt.value)
        elif isinstance(stmt, ast.Assert):
            # Assert statement
            return self._extract_step_from_assert(stmt)
        elif isinstance(stmt, ast.Assign):
            # Assignment - check if it's a meaningful operation
            return self._extract_step_from_assign(stmt)
        elif isinstance(stmt, ast.With):
            # Context manager (e.g., `with oc.project(...):`)
            return self._extract_step_from_with(stmt)
        elif isinstance(stmt, ast.For):
            # For loop - extract step from body
            return self._extract_step_from_for(stmt)
        elif isinstance(stmt, ast.If):
            # If statement - extract step from condition
            return self._extract_step_from_if(stmt)
        elif isinstance(stmt, ast.Return):
            # Return statement
            return None  # Usually not a meaningful step
        elif isinstance(stmt, ast.Pass):
            # Pass statement
            return None
        else:
            # Other statement types - try to extract something
            return None

    def _extract_step_from_expr(self, expr: ast.expr) -> Optional[str]:
        """Extract step description from an expression."""
        if isinstance(expr, ast.Call):
            return self._extract_step_from_call(expr)
        elif isinstance(expr, ast.Attribute):
            # Attribute access
            return None
        else:
            return None

    def _extract_step_from_call(self, call: ast.Call) -> Optional[str]:
        """Extract step description from a function call."""
        func = call.func

        # Handle oc.selector(...) patterns
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "oc":
                if func.attr == "selector":
                    if call.args:
                        arg = call.args[0]
                        resource = self._extract_string_literal(arg)
                        if resource:
                            return f"getting {resource} resources"
                    return "getting resources"
                elif func.attr in ["get", "create", "delete", "patch", "replace"]:
                    return f"{func.attr}ing resource"

        # Handle get_resource(...) patterns
        if isinstance(func, ast.Name):
            func_name = func.id
            if func_name == "get_resource":
                if call.args:
                    resource = self._extract_string_literal(call.args[0])
                    if resource:
                        return f"getting {resource} resource"
                return "getting resource"
            elif func_name == "get_resource_from_namespace":
                return "getting resource from namespace"
            elif func_name == "get_pods_list":
                return "getting pods list"
            elif func_name == "get_node_status":
                return "getting node status"
            elif func_name.startswith("test_"):
                # Test helper function call
                return f"running {func_name.replace('_', ' ')}"

        # Handle subprocess calls
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                if func.attr == "run":
                    # Extract command
                    if call.args:
                        cmd_arg = call.args[0]
                        if isinstance(cmd_arg, (ast.List, ast.Tuple)):
                            cmd_parts = []
                            for item in cmd_arg.elts:
                                if isinstance(item, ast.Constant):
                                    cmd_parts.append(str(item.value))
                                elif isinstance(item, ast.Str):  # Python < 3.8
                                    cmd_parts.append(item.s)
                            if cmd_parts:
                                cmd = " ".join(cmd_parts[:3])  # Limit length
                                return f"running {cmd} command"
                    return "running subprocess command"

        # Generic function call
        if isinstance(func, ast.Name):
            func_name = func.id
            # Convert snake_case to readable text
            readable = func_name.replace("_", " ")
            return readable
        elif isinstance(func, ast.Attribute):
            attr_name = func.attr
            readable = attr_name.replace("_", " ")
            return readable

        return None

    def _extract_step_from_assert(self, stmt: ast.Assert) -> Optional[str]:
        """Extract step description from an assert statement.
        
        Extracts the assert message/condition and converts to positive form,
        matching the Go extractor's By() style (e.g., "ensuring all clocks are LOCKED").
        """
        # First, try to extract the assert message if present
        assert_msg = None
        if stmt.msg:
            assert_msg = _get_string_value(stmt.msg)
            if assert_msg:
                # Strip any surrounding quotes and whitespace
                assert_msg = assert_msg.strip().strip("'\"")
                # Convert to positive form by removing "Fail with", "fail", etc. prefixes
                assert_msg = self._convert_to_positive_form(assert_msg)
                if assert_msg:
                    return assert_msg
        
        # If no message, extract the condition text and convert to positive form
        condition_text = self._extract_assert_condition_text(stmt.test)
        if condition_text:
            return self._convert_to_positive_form(condition_text)
        
        return "verifying condition"
    
    def _convert_to_positive_form(self, text: str) -> str:
        """Convert assert message/condition to positive form.
        
        Removes negative prefixes like "Fail with", "fail", etc. and converts
        to positive statements matching Go extractor style (e.g., "ensuring", "verifying").
        """
        if not text:
            return "verifying condition"
        
        text = text.strip()
        original_text = text
        
        # Remove common negative prefixes (case-insensitive)
        prefixes_to_remove = [
            "fail with ",
            "fail ",
            "error: ",
            "error ",
        ]
        
        text_lower = text.lower()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # If text is empty after removing prefix, return a generic positive form
        if not text:
            return "verifying condition"
        
        # Check if text already starts with a positive verb (ensuring, verifying, checking, etc.)
        positive_verbs = ["ensuring", "verifying", "checking", "waiting", "getting", "updating", "resetting", "creating"]
        text_lower = text.lower()
        if any(text_lower.startswith(verb + " ") for verb in positive_verbs):
            return text
        
        # For function calls, always add "verifying" prefix
        if "(" in text:
            return f"verifying {text}"
        
        # For descriptive messages, add appropriate verb based on content
        # If it's about checking/verifying something, use "verifying" or "ensuring"
        if any(word in text_lower for word in ["check", "verify", "validate", "test", "confirm"]):
            # If it doesn't already start with a verb, add "verifying"
            if not any(text_lower.startswith(verb) for verb in ["check", "verify", "validate", "test", "confirm"]):
                return f"verifying {text}"
            return text
        
        # For simple conditions (short, no spaces), add "verifying"
        if len(text) < 30 and " " not in text:
            return f"verifying {text}"
        
        # For longer descriptive text, try to add appropriate verb
        # If it starts with a noun or describes an action, add "ensuring" or "verifying"
        if text_lower.startswith(("health", "status", "state", "condition", "result")):
            return f"verifying {text}"
        
        # Default: add "verifying" for consistency
        return f"verifying {text}"
    
    def _extract_assert_condition_text(self, expr: ast.expr) -> Optional[str]:
        """Extract readable text from assert condition expression."""
        # Try using ast.unparse if available (Python 3.9+)
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(expr)
        except Exception:
            pass
        
        # Fallback to manual extraction
        if isinstance(expr, ast.Compare):
            # Comparison: x == y, x > y, etc.
            left = self._extract_expr_text(expr.left)
            if left:
                op_strs = []
                for i, op in enumerate(expr.ops):
                    op_text = self._op_to_text(op)
                    comparator = self._extract_expr_text(expr.comparators[i]) if i < len(expr.comparators) else None
                    if comparator:
                        op_strs.append(f"{left} {op_text} {comparator}")
                    else:
                        op_strs.append(f"{left} {op_text} ?")
                if op_strs:
                    return " and ".join(op_strs)
        
        # Try to extract as expression text
        expr_text = self._extract_expr_text(expr)
        if expr_text:
            return expr_text
        
        # Last resort: try to get string representation
        try:
            return str(expr)
        except Exception:
            return None

    def _extract_step_from_assign(self, stmt: ast.Assign) -> Optional[str]:
        """Extract step description from assignment."""
        # Only extract if it's a meaningful operation
        if isinstance(stmt.value, ast.Call):
            return self._extract_step_from_call(stmt.value)
        return None

    def _extract_step_from_with(self, stmt: ast.With) -> Optional[str]:
        """Extract step description from with statement."""
        # Extract context manager name
        for item in stmt.items:
            if isinstance(item.context_expr, ast.Call):
                return self._extract_step_from_call(item.context_expr)
            elif isinstance(item.context_expr, ast.Attribute):
                if isinstance(item.context_expr.value, ast.Name):
                    if item.context_expr.value.id == "oc":
                        if item.context_expr.attr == "project":
                            return "switching to project namespace"
        return None

    def _extract_step_from_for(self, stmt: ast.For) -> Optional[str]:
        """Extract step description from for loop."""
        # Extract what we're iterating over
        iter_text = self._extract_expr_text(stmt.iter)
        if iter_text:
            return f"iterating over {iter_text}"
        return "iterating over items"

    def _extract_step_from_if(self, stmt: ast.If) -> Optional[str]:
        """Extract step description from if statement."""
        # Extract condition
        cond_text = self._extract_expr_text(stmt.test)
        if cond_text:
            return f"checking if {cond_text}"
        return "checking condition"

    def _extract_string_literal(self, expr: ast.expr) -> Optional[str]:
        """Extract string literal from expression."""
        return _get_string_value(expr)

    def _extract_expr_text(self, expr: ast.expr) -> Optional[str]:
        """Extract readable text from expression."""
        if isinstance(expr, ast.Name):
            return expr.id
        elif isinstance(expr, ast.Attribute):
            return expr.attr
        elif isinstance(expr, ast.Call):
            return self._extract_step_from_call(expr)
        else:
            str_val = _get_string_value(expr)
            if str_val is not None:
                return str_val
        return None

    def _op_to_text(self, op: ast.cmpop) -> str:
        """Convert comparison operator to text."""
        op_map = {
            ast.Eq: "equals",
            ast.NotEq: "not equals",
            ast.Lt: "less than",
            ast.LtE: "less than or equal",
            ast.Gt: "greater than",
            ast.GtE: "greater than or equal",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
        }
        return op_map.get(type(op), "compares")

