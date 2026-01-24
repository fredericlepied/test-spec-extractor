"""Fixture registry for tracking pytest fixtures across conftest.py files."""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional

from .types import TestStep


class FixtureRegistry:
    """Registry for tracking pytest fixtures from conftest.py files."""

    def __init__(self):
        # fixture_name -> (steps, scope, file_path)
        self.fixtures: Dict[str, tuple[List[TestStep], str, str]] = {}

    def discover_conftest_files(self, root_path: str) -> List[str]:
        """Find all conftest.py files in the directory tree."""
        conftest_files = []
        root = Path(root_path)
        for conftest in root.rglob("conftest.py"):
            conftest_files.append(str(conftest))
        return conftest_files

    def parse_conftest(self, file_path: str) -> None:
        """Parse a conftest.py file and extract fixtures."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=file_path)
        except (SyntaxError, OSError):
            return

        # Walk AST to find fixture definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_pytest_fixture(node):
                    fixture_name = node.name
                    scope = self._extract_fixture_scope(node)
                    steps = self._extract_fixture_steps(node)
                    self.fixtures[fixture_name] = (steps, scope, file_path)

    def get_fixture(self, name: str) -> Optional[List[TestStep]]:
        """Get fixture steps by name."""
        if name in self.fixtures:
            steps, _, _ = self.fixtures[name]
            return steps
        return None

    def load_fixtures_from_root(self, root_path: str) -> None:
        """Discover and load all fixtures from conftest.py files in root path."""
        conftest_files = self.discover_conftest_files(root_path)
        for conftest_file in conftest_files:
            self.parse_conftest(conftest_file)

    def _is_pytest_fixture(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a pytest fixture."""
        for decorator in node.decorator_list:
            # @pytest.fixture
            if isinstance(decorator, ast.Attribute):
                if decorator.attr == "fixture":
                    if isinstance(decorator.value, ast.Name) and decorator.value.id == "pytest":
                        return True
            # @pytest.fixture(...)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr == "fixture":
                        if (
                            isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == "pytest"
                        ):
                            return True
        return False

    def _extract_fixture_scope(self, node: ast.FunctionDef) -> str:
        """Extract fixture scope from decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "fixture":
                    # Look for scope in keyword arguments
                    for keyword in decorator.keywords:
                        if keyword.arg == "scope":
                            if isinstance(keyword.value, ast.Constant):
                                return str(keyword.value.value)
        return "function"  # default scope

    def _extract_fixture_steps(self, node: ast.FunctionDef) -> List[TestStep]:
        """Extract meaningful steps from fixture function."""
        steps = []

        # Get docstring as a description
        docstring = ast.get_docstring(node)
        if docstring:
            # Use first line of docstring
            first_line = docstring.split("\n")[0].strip()
            if first_line:
                steps.append(TestStep(text=f"Setup: {first_line}"))

        # Extract key operations from function body
        for stmt in node.body:
            step_text = self._extract_operation_from_stmt(stmt)
            if step_text:
                steps.append(TestStep(text=step_text))

        # If no specific steps found, use generic fixture name
        if not steps:
            readable_name = node.name.replace("_", " ")
            steps.append(TestStep(text=f"Setup fixture: {readable_name}"))

        return steps

    def _extract_operation_from_stmt(self, stmt: ast.stmt) -> Optional[str]:
        """Extract human-readable operation from statement."""
        # Look for yield statements (fixture setup/teardown boundary)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
            return None  # Yield marks boundary, not an operation

        # Look for function calls that indicate setup operations
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func_name = self._get_func_name(call.func)
            if func_name:
                # Common setup operations
                setup_verbs = {
                    "create": "creating",
                    "deploy": "deploying",
                    "setup": "setting up",
                    "configure": "configuring",
                    "initialize": "initializing",
                    "install": "installing",
                    "start": "starting",
                    "enable": "enabling",
                }
                for verb, gerund in setup_verbs.items():
                    if verb in func_name.lower():
                        resource = func_name.replace(verb, "").replace("_", " ").strip()
                        return f"{gerund} {resource}" if resource else f"{gerund}"

                # Generic function call
                readable = func_name.replace("_", " ")
                return f"executing {readable}"

        # Assignment that might be important
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call):
                func_name = self._get_func_name(stmt.value.func)
                if func_name:
                    readable = func_name.replace("_", " ")
                    return f"preparing {readable}"

        return None

    def _get_func_name(self, func_node: ast.expr) -> Optional[str]:
        """Extract function name from AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None


def _get_string_value(expr: ast.expr) -> Optional[str]:
    """Extract string value from AST expression."""
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return expr.value
    elif hasattr(ast, "Str") and isinstance(expr, ast.Str):
        return expr.s
    return None
