package extractor

import (
	"go/ast"
	"go/token"
)

// BuildFileSpec walks the AST and builds a high-level spec tree.
func BuildFileSpec(res *FileResult, cliAliases map[string][]string, goModPath string) *FileSpec {
	recog := NewRecognizer(res.ImportMap, cliAliases)
	constResolver := NewConstantResolver(res, goModPath)
	root := &Container{Kind: "Root"}
	v := &visitor{
		recog:         recog,
		constResolver: constResolver,
		containerSt:   []*Container{root},
		fset:          res.FileSet,
	}
	ast.Inspect(res.AST, v.visit)
	return &FileSpec{FilePath: res.FilePath, Root: root}
}

type visitor struct {
	recog         *Recognizer
	constResolver *ConstantResolver
	containerSt   []*Container
	fset          *token.FileSet
}

func (v *visitor) current() *Container { return v.containerSt[len(v.containerSt)-1] }

func (v *visitor) visit(n ast.Node) bool {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return true
	}

	if kind, isCont := v.recog.IsContainer(call); isCont {
		desc := firstStringArg(call)
		c := &Container{Kind: kind, Description: desc}
		c.Labels = append(c.Labels, extractLabels(v.recog, v.constResolver, call)...)
		// push container and walk its body (usually last arg is func literal)
		parent := v.current()
		parent.Children = append(parent.Children, c)
		v.containerSt = append(v.containerSt, c)
		// Continue traversal; do not short-circuit, as ast.Inspect will handle children
		return true
	}

	if _, isBefore := v.recog.IsBefore(call); isBefore {
		if fn := firstFuncLit(call); fn != nil {
			// Track variable assignments to resolve Skip(variableName) calls
			varAssignments := make(map[string]string)
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				// Track simple string variable assignments
				if assign, ok := n.(*ast.AssignStmt); ok {
					for i, lhs := range assign.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok && i < len(assign.Rhs) {
							if bl, ok := assign.Rhs[i].(*ast.BasicLit); ok && bl.Kind == token.STRING {
								varAssignments[ident.Name] = unquote(bl.Value)
							}
						}
					}
				}

				if be, ok := n.(*ast.CallExpr); ok {
					if v.recog.IsBy(be) {
						if s := firstStringArg(be); s != "" {
							v.current().PrepSteps = append(v.current().PrepSteps, TestStep{Text: s})
						}
					} else if v.recog.IsSkip(be) {
						// Skip calls in Before* blocks will be transformed into separate test entries
						s := firstStringArg(be)
						if s == "" {
							// Try to resolve from variable
							s = firstVarArg(be, varAssignments)
						}
						if s != "" {
							v.current().SkipConditions = append(v.current().SkipConditions, TestStep{Text: s})
						}
					} else if info := parseExpectAssertion(be, v.recog); info != nil {
						// Extract Expect assertions from Before* blocks as preparation validations
						if info.Description != "" {
							v.current().PrepSteps = append(v.current().PrepSteps, TestStep{Text: info.Description})
						}
					}
				}
				return true
			})
		}
		return true
	}

	if _, isAfter := v.recog.IsAfter(call); isAfter {
		if fn := firstFuncLit(call); fn != nil {
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if be, ok := n.(*ast.CallExpr); ok && v.recog.IsBy(be) {
					if s := firstStringArg(be); s != "" {
						v.current().CleanupSteps = append(v.current().CleanupSteps, TestStep{Text: s})
					}
				}
				return true
			})
		}
		return true
	}

	if v.recog.IsIt(call) {
		desc := firstStringArg(call)

		// Parse test description to extract embedded labels and test ID
		parsed := ParseTestDescription(desc)

		tc := TestCase{
			Description: parsed.Description,
			LineNumber:  v.fset.Position(call.Pos()).Line,
		}

		// Add labels from Label() calls
		tc.Labels = append(tc.Labels, extractLabels(v.recog, v.constResolver, call)...)

		// Add labels extracted from description (if any)
		if len(parsed.Labels) > 0 {
			tc.Labels = append(tc.Labels, parsed.Labels...)
		}

		// Set test ID (from embedded pattern or reportxml.ID())
		if parsed.TestID != "" {
			tc.TestID = parsed.TestID
		} else {
			tc.TestID = extractTestID(v.recog, v.constResolver, call)
		}

		// Collect By steps inside the It body by visiting its function literal argument
		if fn := firstFuncLit(call); fn != nil {
			// Track variable assignments to resolve Skip(variableName) calls
			varAssignments := make(map[string]string)
			// Walk only the body to collect By steps
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				// Track simple string variable assignments
				if assign, ok := n.(*ast.AssignStmt); ok {
					for i, lhs := range assign.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok && i < len(assign.Rhs) {
							if bl, ok := assign.Rhs[i].(*ast.BasicLit); ok && bl.Kind == token.STRING {
								varAssignments[ident.Name] = unquote(bl.Value)
							}
						}
					}
				}

				if be, ok := n.(*ast.CallExpr); ok {
					// Check for recognized By() calls (g.By, ginkgo.By, etc.) or custom wrappers (compat_otp.By, etc.)
					if v.recog.IsBy(be) || isCustomByCall(be) {
						if s := firstStringArg(be); s != "" {
							tc.Steps = append(tc.Steps, TestStep{Text: s})
						}
					} else if v.recog.IsFail(be) {
						// Fail messages go to cleanup section
						if s := firstStringArg(be); s != "" {
							tc.CleanupSteps = append(tc.CleanupSteps, TestStep{Text: s})
						}
					} else if v.recog.IsSkip(be) {
						// Skip messages will be transformed into separate test entries
						s := firstStringArg(be)
						if s == "" {
							// Try to resolve from variable
							s = firstVarArg(be, varAssignments)
						}
						if s != "" {
							tc.SkipConditions = append(tc.SkipConditions, TestStep{Text: s})
						}
					} else if info := parseExpectAssertion(be, v.recog); info != nil {
						// Extract and generate descriptions from Expect assertions
						if info.Description != "" {
							tc.Validations = append(tc.Validations, TestStep{Text: info.Description})
						}
					}
				}
				return true
			})
		}
		v.current().Cases = append(v.current().Cases, tc)
		return true
	}

	// Handle Entry calls for table-driven tests
	if v.recog.IsEntry(call) {
		desc := firstStringArg(call)
		if desc != "" {
			// Parse test description to extract embedded labels and test ID
			parsed := ParseTestDescription(desc)

			tc := TestCase{
				Description: parsed.Description,
				LineNumber:  v.fset.Position(call.Pos()).Line,
			}

			// Add labels from Label() calls
			tc.Labels = append(tc.Labels, extractLabels(v.recog, v.constResolver, call)...)

			// Add labels extracted from description (if any)
			if len(parsed.Labels) > 0 {
				tc.Labels = append(tc.Labels, parsed.Labels...)
			}

			// Set test ID (from embedded pattern or reportxml.ID())
			if parsed.TestID != "" {
				tc.TestID = parsed.TestID
			} else {
				tc.TestID = extractTestID(v.recog, v.constResolver, call)
			}

			v.current().Cases = append(v.current().Cases, tc)
		}
		return true
	}

	// Handle DeferCleanup calls that might contain descriptions
	if v.recog.IsDeferCleanup(call) {
		if s := firstStringArg(call); s != "" {
			v.current().CleanupSteps = append(v.current().CleanupSteps, TestStep{Text: s})
		}
		return true
	}

	// Standalone By calls at top-level are ignored; By is handled inside It/Before/After bodies above.

	return true
}

// isCustomByCall checks if a call is a custom By() wrapper (e.g., compat_otp.By())
// that the Recognizer doesn't know about. This provides generic pattern matching
// for any package.By() call with a single string argument, working across any test suite.
func isCustomByCall(call *ast.CallExpr) bool {
	// Check if this is a selector expression (package.Method)
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}

	// Check if the method name is "By"
	if sel.Sel.Name != "By" {
		return false
	}

	// Verify it has exactly one string argument
	if len(call.Args) != 1 {
		return false
	}

	// Check if the argument is a string literal
	if bl, ok := call.Args[0].(*ast.BasicLit); ok && bl.Kind == token.STRING {
		return true
	}

	return false
}

func firstStringArg(call *ast.CallExpr) string {
	for _, a := range call.Args {
		if bl, ok := a.(*ast.BasicLit); ok && bl.Kind == token.STRING {
			return unquote(bl.Value)
		}
	}
	return ""
}

func firstVarArg(call *ast.CallExpr, varAssignments map[string]string) string {
	for _, a := range call.Args {
		if ident, ok := a.(*ast.Ident); ok {
			if val, found := varAssignments[ident.Name]; found {
				return val
			}
		}
	}
	return ""
}

func firstFuncLit(call *ast.CallExpr) *ast.FuncLit {
	for _, a := range call.Args {
		if fn, ok := a.(*ast.FuncLit); ok {
			return fn
		}
	}
	return nil
}
