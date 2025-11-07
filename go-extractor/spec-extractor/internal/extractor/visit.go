package extractor

import (
	"go/ast"
	"go/token"
	"strconv"
)

// BuildFileSpec walks the AST and builds a high-level spec tree.
func BuildFileSpec(res *FileResult, cliAliases map[string][]string) *FileSpec {
	recog := NewRecognizer(res.ImportMap, cliAliases)
	root := &Container{Kind: "Root"}
	v := &visitor{
		recog:       recog,
		containerSt: []*Container{root},
	}
	ast.Inspect(res.AST, v.visit)
	return &FileSpec{FilePath: res.FilePath, Root: root}
}

type visitor struct {
	recog       *Recognizer
	containerSt []*Container
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
		c.Labels = append(c.Labels, extractLabels(v.recog, call)...)
		// push container and walk its body (usually last arg is func literal)
		parent := v.current()
		parent.Children = append(parent.Children, c)
		v.containerSt = append(v.containerSt, c)
		// Continue traversal; do not short-circuit, as ast.Inspect will handle children
		return true
	}

	if _, isBefore := v.recog.IsBefore(call); isBefore {
		if fn := firstFuncLit(call); fn != nil {
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if be, ok := n.(*ast.CallExpr); ok && v.recog.IsBy(be) {
					if s := firstStringArg(be); s != "" {
						v.current().PrepSteps = append(v.current().PrepSteps, TestStep{Text: s})
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
		tc := TestCase{Description: desc}
		tc.Labels = append(tc.Labels, extractLabels(v.recog, call)...)
		// Collect By steps inside the It body by visiting its function literal argument
		if fn := firstFuncLit(call); fn != nil {
			// Walk only the body to collect By steps
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if be, ok := n.(*ast.CallExpr); ok {
					if v.recog.IsBy(be) {
						if s := firstStringArg(be); s != "" {
							tc.Steps = append(tc.Steps, TestStep{Text: s})
						}
					} else if v.recog.IsFail(be) {
						// Fail messages go to cleanup section
						if s := firstStringArg(be); s != "" {
							tc.CleanupSteps = append(tc.CleanupSteps, TestStep{Text: s})
						}
					} else if v.recog.IsSkip(be) {
						// Skip messages go to test case prep steps as negative prerequisites
						if s := firstStringArg(be); s != "" {
							tc.PrepSteps = append(tc.PrepSteps, TestStep{Text: "SKIP: " + s})
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
			tc := TestCase{Description: desc}
			tc.Labels = append(tc.Labels, extractLabels(v.recog, call)...)
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

func firstStringArg(call *ast.CallExpr) string {
	for _, a := range call.Args {
		if bl, ok := a.(*ast.BasicLit); ok && bl.Kind == token.STRING {
			return unquote(bl.Value)
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

func unquote(s string) string {
	if s == "" {
		return s
	}
	if uq, err := strconv.Unquote(s); err == nil {
		return uq
	}
	return s
}
