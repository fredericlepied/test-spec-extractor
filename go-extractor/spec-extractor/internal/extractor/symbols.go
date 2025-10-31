package extractor

import (
	"go/ast"
)

// Recognizer determines whether a call refers to a Ginkgo construct, honoring aliases.
type Recognizer struct {
	// key: canonical name (Describe, Context, When, It, By, BeforeEach, JustBeforeEach, BeforeAll, BeforeSuite, Label)
	// value: allowed names including aliases (e.g., K8sDescribe) and selector prefixes from imports (e.g., ginkgo.Describe)
	allowed map[string]map[string]struct{}
}

func NewRecognizer(imports map[string]string, cliAliases map[string][]string) *Recognizer {
	r := &Recognizer{allowed: map[string]map[string]struct{}{}}
	add := func(canon string, names ...string) {
		m := r.allowed[canon]
		if m == nil {
			m = map[string]struct{}{}
			r.allowed[canon] = m
		}
		for _, n := range names {
			m[n] = struct{}{}
		}
	}

	// base names (unqualified)
	add("Describe", "Describe")
	add("Context", "Context")
	add("When", "When")
	add("It", "It")
	add("By", "By")
	add("BeforeEach", "BeforeEach")
	add("JustBeforeEach", "JustBeforeEach")
	add("BeforeAll", "BeforeAll")
	add("BeforeSuite", "BeforeSuite")
	add("Label", "Label")

	// qualified ginkgo.* if imported under an alias
	for local, full := range imports {
		if full == "github.com/onsi/ginkgo/v2" || full == "github.com/onsi/ginkgo" {
			for canon, base := range map[string]string{
				"Describe":       "Describe",
				"Context":        "Context",
				"When":           "When",
				"It":             "It",
				"By":             "By",
				"BeforeEach":     "BeforeEach",
				"JustBeforeEach": "JustBeforeEach",
				"BeforeAll":      "BeforeAll",
				"BeforeSuite":    "BeforeSuite",
				"Label":          "Label",
			} {
				add(canon, local+"."+base)
			}
		}
	}

	// CLI-provided aliases
	for canon, names := range cliAliases {
		add(canon, names...)
	}

	return r
}

func (r *Recognizer) funName(n ast.Expr) string {
	switch x := n.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.SelectorExpr:
		// convert to pkg.Func form for matching
		if pkg, ok := x.X.(*ast.Ident); ok {
			return pkg.Name + "." + x.Sel.Name
		}
		return x.Sel.Name
	default:
		return ""
	}
}

func (r *Recognizer) is(canon string, call *ast.CallExpr) bool {
	if call == nil || call.Fun == nil {
		return false
	}
	name := r.funName(call.Fun)
	if name == "" {
		return false
	}
	allowed, ok := r.allowed[canon]
	if !ok {
		return false
	}
	if _, ok := allowed[name]; ok {
		return true
	}
	return false
}

func (r *Recognizer) IsContainer(call *ast.CallExpr) (string, bool) {
	switch {
	case r.is("Describe", call):
		return "Describe", true
	case r.is("Context", call):
		return "Context", true
	case r.is("When", call):
		return "When", true
	default:
		return "", false
	}
}

func (r *Recognizer) IsIt(call *ast.CallExpr) bool { return r.is("It", call) }
func (r *Recognizer) IsBy(call *ast.CallExpr) bool { return r.is("By", call) }
func (r *Recognizer) IsBefore(call *ast.CallExpr) (string, bool) {
	switch {
	case r.is("BeforeEach", call):
		return "BeforeEach", true
	case r.is("JustBeforeEach", call):
		return "JustBeforeEach", true
	case r.is("BeforeAll", call):
		return "BeforeAll", true
	case r.is("BeforeSuite", call):
		return "BeforeSuite", true
	default:
		return "", false
	}
}
func (r *Recognizer) IsLabel(call *ast.CallExpr) bool { return r.is("Label", call) }
