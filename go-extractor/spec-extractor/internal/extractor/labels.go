package extractor

import (
	"go/ast"
	"go/token"
)

func extractLabels(r *Recognizer, call *ast.CallExpr) []string {
	labels := []string{}
	for _, arg := range call.Args {
		nested, ok := arg.(*ast.CallExpr)
		if !ok {
			continue
		}
		if !r.IsLabel(nested) {
			continue
		}
		for _, la := range nested.Args {
			if bl, ok := la.(*ast.BasicLit); ok && bl.Kind == token.STRING {
				labels = append(labels, unquote(bl.Value))
			}
		}
	}
	return labels
}
