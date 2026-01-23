package extractor

import (
	"go/ast"
)

func extractLabels(r *Recognizer, cr *ConstantResolver, call *ast.CallExpr) []string {
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
			// Use constant resolver to handle both literals and constant references
			if value := cr.Resolve(la); value != "" {
				labels = append(labels, value)
			}
		}
	}
	return labels
}
