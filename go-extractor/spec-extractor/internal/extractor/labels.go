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
			// First try to extract as string literal
			if lit, ok := la.(*ast.BasicLit); ok {
				value := unquote(lit.Value)
				if value != "" {
					labels = append(labels, value)
				}
			} else if ident, ok := la.(*ast.Ident); ok {
				// Try to resolve as constant
				if value, found := cr.Resolve(ident.Name); found && value != "" {
					labels = append(labels, value)
				}
			}
		}
	}
	return labels
}
