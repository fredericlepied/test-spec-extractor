package extractor

import (
	"go/ast"
	"regexp"
	"strings"
)

// extractTestID attempts to extract a test ID from various sources
// Priority order:
// 1. reportxml.ID("12345") call in the test body
// 2. ID embedded in test description
// 3. Return empty string if no ID found
func extractTestID(recog *Recognizer, constResolver *ConstantResolver, call *ast.CallExpr) string {
	// Get the test body (function literal)
	fn := firstFuncLit(call)
	if fn == nil || fn.Body == nil {
		return ""
	}

	// Search for reportxml.ID() calls in the test body
	var testID string
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if testID != "" {
			return false // Already found, stop searching
		}

		callExpr, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// Check if this is reportxml.ID() or similar
		if sel, ok := callExpr.Fun.(*ast.SelectorExpr); ok {
			// Check for reportxml.ID pattern
			if sel.Sel.Name == "ID" {
				if pkg, ok := sel.X.(*ast.Ident); ok {
					if pkg.Name == "reportxml" || pkg.Name == "polarion" {
						// Extract the ID argument
						if len(callExpr.Args) > 0 {
							if lit, ok := callExpr.Args[0].(*ast.BasicLit); ok {
								testID = unquote(lit.Value)
								return false
							}
							// Try to resolve as constant
							if ident, ok := callExpr.Args[0].(*ast.Ident); ok {
								if val, found := constResolver.Resolve(ident.Name); found {
									testID = val
									return false
								}
							}
						}
					}
				}
			}
		}

		return true
	})

	return testID
}

// extractTestIDFromDescription extracts test ID from description text
func extractTestIDFromDescription(desc string) string {
	// Match patterns like [test_id:12345] or (test_id:12345)
	patterns := []string{
		`\[test_id:([^\]]+)\]`,
		`\(test_id:([^\)]+)\)`,
		`test_id:(\S+)`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindStringSubmatch(desc); len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	}

	return ""
}
