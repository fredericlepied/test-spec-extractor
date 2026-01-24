package extractor

import (
	"go/ast"
	"regexp"
	"strings"
)

// ConstantResolver resolves string constants from the AST
type ConstantResolver struct {
	constants map[string]string
}

// NewConstantResolver creates a new constant resolver
func NewConstantResolver(res *FileResult, goModPath string) *ConstantResolver {
	cr := &ConstantResolver{
		constants: make(map[string]string),
	}

	// Walk the AST to find constant declarations
	if res.AST != nil {
		ast.Inspect(res.AST, func(n ast.Node) bool {
			// Look for const declarations
			if spec, ok := n.(*ast.ValueSpec); ok {
				for i, name := range spec.Names {
					if i < len(spec.Values) {
						if lit, ok := spec.Values[i].(*ast.BasicLit); ok {
							cr.constants[name.Name] = lit.Value
						}
					}
				}
			}
			return true
		})
	}

	return cr
}

// Resolve attempts to resolve a constant name to its value
func (cr *ConstantResolver) Resolve(name string) (string, bool) {
	val, ok := cr.constants[name]
	if !ok {
		return "", false
	}
	return unquote(val), true
}

// ParsedTestDescription holds the result of parsing a test description
type ParsedTestDescription struct {
	Description string
	TestID      string
	Labels      []string
}

// ParseTestDescription parses a test description to extract embedded labels and test IDs
func ParseTestDescription(desc string) ParsedTestDescription {
	result := ParsedTestDescription{
		Description: desc,
		Labels:      []string{},
	}

	// Extract test ID from format like "[test_id:12345]" or "(test_id:12345)"
	testIDPattern := regexp.MustCompile(`[\[\(]test_id:([^\]\)]+)[\]\)]`)
	if matches := testIDPattern.FindStringSubmatch(desc); len(matches) > 1 {
		result.TestID = strings.TrimSpace(matches[1])
		// Remove the test ID from description
		result.Description = testIDPattern.ReplaceAllString(desc, "")
		result.Description = strings.TrimSpace(result.Description)
	}

	// Extract labels from format like "[label1][label2]"
	labelPattern := regexp.MustCompile(`\[([^\]]+)\]`)
	labelMatches := labelPattern.FindAllStringSubmatch(desc, -1)
	for _, match := range labelMatches {
		if len(match) > 1 {
			label := strings.TrimSpace(match[1])
			// Skip if it's a test_id
			if !strings.HasPrefix(label, "test_id:") && label != "" {
				result.Labels = append(result.Labels, label)
			}
		}
	}

	// Clean up description by removing label brackets
	result.Description = labelPattern.ReplaceAllStringFunc(result.Description, func(s string) string {
		// Keep test_id brackets, remove others
		if strings.Contains(s, "test_id:") {
			return ""
		}
		return ""
	})
	result.Description = strings.TrimSpace(result.Description)

	return result
}
