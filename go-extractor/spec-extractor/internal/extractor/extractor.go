package extractor

import (
	"go/ast"
	"go/parser"
	"go/token"
)

// FileResult holds minimal parse information; more fields will be added in later steps.
type FileResult struct {
	FilePath  string
	Package   string
	ImportMap map[string]string // local import name -> full path
	AST       *ast.File
	FileSet   *token.FileSet
}

// ParseFile parses a single Go file, returning the AST and an import alias map.
func ParseFile(path string) (*FileResult, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	imports := map[string]string{}
	for _, imp := range file.Imports {
		pathLit := ""
		if imp.Path != nil {
			pathLit = trimQuotes(imp.Path.Value)
		}
		local := ""
		if imp.Name != nil {
			local = imp.Name.Name
		} else {
			// derive default local name from last path segment
			local = defaultLocalName(pathLit)
		}
		if local != "" {
			imports[local] = pathLit
		}
	}
	return &FileResult{
		FilePath:  path,
		Package:   file.Name.Name,
		ImportMap: imports,
		AST:       file,
		FileSet:   fset,
	}, nil
}

func trimQuotes(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '`' && s[len(s)-1] == '`') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func defaultLocalName(importPath string) string {
	// naive: take last segment after '/'
	last := importPath
	for i := len(importPath) - 1; i >= 0; i-- {
		if importPath[i] == '/' {
			last = importPath[i+1:]
			break
		}
	}
	if last == "" {
		return ""
	}
	return last
}
