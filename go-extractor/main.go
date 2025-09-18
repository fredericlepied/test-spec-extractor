// go-extractor/main.go
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Action struct {
	GVK       string                 `json:"gvk,omitempty"`
	Verb      string                 `json:"verb,omitempty"`
	NS        string                 `json:"ns,omitempty"`
	Selector  map[string]interface{} `json:"selector,omitempty"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
	Condition string                 `json:"condition,omitempty"`
	Count     int                    `json:"count,omitempty"`
}

type KubeSpec struct {
	TestID            string              `json:"test_id"`
	Level             string              `json:"level"`
	Preconditions     []string            `json:"preconditions"`
	Actions           []Action            `json:"actions"`
	Expectations      []map[string]string `json:"expectations"`
	Externals         []string            `json:"externals"`
	OpenShiftSpecific []string            `json:"openshift_specific"`
	Concurrency       []string            `json:"concurrency"`
	Artifacts         []string            `json:"artifacts"`
}

var (
	verbSet  = map[string]bool{"Create": true, "Update": true, "Patch": true, "Delete": true, "Get": true, "List": true, "Watch": true}
	goldenRe = regexp.MustCompile(`(?i)testdata/[^"']+`)

	psaKeys = []string{
		"pod-security.kubernetes.io/enforce",
		"pod-security.kubernetes.io/audit",
		"pod-security.kubernetes.io/warn",
		"security.openshift.io/scc",
	}
)

func aliasToGroup(alias string) (group, version string) {
	al := strings.ToLower(alias)
	switch {
	case strings.Contains(al, "appsv1"):
		return "apps", "v1"
	case strings.Contains(al, "corev1") || al == "v1":
		return "", "v1"
	case strings.Contains(al, "batchv1"):
		return "batch", "v1"
	case strings.Contains(al, "rbacv1"):
		return "rbac.authorization.k8s.io", "v1"
	case strings.Contains(al, "networkingv1"):
		return "networking.k8s.io", "v1"
	case strings.Contains(al, "routev1"):
		return "route.openshift.io", "v1"
	case strings.Contains(al, "securityv1"):
		return "security.openshift.io", "v1"
	}
	return "", ""
}

func kindFromCompositeLit(t ast.Expr) (gvk string, openshiftHint string) {
	if u, ok := t.(*ast.UnaryExpr); ok {
		return kindFromCompositeLit(u.X)
	}
	se, ok := t.(*ast.CompositeLit)
	if !ok {
		return "", ""
	}
	ts := se.Type
	sel, ok := ts.(*ast.SelectorExpr)
	if !ok {
		return "", ""
	}
	pkgIdent, ok := sel.X.(*ast.Ident)
	if !ok {
		return "", ""
	}
	group, version := aliasToGroup(pkgIdent.Name)
	if version == "" {
		return "", ""
	}
	kind := sel.Sel.Name
	var g string
	if group != "" {
		g = fmt.Sprintf("%s/%s/%s", group, version, kind)
	} else {
		g = fmt.Sprintf("%s/%s", version, kind) // core/v1/Pod -> v1/Pod
	}
	if strings.Contains(group, "openshift") || strings.Contains(group, "route.openshift.io") {
		return g, g
	}
	return g, ""
}

func collectPSALabels(m ast.Expr) (labels []string) {
	cl, ok := m.(*ast.CompositeLit)
	if !ok {
		return
	}
	for _, elt := range cl.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		var k string
		switch kt := kv.Key.(type) {
		case *ast.Ident:
			k = kt.Name
		case *ast.BasicLit:
			k = strings.Trim(kt.Value, "\"")
		}
		if bl, ok := kv.Value.(*ast.BasicLit); ok && bl.Kind == token.STRING {
			v := strings.Trim(bl.Value, "\"")
			for _, pk := range psaKeys {
				if k == pk {
					labels = append(labels, fmt.Sprintf("%s=%s", pk, v))
				}
			}
		}
		if sub, ok := kv.Value.(*ast.CompositeLit); ok {
			labels = append(labels, collectPSALabels(sub)...)
		}
	}
	return
}

func main() {
	root := flag.String("root", ".", "root of the Go repo to scan")
	flag.Parse()

	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	_ = filepath.WalkDir(*root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, "_test.go") {
			return nil
		}

		fset := token.NewFileSet()
		f, perr := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if perr != nil {
			return nil
		}

		fileGolden := []string{}
		if f.Comments != nil {
			for _, cg := range f.Comments {
				if m := goldenRe.FindString(cg.Text()); m != "" {
					fileGolden = append(fileGolden, m)
				}
			}
		}

		ast.Inspect(f, func(n ast.Node) bool {
			fd, ok := n.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || fd.Name == nil || !strings.HasPrefix(fd.Name.Name, "Test") {
				return true
			}

			spec := KubeSpec{Level: "unknown"}
			spec.TestID = fmt.Sprintf("%s:%s", path, fd.Name.Name)
			spec.Artifacts = append(spec.Artifacts, fileGolden...)

			openshiftSet := map[string]bool{}
			verbs := map[string]bool{}

			ast.Inspect(fd, func(n2 ast.Node) bool {
				switch x := n2.(type) {
				case *ast.CallExpr:
					if sel, ok := x.Fun.(*ast.SelectorExpr); ok {
						if verbSet[sel.Sel.Name] {
							verbs[sel.Sel.Name] = true
						}
						if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "t" && sel.Sel.Name == "Parallel" {
							spec.Concurrency = append(spec.Concurrency, "parallel")
						}
					}
					for _, a := range x.Args {
						if bl, ok := a.(*ast.BasicLit); ok && bl.Kind == token.STRING {
							if m := goldenRe.FindString(bl.Value); m != "" {
								spec.Artifacts = append(spec.Artifacts, strings.Trim(bl.Value, "\""))
							}
						}
					}
				case *ast.CompositeLit:
					if gvk, osh := kindFromCompositeLit(x); gvk != "" {
						spec.Actions = append(spec.Actions, Action{GVK: gvk})
						if osh != "" {
							openshiftSet[osh] = true
						}
						if strings.Contains(strings.ToLower(gvk), "/namespace") {
							if lbs := collectPSALabels(x); len(lbs) > 0 {
								for _, l := range lbs {
									spec.Preconditions = append(spec.Preconditions, "psa:"+l)
								}
							}
						}
					}
				}
				return true
			})

			for v := range verbs {
				spec.Actions = append(spec.Actions, Action{Verb: strings.ToLower(v)})
			}
			for k := range openshiftSet {
				spec.OpenShiftSpecific = append(spec.OpenShiftSpecific, k)
			}
			if verbs["Create"] || verbs["Delete"] || verbs["Watch"] {
				spec.Level = "integration"
			}

			bridges := map[string]bool{}
			for _, a := range spec.Actions {
				g := strings.ToLower(a.GVK)
				if strings.Contains(g, "route.openshift.io") && strings.Contains(g, "/route") {
					bridges["equiv:route~ingress"] = true
				}
				if strings.Contains(g, "networking.k8s.io") && strings.Contains(g, "/ingress") {
					bridges["equiv:route~ingress"] = true
				}
				if strings.Contains(g, "security.openshift.io") {
					bridges["equiv:scc~psa"] = true
				}
			}
			for _, p := range spec.Preconditions {
				if strings.HasPrefix(p, "psa:") {
					bridges["equiv:scc~psa"] = true
				}
			}
			for k := range bridges {
				spec.Preconditions = append(spec.Preconditions, k)
			}

			b, _ := json.Marshal(spec)
			fmt.Fprintln(out, string(b))
			return true
		})
		return nil
	})
}
