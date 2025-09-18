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

	// eco-goinfra patterns
	ecoGoinfraBuilders = map[string]string{
		"pod.NewBuilder":        "v1/Pod",
		"deployment.NewBuilder": "apps/v1/Deployment",
		"namespace.NewBuilder":  "v1/Namespace",
		"service.NewBuilder":    "v1/Service",
		"route.NewBuilder":      "route.openshift.io/v1/Route",
		"nad.NewBuilder":        "k8s.cni.cncf.io/v1/NetworkAttachmentDefinition",
		"clusterversion.Pull":   "config.openshift.io/v1/ClusterVersion",
		"clusteroperator.List":  "config.openshift.io/v1/ClusterOperator",
	}

	// eco-goinfra operation patterns
	ecoGoinfraOperations = map[string]string{
		"deleteres.Namespace":  "v1/Namespace",
		"deleteres.Pod":        "v1/Pod",
		"deleteres.Deployment": "apps/v1/Deployment",
		"deleteres.Service":    "v1/Service",
		"deleteres.Route":      "route.openshift.io/v1/Route",
		"nodes.List":           "v1/Node",
		"pod.List":             "v1/Pod",
		"deployment.List":      "apps/v1/Deployment",
		"service.List":         "v1/Service",
		"route.List":           "route.openshift.io/v1/Route",
		"namespace.List":       "v1/Namespace",
		"clusterversion.Pull":  "config.openshift.io/v1/ClusterVersion",
		"clusteroperator.List": "config.openshift.io/v1/ClusterOperator",
	}

	// Kubernetes API type patterns
	k8sTypePatterns = map[string]string{
		"corev1.":       "v1/",
		"appsv1.":       "apps/v1/",
		"batchv1.":      "batch/v1/",
		"rbacv1.":       "rbac.authorization.k8s.io/v1/",
		"networkingv1.": "networking.k8s.io/v1/",
		"routev1.":      "route.openshift.io/v1/",
		"securityv1.":   "security.openshift.io/v1/",
		"metav1.":       "meta/v1/",
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

func detectEcoGoinfraPattern(call *ast.CallExpr) (gvk string, verb string) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return "", ""
	}

	// Check if it's a method call on a builder
	if _, ok := sel.X.(*ast.Ident); ok {
		// Look for patterns like podBuilder.Create(), namespaceBuilder.Delete()
		if verbSet[sel.Sel.Name] {
			return "", sel.Sel.Name
		}
	}

	// Check if it's a method call on a selector (e.g., pod.NewBuilder().Create())
	if sel2, ok := sel.X.(*ast.SelectorExpr); ok {
		if ident, ok := sel2.X.(*ast.Ident); ok {
			callName := fmt.Sprintf("%s.%s", ident.Name, sel2.Sel.Name)
			if gvk, exists := ecoGoinfraBuilders[callName]; exists {
				if verbSet[sel.Sel.Name] {
					return gvk, sel.Sel.Name
				}
				return gvk, ""
			}
		}
	}

	// Check if it's a direct call like pod.NewBuilder()
	if ident, ok := sel.X.(*ast.Ident); ok {
		callName := fmt.Sprintf("%s.%s", ident.Name, sel.Sel.Name)
		if gvk, exists := ecoGoinfraBuilders[callName]; exists {
			return gvk, ""
		}
	}

	// Check for eco-goinfra operation patterns like deleteres.Namespace(), nodes.List()
	if ident, ok := sel.X.(*ast.Ident); ok {
		callName := fmt.Sprintf("%s.%s", ident.Name, sel.Sel.Name)
		if gvk, exists := ecoGoinfraOperations[callName]; exists {
			// Determine verb based on the operation
			if strings.HasPrefix(ident.Name, "deleteres") {
				return gvk, "delete"
			} else if sel.Sel.Name == "List" {
				return gvk, "list"
			} else if sel.Sel.Name == "Pull" {
				return gvk, "get"
			}
			return gvk, "unknown"
		}
	}

	// Check for builder pattern method calls (e.g., nfdNsBuilder.Create(), metallbenv.CreateNewMetalLbDaemonSetAndWaitUntilItsRunning())
	if ident, ok := sel.X.(*ast.Ident); ok {
		// Look for patterns like *Builder.Create(), *Builder.Delete(), *Builder.Update()
		if strings.HasSuffix(ident.Name, "Builder") || strings.HasSuffix(ident.Name, "CSV") || strings.HasSuffix(ident.Name, "CR") {
			if verbSet[sel.Sel.Name] {
				// Try to determine GVK from the builder name
				gvk := ""
				if strings.Contains(ident.Name, "Ns") || strings.Contains(ident.Name, "Namespace") {
					gvk = "v1/Namespace"
				} else if strings.Contains(ident.Name, "Pod") {
					gvk = "v1/Pod"
				} else if strings.Contains(ident.Name, "Deployment") {
					gvk = "apps/v1/Deployment"
				} else if strings.Contains(ident.Name, "Service") {
					gvk = "v1/Service"
				} else if strings.Contains(ident.Name, "Route") {
					gvk = "route.openshift.io/v1/Route"
				} else if strings.Contains(ident.Name, "CSV") {
					gvk = "operators.coreos.com/v1alpha1/ClusterServiceVersion"
				} else if strings.Contains(ident.Name, "Sub") || strings.Contains(ident.Name, "Subscription") {
					gvk = "operators.coreos.com/v1alpha1/Subscription"
				} else if strings.Contains(ident.Name, "Og") || strings.Contains(ident.Name, "OperatorGroup") {
					gvk = "operators.coreos.com/v1alpha1/OperatorGroup"
				} else if strings.Contains(ident.Name, "DaemonSet") {
					gvk = "apps/v1/DaemonSet"
				}
				return gvk, sel.Sel.Name
			}
		}
	}

	// Check for environment/utility function patterns (e.g., metallbenv.*, nfdenv.*)
	if ident, ok := sel.X.(*ast.Ident); ok {
		if strings.HasSuffix(ident.Name, "env") || strings.HasSuffix(ident.Name, "Env") {
			// These are typically environment setup/teardown functions
			// Try to determine the resource type from the function name
			gvk := ""
			if strings.Contains(ident.Name, "metallb") {
				gvk = "apps/v1/DaemonSet" // MetalLB typically uses DaemonSets
			} else if strings.Contains(ident.Name, "nfd") {
				gvk = "apps/v1/DaemonSet" // NFD typically uses DaemonSets
			} else if strings.Contains(ident.Name, "sriov") {
				gvk = "apps/v1/DaemonSet" // SR-IOV typically uses DaemonSets
			}
			if gvk != "" {
				// Determine verb from function name
				verb := "unknown"
				if strings.Contains(sel.Sel.Name, "Create") {
					verb = "create"
				} else if strings.Contains(sel.Sel.Name, "Delete") {
					verb = "delete"
				} else if strings.Contains(sel.Sel.Name, "Update") {
					verb = "update"
				} else if strings.Contains(sel.Sel.Name, "Get") {
					verb = "get"
				} else if strings.Contains(sel.Sel.Name, "List") {
					verb = "list"
				}
				return gvk, verb
			}
		}
	}

	return "", ""
}

func detectK8sTypePattern(expr ast.Expr) (gvk string) {
	switch x := expr.(type) {
	case *ast.SelectorExpr:
		if ident, ok := x.X.(*ast.Ident); ok {
			prefix := ident.Name + "."
			if groupVersion, exists := k8sTypePatterns[prefix]; exists {
				return groupVersion + x.Sel.Name
			}
		}
	case *ast.CompositeLit:
		return detectK8sTypePattern(x.Type)
	}
	return ""
}

func isGinkgoTest(node ast.Node) bool {
	switch x := node.(type) {
	case *ast.CallExpr:
		if sel, ok := x.Fun.(*ast.SelectorExpr); ok {
			// Check for Ginkgo patterns like Describe, It, By
			ginkgoPatterns := []string{"Describe", "It", "By", "Context", "BeforeAll", "AfterAll", "BeforeEach", "AfterEach"}
			for _, pattern := range ginkgoPatterns {
				if sel.Sel.Name == pattern {
					return true
				}
			}
		}
		// Also check for direct function calls like Describe(...)
		if ident, ok := x.Fun.(*ast.Ident); ok {
			ginkgoPatterns := []string{"Describe", "It", "By", "Context", "BeforeAll", "AfterAll", "BeforeEach", "AfterEach"}
			for _, pattern := range ginkgoPatterns {
				if ident.Name == pattern {
					return true
				}
			}
		}
	}
	return false
}

func detectExpectations(call *ast.CallExpr) []map[string]string {
	var expectations []map[string]string

	// Check for Ginkgo/Gomega expectation patterns
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		// Check for Expect() calls
		if sel.Sel.Name == "Expect" {
			expectation := extractGinkgoExpectation(call)
			if expectation != nil {
				expectations = append(expectations, expectation)
			}
		} else if sel.Sel.Name == "ToNot" || sel.Sel.Name == "To" || sel.Sel.Name == "Should" || sel.Sel.Name == "Eventually" || sel.Sel.Name == "Consistently" {
			// Check for chained matcher methods like .ToNot(HaveOccurred())
			expectation := extractChainedMatcher(call)
			if expectation != nil {
				expectations = append(expectations, expectation)
			}
		}
	} else if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "Expect" {
		// Check for direct function calls like Expect(...)
		expectation := extractGinkgoExpectation(call)
		if expectation != nil {
			expectations = append(expectations, expectation)
		}
	}

	return expectations
}

func extractGinkgoExpectation(call *ast.CallExpr) map[string]string {
	// Handle Expect(...) patterns
	if len(call.Args) == 0 {
		return nil
	}

	// Get the first argument (the value being tested)
	arg := call.Args[0]

	// Convert the argument to a string representation
	argStr := astToString(arg)
	if argStr == "" {
		return nil
	}

	// For direct Expect calls, analyze the content for better classification
	return extractChainedExpectation(arg, call)
}

func extractChainedMatcher(call *ast.CallExpr) map[string]string {
	// This is a chained matcher method like .ToNot(HaveOccurred())
	// We need to find the original value being tested
	// For now, we'll extract the matcher method name and arguments
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil
	}

	matcherName := sel.Sel.Name
	args := make([]string, len(call.Args))
	for i, arg := range call.Args {
		args[i] = astToString(arg)
	}

	// Create a meaningful condition from the matcher
	condition := matcherName + "(" + strings.Join(args, ", ") + ")"

	// Try to classify based on the matcher type
	target := "test_condition"
	if strings.Contains(condition, "HaveOccurred") || strings.Contains(condition, "Not(HaveOccurred)") {
		target = "test_condition" // Error handling
	} else if strings.Contains(condition, "BeEmpty") || strings.Contains(condition, "Not(BeEmpty)") {
		target = "resource_count" // Empty/not empty usually means count checks
	} else if strings.Contains(condition, "BeTrue") || strings.Contains(condition, "BeFalse") {
		target = "test_condition" // Boolean conditions
	}

	return map[string]string{
		"target":    target,
		"condition": condition,
	}
}

func extractChainedExpectation(arg ast.Expr, call *ast.CallExpr) map[string]string {
	// Extract the actual value being tested, not just the matcher
	argStr := astToString(arg)

	// For Ginkgo/Gomega, we want to extract the actual value being tested
	// The matcher (BeTrue, Not(BeEmpty), etc.) is less important than the value

	// Standardized target classification for similarity search compatibility
	target := "test_condition" // Default target

	// Check for common patterns in the actual value being tested
	if strings.Contains(argStr, "err") {
		target = "test_condition" // Error conditions are still test conditions
	} else if strings.Contains(argStr, "version") || strings.Contains(argStr, "image") {
		target = "resource_version"
	} else if strings.Contains(argStr, "len(") || strings.Contains(argStr, "count") {
		target = "resource_count"
	} else if strings.Contains(argStr, "online") || strings.Contains(argStr, "status") {
		target = "resource_status"
	} else if strings.Contains(argStr, "empty") || strings.Contains(argStr, "deleted") {
		target = "resource_deletion"
	} else if strings.Contains(argStr, "cluster") || strings.Contains(argStr, "namespace") || strings.Contains(argStr, "pod") {
		target = "resource_count" // Cluster/namespace/pod related tests are usually about counts
	}

	// Create a more meaningful condition that includes both the value and the matcher
	condition := argStr
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Expect" {
		// Try to find the chained matcher method
		// This is a simplified approach - in practice, you'd need to traverse the full call chain
		condition = argStr + " (Gomega matcher)"
	}

	return map[string]string{
		"target":    target,
		"condition": condition,
	}
}

func astToString(node ast.Node) string {
	// Enhanced AST to string conversion for better Ginkgo/Gomega parsing
	switch x := node.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.BasicLit:
		return x.Value
	case *ast.SelectorExpr:
		return astToString(x.X) + "." + x.Sel.Name
	case *ast.CallExpr:
		funcStr := astToString(x.Fun)
		args := make([]string, len(x.Args))
		for i, arg := range x.Args {
			args[i] = astToString(arg)
		}
		return funcStr + "(" + strings.Join(args, ", ") + ")"
	case *ast.BinaryExpr:
		return astToString(x.X) + " " + x.Op.String() + " " + astToString(x.Y)
	case *ast.UnaryExpr:
		return x.Op.String() + astToString(x.X)
	case *ast.IndexExpr:
		return astToString(x.X) + "[" + astToString(x.Index) + "]"
	case *ast.KeyValueExpr:
		return astToString(x.Key) + ":" + astToString(x.Value)
	case *ast.CompositeLit:
		// For composite literals, try to extract meaningful info
		if sel, ok := x.Type.(*ast.SelectorExpr); ok {
			return "composite_" + astToString(sel)
		}
		return "composite_literal"
	default:
		// For unknown types, try to extract a meaningful string
		return fmt.Sprintf("%T", node)
	}
}

func detectExternals(call *ast.CallExpr) []string {
	var externals []string

	// Check for exec.Command patterns
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "Command" || sel.Sel.Name == "CommandContext" {
			if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "exec" {
				// Extract command arguments
				var cmdParts []string
				for _, arg := range call.Args {
					if bl, ok := arg.(*ast.BasicLit); ok && bl.Kind == token.STRING {
						cmdParts = append(cmdParts, strings.Trim(bl.Value, "\""))
					}
				}
				if len(cmdParts) > 0 {
					cmd := strings.Join(cmdParts, " ")
					// Check if it's a Kubernetes/OpenShift command
					if strings.Contains(cmd, "kubectl") || strings.Contains(cmd, "oc") {
						externals = append(externals, cmd)
					}
				}
			}
		}
	}

	return externals
}

func mapCommandToAPI(cmd string) (gvk string, verb string) {
	cmd = strings.ToLower(cmd)

	// kubectl/oc create patterns
	if strings.Contains(cmd, " create ") {
		if strings.Contains(cmd, " pod ") || strings.Contains(cmd, " pods ") {
			return "v1/Pod", "create"
		}
		if strings.Contains(cmd, " service ") || strings.Contains(cmd, " svc ") {
			return "v1/Service", "create"
		}
		if strings.Contains(cmd, " deployment ") || strings.Contains(cmd, " deploy ") {
			return "apps/v1/Deployment", "create"
		}
		if strings.Contains(cmd, " route ") {
			return "route.openshift.io/v1/Route", "create"
		}
		if strings.Contains(cmd, " namespace ") || strings.Contains(cmd, " ns ") {
			return "v1/Namespace", "create"
		}
		if strings.Contains(cmd, " configmap ") {
			return "v1/ConfigMap", "create"
		}
		if strings.Contains(cmd, " secret ") {
			return "v1/Secret", "create"
		}
		if strings.Contains(cmd, " ingress ") {
			return "networking.k8s.io/v1/Ingress", "create"
		}
	}

	// kubectl/oc get patterns
	if strings.Contains(cmd, " get ") {
		if strings.Contains(cmd, " pod ") || strings.Contains(cmd, " pods ") {
			return "v1/Pod", "get"
		}
		if strings.Contains(cmd, " service ") || strings.Contains(cmd, " svc ") {
			return "v1/Service", "get"
		}
		if strings.Contains(cmd, " deployment ") || strings.Contains(cmd, " deploy ") {
			return "apps/v1/Deployment", "get"
		}
		if strings.Contains(cmd, " route ") {
			return "route.openshift.io/v1/Route", "get"
		}
		if strings.Contains(cmd, " namespace ") || strings.Contains(cmd, " ns ") {
			return "v1/Namespace", "get"
		}
		if strings.Contains(cmd, " configmap ") {
			return "v1/ConfigMap", "get"
		}
		if strings.Contains(cmd, " secret ") {
			return "v1/Secret", "get"
		}
		if strings.Contains(cmd, " ingress ") {
			return "networking.k8s.io/v1/Ingress", "get"
		}
	}

	// kubectl/oc delete patterns
	if strings.Contains(cmd, " delete ") {
		if strings.Contains(cmd, " pod ") || strings.Contains(cmd, " pods ") {
			return "v1/Pod", "delete"
		}
		if strings.Contains(cmd, " service ") || strings.Contains(cmd, " svc ") {
			return "v1/Service", "delete"
		}
		if strings.Contains(cmd, " deployment ") || strings.Contains(cmd, " deploy ") {
			return "apps/v1/Deployment", "delete"
		}
		if strings.Contains(cmd, " route ") {
			return "route.openshift.io/v1/Route", "delete"
		}
		if strings.Contains(cmd, " namespace ") || strings.Contains(cmd, " ns ") {
			return "v1/Namespace", "delete"
		}
		if strings.Contains(cmd, " configmap ") {
			return "v1/ConfigMap", "delete"
		}
		if strings.Contains(cmd, " secret ") {
			return "v1/Secret", "delete"
		}
		if strings.Contains(cmd, " ingress ") {
			return "networking.k8s.io/v1/Ingress", "delete"
		}
	}

	// kubectl/oc patch patterns
	if strings.Contains(cmd, " patch ") {
		if strings.Contains(cmd, " pod ") || strings.Contains(cmd, " pods ") {
			return "v1/Pod", "patch"
		}
		if strings.Contains(cmd, " service ") || strings.Contains(cmd, " svc ") {
			return "v1/Service", "patch"
		}
		if strings.Contains(cmd, " deployment ") || strings.Contains(cmd, " deploy ") {
			return "apps/v1/Deployment", "patch"
		}
		if strings.Contains(cmd, " route ") {
			return "route.openshift.io/v1/Route", "patch"
		}
		if strings.Contains(cmd, " namespace ") || strings.Contains(cmd, " ns ") {
			return "v1/Namespace", "patch"
		}
		if strings.Contains(cmd, " configmap ") {
			return "v1/ConfigMap", "patch"
		}
		if strings.Contains(cmd, " secret ") {
			return "v1/Secret", "patch"
		}
		if strings.Contains(cmd, " ingress ") {
			return "networking.k8s.io/v1/Ingress", "patch"
		}
	}

	// kubectl/oc apply patterns
	if strings.Contains(cmd, " apply ") {
		// Apply is more generic, could be any resource
		// We'll need to parse the YAML/JSON to determine the resource type
		// For now, return a generic mapping
		return "unknown/unknown", "apply"
	}

	return "", ""
}

func main() {
	root := flag.String("root", ".", "root of the Go repo to scan")
	flag.Parse()

	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	_ = filepath.WalkDir(*root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || (!strings.HasSuffix(path, "_test.go") && !strings.HasSuffix(path, ".go")) {
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

		// Check if file contains Ginkgo patterns
		hasGinkgoPatterns := false
		ast.Inspect(f, func(n ast.Node) bool {
			if isGinkgoTest(n) {
				hasGinkgoPatterns = true
				return false // Stop searching
			}
			return true
		})

		// If file has Ginkgo patterns, create a consolidated spec
		if hasGinkgoPatterns {
			spec := KubeSpec{
				Level:             "integration",
				Preconditions:     []string{},
				Actions:           []Action{},
				Expectations:      []map[string]string{},
				OpenShiftSpecific: []string{},
				Concurrency:       []string{},
				Artifacts:         []string{},
			}
			// Replace full path with basename in test_id
			basename := filepath.Base(*root)
			relativePath := strings.TrimPrefix(strings.TrimPrefix(path, *root), "/")
			spec.TestID = fmt.Sprintf("%s/%s:ginkgo_tests", basename, relativePath)
			spec.Artifacts = append(spec.Artifacts, fileGolden...)

			openshiftSet := map[string]bool{}
			verbs := map[string]bool{}
			gvkSet := map[string]bool{}
			expectationsSet := map[string]bool{}

			// Analyze the entire file for patterns
			ast.Inspect(f, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.CallExpr:
					// Check for eco-goinfra patterns
					if gvk, verb := detectEcoGoinfraPattern(x); gvk != "" || verb != "" {
						if gvk != "" {
							gvkSet[gvk] = true
							if strings.Contains(gvk, "openshift") || strings.Contains(gvk, "route.openshift.io") {
								openshiftSet[gvk] = true
							}
						}
						if verb != "" {
							verbs[verb] = true
						}
					}

					// Check for standard Kubernetes API calls
					if sel, ok := x.Fun.(*ast.SelectorExpr); ok {
						if verbSet[sel.Sel.Name] {
							verbs[sel.Sel.Name] = true
						}
					}

					// Check for expectations (assertions)
					if exps := detectExpectations(x); len(exps) > 0 {
						for _, exp := range exps {
							key := fmt.Sprintf("%s:%s", exp["target"], exp["condition"])
							expectationsSet[key] = true
						}
					}

					// Check for external commands and map to API operations
					if exts := detectExternals(x); len(exts) > 0 {
						for _, ext := range exts {
							// Map external command to equivalent API operation
							if gvk, verb := mapCommandToAPI(ext); gvk != "" && verb != "" {
								gvkSet[gvk] = true
								verbs[verb] = true
								if strings.Contains(gvk, "openshift") || strings.Contains(gvk, "route.openshift.io") {
									openshiftSet[gvk] = true
								}
							}
						}
					}

					// Check for golden files in string arguments
					for _, a := range x.Args {
						if bl, ok := a.(*ast.BasicLit); ok && bl.Kind == token.STRING {
							if m := goldenRe.FindString(bl.Value); m != "" {
								spec.Artifacts = append(spec.Artifacts, strings.Trim(bl.Value, "\""))
							}
						}
					}
				case *ast.CompositeLit:
					// Check for Kubernetes composite literals
					if gvk, osh := kindFromCompositeLit(x); gvk != "" {
						gvkSet[gvk] = true
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

					// Check for Kubernetes API types in composite literals
					if gvk := detectK8sTypePattern(x); gvk != "" {
						gvkSet[gvk] = true
						if strings.Contains(gvk, "openshift") || strings.Contains(gvk, "route.openshift.io") {
							openshiftSet[gvk] = true
						}
					}
				}
				return true
			})

			// Add unique GVKs to actions
			for gvk := range gvkSet {
				spec.Actions = append(spec.Actions, Action{GVK: gvk})
			}

			// Add unique verbs to actions
			for v := range verbs {
				spec.Actions = append(spec.Actions, Action{Verb: strings.ToLower(v)})
			}

			// Add OpenShift-specific resources
			for k := range openshiftSet {
				spec.OpenShiftSpecific = append(spec.OpenShiftSpecific, k)
			}

			// Add expectations
			for exp := range expectationsSet {
				parts := strings.Split(exp, ":")
				if len(parts) == 2 {
					spec.Expectations = append(spec.Expectations, map[string]string{
						"target":    parts[0],
						"condition": parts[1],
					})
				}
			}

			// Add equivalence bridges
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

			// Only output if we found meaningful patterns
			if len(spec.Actions) > 0 || len(spec.OpenShiftSpecific) > 0 || len(spec.Preconditions) > 0 {
				b, _ := json.Marshal(spec)
				fmt.Fprintln(out, string(b))
			}
		}

		// Then check for test functions
		ast.Inspect(f, func(n ast.Node) bool {
			fd, ok := n.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || fd.Name == nil {
				return true
			}

			// Check if it's a test function or contains Ginkgo patterns
			isTestFunc := strings.HasPrefix(fd.Name.Name, "Test")
			hasGinkgoInFunc := false

			// Check if the function contains Ginkgo patterns
			ast.Inspect(fd, func(n2 ast.Node) bool {
				if isGinkgoTest(n2) {
					hasGinkgoInFunc = true
					return false // Stop searching
				}
				return true
			})

			if !isTestFunc && !hasGinkgoInFunc {
				return true
			}

			spec := KubeSpec{
				Level:             "unknown",
				Preconditions:     []string{},
				Actions:           []Action{},
				Expectations:      []map[string]string{},
				OpenShiftSpecific: []string{},
				Concurrency:       []string{},
				Artifacts:         []string{},
			}
			// Replace full path with basename in test_id
			basename := filepath.Base(*root)
			relativePath := strings.TrimPrefix(strings.TrimPrefix(path, *root), "/")
			spec.TestID = fmt.Sprintf("%s/%s:%s", basename, relativePath, fd.Name.Name)
			spec.Artifacts = append(spec.Artifacts, fileGolden...)

			openshiftSet := map[string]bool{}
			verbs := map[string]bool{}
			hasGinkgoPatterns := false
			expectationsSet := map[string]bool{}
			gvkSet := map[string]bool{}

			ast.Inspect(fd, func(n2 ast.Node) bool {
				switch x := n2.(type) {
				case *ast.CallExpr:
					// Check for eco-goinfra patterns
					if gvk, verb := detectEcoGoinfraPattern(x); gvk != "" || verb != "" {
						if gvk != "" {
							spec.Actions = append(spec.Actions, Action{GVK: gvk})
							if strings.Contains(gvk, "openshift") || strings.Contains(gvk, "route.openshift.io") {
								openshiftSet[gvk] = true
							}
						}
						if verb != "" {
							verbs[verb] = true
						}
					}

					// Check for standard Kubernetes API calls
					if sel, ok := x.Fun.(*ast.SelectorExpr); ok {
						if verbSet[sel.Sel.Name] {
							verbs[sel.Sel.Name] = true
						}
						if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "t" && sel.Sel.Name == "Parallel" {
							spec.Concurrency = append(spec.Concurrency, "parallel")
						}
					}

					// Check for expectations (assertions)
					if exps := detectExpectations(x); len(exps) > 0 {
						for _, exp := range exps {
							key := fmt.Sprintf("%s:%s", exp["target"], exp["condition"])
							expectationsSet[key] = true
						}
					}

					// Check for external commands and map to API operations
					if exts := detectExternals(x); len(exts) > 0 {
						for _, ext := range exts {
							// Map external command to equivalent API operation
							if gvk, verb := mapCommandToAPI(ext); gvk != "" && verb != "" {
								gvkSet[gvk] = true
								verbs[verb] = true
								if strings.Contains(gvk, "openshift") || strings.Contains(gvk, "route.openshift.io") {
									openshiftSet[gvk] = true
								}
							}
						}
					}

					// Check for Ginkgo patterns
					if isGinkgoTest(x) {
						hasGinkgoPatterns = true
					}

					// Check for golden files in string arguments
					for _, a := range x.Args {
						if bl, ok := a.(*ast.BasicLit); ok && bl.Kind == token.STRING {
							if m := goldenRe.FindString(bl.Value); m != "" {
								spec.Artifacts = append(spec.Artifacts, strings.Trim(bl.Value, "\""))
							}
						}
					}
				case *ast.CompositeLit:
					// Check for Kubernetes composite literals
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

					// Check for Kubernetes API types in composite literals
					if gvk := detectK8sTypePattern(x); gvk != "" {
						spec.Actions = append(spec.Actions, Action{GVK: gvk})
						if strings.Contains(gvk, "openshift") || strings.Contains(gvk, "route.openshift.io") {
							openshiftSet[gvk] = true
						}
					}
				}
				return true
			})

			// Add unique GVKs to actions
			for gvk := range gvkSet {
				spec.Actions = append(spec.Actions, Action{GVK: gvk})
			}

			for v := range verbs {
				spec.Actions = append(spec.Actions, Action{Verb: strings.ToLower(v)})
			}
			for k := range openshiftSet {
				spec.OpenShiftSpecific = append(spec.OpenShiftSpecific, k)
			}

			// Add expectations
			for exp := range expectationsSet {
				parts := strings.Split(exp, ":")
				if len(parts) == 2 {
					spec.Expectations = append(spec.Expectations, map[string]string{
						"target":    parts[0],
						"condition": parts[1],
					})
				}
			}

			// Determine test level
			if hasGinkgoPatterns {
				spec.Level = "integration" // Ginkgo tests are typically integration tests
			} else if verbs["Create"] || verbs["Delete"] || verbs["Watch"] {
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
