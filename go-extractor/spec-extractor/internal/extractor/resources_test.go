package extractor

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindGoMod(t *testing.T) {
	// Create temp dir with a go.mod
	tmpDir := t.TempDir()
	modContent := "module github.com/example/myproject\n\ngo 1.21\n"
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte(modContent), 0o644); err != nil {
		t.Fatal(err)
	}
	subDir := filepath.Join(tmpDir, "pkg", "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}

	modulePath, modDir, err := FindGoMod(subDir)
	if err != nil {
		t.Fatalf("FindGoMod: %v", err)
	}
	if modulePath != "github.com/example/myproject" {
		t.Errorf("modulePath = %q, want %q", modulePath, "github.com/example/myproject")
	}
	if modDir != tmpDir {
		t.Errorf("modDir = %q, want %q", modDir, tmpDir)
	}
}

func TestFindGoMod_NotFound(t *testing.T) {
	tmpDir := t.TempDir()
	_, _, err := FindGoMod(tmpDir)
	if err == nil {
		t.Error("expected error when no go.mod exists")
	}
}

func TestExtractResourceFromEcoGoinfra(t *testing.T) {
	tests := []struct {
		importPath string
		want       string
	}{
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/pod", "Pod"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/deployment", "Deployment"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/configmap", "ConfigMap"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/nodes", "Node"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/namespace", "Namespace"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/daemonset", "DaemonSet"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/statefulset", "StatefulSet"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/serviceaccount", "ServiceAccount"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/secret", "Secret"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/service", "Service"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/rbac", "RBAC"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/networkpolicy", "NetworkPolicy"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/nad", "NetworkAttachmentDefinition"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/route", "Route"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/ingress", "Ingress"},
		// Operator/domain-specific - kept as titlecase
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/sriov", "SRIOV"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/metallb", "MetalLB"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/ptp", "PTP"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/nfd", "NFD"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/olm", "OLM"},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/nmstate", "NMState"},
		// Excluded packages
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/clients", ""},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/reportxml", ""},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/schemes/argocd", ""},
		{"github.com/rh-ecosystem-edge/eco-goinfra/pkg/schemes/olm/operators/v1alpha1", ""},
		// Not eco-goinfra
		{"github.com/onsi/ginkgo/v2", ""},
		{"k8s.io/api/core/v1", ""},
	}
	for _, tt := range tests {
		got := extractResourceFromEcoGoinfra(tt.importPath)
		if got != tt.want {
			t.Errorf("extractResourceFromEcoGoinfra(%q) = %q, want %q", tt.importPath, got, tt.want)
		}
	}
}

func TestExtractResourcesFromK8sTypes(t *testing.T) {
	tests := []struct {
		typeRef string
		want    string
	}{
		{"Pod", "Pod"},
		{"Deployment", "Deployment"},
		{"StatefulSet", "StatefulSet"},
		{"DaemonSet", "DaemonSet"},
		{"Service", "Service"},
		{"ConfigMap", "ConfigMap"},
		{"Secret", "Secret"},
		{"Namespace", "Namespace"},
		{"Node", "Node"},
		{"Job", "Job"},
		{"CronJob", "CronJob"},
		{"ReplicaSet", "ReplicaSet"},
		{"PersistentVolume", "PersistentVolume"},
		{"PersistentVolumeClaim", "PersistentVolumeClaim"},
		{"NetworkPolicy", "NetworkPolicy"},
		{"Ingress", "Ingress"},
		{"Role", "Role"},
		{"ClusterRole", "ClusterRole"},
		{"RoleBinding", "RoleBinding"},
		{"ClusterRoleBinding", "ClusterRoleBinding"},
		{"ServiceAccount", "ServiceAccount"},
		{"StorageClass", "StorageClass"},
		{"PodDisruptionBudget", "PodDisruptionBudget"},
		{"HorizontalPodAutoscaler", "HorizontalPodAutoscaler"},
		// Sub-types and constants - should NOT match
		{"Container", ""},
		{"Volume", ""},
		{"SecurityContext", ""},
		{"PodRunning", ""},
		{"ProtocolTCP", ""},
		{"Toleration", ""},
		{"Affinity", ""},
		{"ResourceList", ""},
		{"Capability", ""},
		{"VolumeMount", ""},
		{"ContainerPort", ""},
	}
	for _, tt := range tests {
		got := lookupK8sResourceType(tt.typeRef)
		if got != tt.want {
			t.Errorf("lookupK8sResourceType(%q) = %q, want %q", tt.typeRef, got, tt.want)
		}
	}
}

func TestScanGoFile(t *testing.T) {
	tmpDir := t.TempDir()
	goFile := filepath.Join(tmpDir, "test.go")
	content := `package test

import (
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	appsv1 "k8s.io/api/apps/v1"
	"github.com/rh-ecosystem-edge/eco-goinfra/pkg/pod"
	"github.com/rh-ecosystem-edge/eco-goinfra/pkg/clients"
	"github.com/rh-ecosystem-edge/eco-gotests/tests/internal/helpers"
)

func TestSomething() {
	p := &corev1.Pod{}
	d := &appsv1.Deployment{}
	_ = corev1.PodRunning
	_ = corev1.Container{}
	pod.List()
	fmt.Println(time.Now())
}
`
	if err := os.WriteFile(goFile, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	result := scanGoFile(goFile)

	// Check imports
	wantImports := map[string]bool{
		"k8s.io/api/core/v1": true,
		"k8s.io/api/apps/v1": true,
		"github.com/rh-ecosystem-edge/eco-goinfra/pkg/pod":                true,
		"github.com/rh-ecosystem-edge/eco-goinfra/pkg/clients":            true,
		"github.com/rh-ecosystem-edge/eco-gotests/tests/internal/helpers": true,
	}
	for _, imp := range result.imports {
		if !wantImports[imp] {
			// stdlib imports are fine to skip
			if imp == "fmt" || imp == "time" {
				continue
			}
			t.Errorf("unexpected import: %q", imp)
		}
	}

	// Check k8s type refs - should find Pod and Deployment but not PodRunning or Container
	foundPod := false
	foundDeployment := false
	for _, ref := range result.k8sTypeRefs {
		if ref == "Pod" {
			foundPod = true
		}
		if ref == "Deployment" {
			foundDeployment = true
		}
		if ref == "PodRunning" || ref == "Container" {
			t.Errorf("should not include sub-type/constant: %q", ref)
		}
	}
	if !foundPod {
		t.Error("missing Pod type ref")
	}
	if !foundDeployment {
		t.Error("missing Deployment type ref")
	}
}

func TestScanGoFile_SingleImport(t *testing.T) {
	tmpDir := t.TempDir()
	goFile := filepath.Join(tmpDir, "single.go")
	content := `package test

import "github.com/rh-ecosystem-edge/eco-goinfra/pkg/service"

func Do() {}
`
	if err := os.WriteFile(goFile, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	result := scanGoFile(goFile)
	found := false
	for _, imp := range result.imports {
		if imp == "github.com/rh-ecosystem-edge/eco-goinfra/pkg/service" {
			found = true
		}
	}
	if !found {
		t.Error("missing single import")
	}
}

func TestResourceScanner_ScanFile(t *testing.T) {
	// Set up a mini repo with go.mod and test files
	tmpDir := t.TempDir()
	modContent := "module github.com/example/myrepo\n\ngo 1.21\n"
	os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte(modContent), 0o644)

	// Create a helper package that imports eco-goinfra
	helperDir := filepath.Join(tmpDir, "internal", "helper")
	os.MkdirAll(helperDir, 0o755)
	helperContent := `package helper

import (
	"github.com/rh-ecosystem-edge/eco-goinfra/pkg/deployment"
	corev1 "k8s.io/api/core/v1"
)

func Deploy() {
	deployment.New()
	_ = &corev1.Service{}
}
`
	os.WriteFile(filepath.Join(helperDir, "helper.go"), []byte(helperContent), 0o644)

	// Test file that imports the helper + direct eco-goinfra
	importMap := map[string]string{
		"pod":    "github.com/rh-ecosystem-edge/eco-goinfra/pkg/pod",
		"helper": "github.com/example/myrepo/internal/helper",
	}

	// Create a test file that uses oc.Run
	testFile := filepath.Join(tmpDir, "test.go")
	testContent := `package test

import (
	"github.com/rh-ecosystem-edge/eco-goinfra/pkg/pod"
	"github.com/example/myrepo/internal/helper"
)

func TestSomething() {
	pod.List()
}
`
	os.WriteFile(testFile, []byte(testContent), 0o644)

	scanner := NewResourceScanner("github.com/example/myrepo", tmpDir)
	resources := scanner.ScanFile(testFile, importMap)

	// Should find: Pod (direct), Deployment (transitive via helper), Service (k8s type in helper)
	want := map[string]bool{"Pod": true, "Deployment": true, "Service": true}
	got := map[string]bool{}
	for _, r := range resources {
		got[r] = true
	}
	for r := range want {
		if !got[r] {
			t.Errorf("missing resource: %s (got: %v)", r, resources)
		}
	}
}

func TestImportToDir(t *testing.T) {
	tests := []struct {
		importPath string
		modulePath string
		repoRoot   string
		want       string
	}{
		{
			"github.com/example/repo/tests/internal/helpers",
			"github.com/example/repo",
			"/home/user/repo",
			"/home/user/repo/tests/internal/helpers",
		},
		{
			"github.com/example/repo/pkg/sub",
			"github.com/example/repo",
			"/tmp/myrepo",
			"/tmp/myrepo/pkg/sub",
		},
	}
	for _, tt := range tests {
		got := importToDir(tt.importPath, tt.modulePath, tt.repoRoot)
		if got != tt.want {
			t.Errorf("importToDir(%q, %q, %q) = %q, want %q",
				tt.importPath, tt.modulePath, tt.repoRoot, got, tt.want)
		}
	}
}
