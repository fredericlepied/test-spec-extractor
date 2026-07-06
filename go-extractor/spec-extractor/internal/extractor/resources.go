package extractor

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// k8sResourceTypes is the set of top-level K8s API resource type names.
// Sub-types (Container, Volume, SecurityContext) and constants (PodRunning, ProtocolTCP) are excluded.
var k8sResourceTypes = map[string]bool{
	// core/v1
	"ConfigMap": true, "Endpoints": true, "Event": true, "LimitRange": true,
	"Namespace": true, "Node": true, "PersistentVolume": true, "PersistentVolumeClaim": true,
	"Pod": true, "ReplicationController": true, "ResourceQuota": true, "Secret": true,
	"Service": true, "ServiceAccount": true,
	// apps/v1
	"DaemonSet": true, "Deployment": true, "ReplicaSet": true, "StatefulSet": true,
	// batch/v1
	"CronJob": true, "Job": true,
	// networking/v1
	"Ingress": true, "IngressClass": true, "NetworkPolicy": true,
	// rbac/v1
	"ClusterRole": true, "ClusterRoleBinding": true, "Role": true, "RoleBinding": true,
	// storage/v1
	"CSIDriver": true, "CSINode": true, "StorageClass": true, "VolumeAttachment": true,
	// policy/v1
	"PodDisruptionBudget": true,
	// autoscaling
	"HorizontalPodAutoscaler": true,
	// certificates
	"CertificateSigningRequest": true,
}

// ecoGoinfraResourceMap normalizes eco-goinfra package names to resource names.
var ecoGoinfraResourceMap = map[string]string{
	// Standard K8s resources
	"pod": "Pod", "deployment": "Deployment", "daemonset": "DaemonSet",
	"statefulset": "StatefulSet", "configmap": "ConfigMap", "secret": "Secret",
	"service": "Service", "serviceaccount": "ServiceAccount", "namespace": "Namespace",
	"nodes": "Node", "rbac": "RBAC", "networkpolicy": "NetworkPolicy",
	"ingress": "Ingress", "route": "Route", "storage": "StorageClass",
	"nad": "NetworkAttachmentDefinition", "egressip": "EgressIP",
	"egressservice": "EgressService",
	// Operator/domain-specific
	"sriov": "SRIOV", "metallb": "MetalLB", "ptp": "PTP", "nfd": "NFD",
	"olm": "OLM", "nmstate": "NMState", "hive": "Hive", "argocd": "ArgoCD",
	"kmm": "KMM", "lca": "LCA", "mco": "MCO", "ocm": "OCM",
	"clusteroperator": "ClusterOperator", "clusterversion": "ClusterVersion",
	"clusterlogging": "ClusterLogging", "siteconfig": "SiteConfig",
	"machine": "Machine", "bmc": "BMC", "lso": "LSO",
	"oran": "ORAN", "imageregistry": "ImageRegistry", "imagestream": "ImageStream",
	"keda": "KEDA", "kserve": "KServe", "velero": "Velero", "oadp": "OADP",
	"infrastructure": "Infrastructure", "console": "Console", "dns": "DNS",
	"proxy": "Proxy", "monitoring": "Monitoring", "webhook": "Webhook",
	"events": "Event", "ovn": "OVN", "network": "Network",
	"assisted": "AssistedInstall", "ibi": "ImageBasedInstall",
	"servicemesh": "ServiceMesh", "neuron": "Neuron", "nvidiagpu": "NvidiaGPU",
	"amdgpu": "AMDGPU",
}

// ecoGoinfraExcluded are eco-goinfra packages that are not resources.
var ecoGoinfraExcluded = map[string]bool{
	"clients": true, "reportxml": true, "msg": true,
	"apiservers": true, "nodesconfig": true, "pfstatus": true,
	"nrop": true, "nto": true, "scc": true,
}

const ecoGoinfraPrefix = "eco-goinfra/pkg/"

// k8sAPIAliasPattern matches import lines like: corev1 "k8s.io/api/core/v1"
var k8sAPIImportRe = regexp.MustCompile(`(\w+)\s+"k8s\.io/api/`)

// k8sTypeRefRe matches type references like corev1.Pod, appsv1.Deployment
var k8sTypeRefRe = regexp.MustCompile(`\b(\w+)\.([A-Z][A-Za-z]+)`)

type fileScanResult struct {
	imports     []string        // all import paths
	k8sTypeRefs []string        // K8s resource type names found (deduplicated)
	k8sAPIAlias map[string]bool // local aliases for k8s.io/api imports
}

// ResourceScanner resolves K8s resource types from imports transitively.
type ResourceScanner struct {
	modulePath string
	repoRoot   string
	cache      map[string]*fileScanResult
}

// FindGoMod walks up from startDir looking for go.mod and returns the module path and directory.
func FindGoMod(startDir string) (string, string, error) {
	dir, err := filepath.Abs(startDir)
	if err != nil {
		return "", "", err
	}
	for {
		modPath := filepath.Join(dir, "go.mod")
		if _, err := os.Stat(modPath); err == nil {
			modulePath, err := parseGoMod(modPath)
			if err != nil {
				return "", "", err
			}
			return modulePath, dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", "", fmt.Errorf("go.mod not found above %s", startDir)
}

func parseGoMod(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "module ") {
			return strings.TrimSpace(line[7:]), nil
		}
	}
	return "", fmt.Errorf("no module directive in %s", path)
}

func NewResourceScanner(modulePath, repoRoot string) *ResourceScanner {
	return &ResourceScanner{
		modulePath: modulePath,
		repoRoot:   repoRoot,
		cache:      make(map[string]*fileScanResult),
	}
}

// ScanFile takes a file's ImportMap and returns sorted, deduplicated K8s resource names.
func (rs *ResourceScanner) ScanFile(importMap map[string]string) []string {
	if rs == nil {
		return nil
	}
	resources := map[string]bool{}
	visited := map[string]bool{}

	for _, fullPath := range importMap {
		// Direct eco-goinfra imports
		if r := extractResourceFromEcoGoinfra(fullPath); r != "" {
			resources[r] = true
		}
		// Follow same-module imports transitively
		if strings.HasPrefix(fullPath, rs.modulePath+"/") {
			dir := importToDir(fullPath, rs.modulePath, rs.repoRoot)
			rs.scanDir(dir, visited, resources)
		}
	}

	if len(resources) == 0 {
		return nil
	}
	result := make([]string, 0, len(resources))
	for r := range resources {
		result = append(result, r)
	}
	sort.Strings(result)
	return result
}

func (rs *ResourceScanner) scanDir(dir string, visited map[string]bool, resources map[string]bool) {
	if visited[dir] {
		return
	}
	visited[dir] = true

	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}

	// Collect all imports and type refs from .go files in this directory
	var allImports []string
	k8sAliases := map[string]bool{}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			continue
		}
		filePath := filepath.Join(dir, entry.Name())

		var result *fileScanResult
		if cached, ok := rs.cache[filePath]; ok {
			result = cached
		} else {
			result = scanGoFile(filePath)
			rs.cache[filePath] = result
		}

		allImports = append(allImports, result.imports...)
		for alias := range result.k8sAPIAlias {
			k8sAliases[alias] = true
		}
		for _, ref := range result.k8sTypeRefs {
			resources[ref] = true
		}
	}

	// Process collected imports
	for _, imp := range allImports {
		if r := extractResourceFromEcoGoinfra(imp); r != "" {
			resources[r] = true
		}
		if strings.HasPrefix(imp, rs.modulePath+"/") {
			subDir := importToDir(imp, rs.modulePath, rs.repoRoot)
			rs.scanDir(subDir, visited, resources)
		}
	}
}

// scanGoFile scans a single .go file for imports and K8s type references.
func scanGoFile(path string) *fileScanResult {
	f, err := os.Open(path)
	if err != nil {
		return &fileScanResult{}
	}
	defer f.Close()

	result := &fileScanResult{
		k8sAPIAlias: map[string]bool{},
	}
	scanner := bufio.NewScanner(f)
	inImportBlock := false
	var lines []string

	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		trimmed := strings.TrimSpace(line)

		if inImportBlock {
			if trimmed == ")" {
				inImportBlock = false
				continue
			}
			if imp := extractImportPath(trimmed); imp != "" {
				result.imports = append(result.imports, imp)
				// Track k8s API aliases
				if m := k8sAPIImportRe.FindStringSubmatch(trimmed); len(m) > 1 {
					result.k8sAPIAlias[m[1]] = true
				}
			}
			continue
		}

		if strings.HasPrefix(trimmed, "import (") {
			inImportBlock = true
			continue
		}
		if strings.HasPrefix(trimmed, "import \"") || strings.HasPrefix(trimmed, "import `") {
			if imp := extractImportPath(trimmed); imp != "" {
				result.imports = append(result.imports, imp)
			}
		}
	}

	// Second pass: find k8s type references if we have k8s API aliases
	if len(result.k8sAPIAlias) > 0 {
		seen := map[string]bool{}
		for _, line := range lines {
			for _, match := range k8sTypeRefRe.FindAllStringSubmatch(line, -1) {
				alias := match[1]
				typeName := match[2]
				if result.k8sAPIAlias[alias] {
					if r := lookupK8sResourceType(typeName); r != "" && !seen[r] {
						seen[r] = true
						result.k8sTypeRefs = append(result.k8sTypeRefs, r)
					}
				}
			}
		}
	}

	return result
}

func extractImportPath(line string) string {
	// Find content between double quotes
	start := strings.IndexByte(line, '"')
	if start < 0 {
		return ""
	}
	end := strings.IndexByte(line[start+1:], '"')
	if end < 0 {
		return ""
	}
	return line[start+1 : start+1+end]
}

func extractResourceFromEcoGoinfra(importPath string) string {
	idx := strings.Index(importPath, ecoGoinfraPrefix)
	if idx < 0 {
		return ""
	}
	pkg := importPath[idx+len(ecoGoinfraPrefix):]
	// Handle sub-packages like schemes/olm/v1alpha1
	if strings.HasPrefix(pkg, "schemes") {
		return ""
	}
	// Take only the first path component
	if slash := strings.IndexByte(pkg, '/'); slash >= 0 {
		pkg = pkg[:slash]
	}
	if ecoGoinfraExcluded[pkg] {
		return ""
	}
	if mapped, ok := ecoGoinfraResourceMap[pkg]; ok {
		return mapped
	}
	// Unknown package - return titlecased name
	if len(pkg) > 0 {
		return strings.ToUpper(pkg[:1]) + pkg[1:]
	}
	return ""
}

func lookupK8sResourceType(typeName string) string {
	if k8sResourceTypes[typeName] {
		return typeName
	}
	// Also check List variants: PodList -> Pod
	if strings.HasSuffix(typeName, "List") {
		base := typeName[:len(typeName)-4]
		if k8sResourceTypes[base] {
			return base
		}
	}
	return ""
}

func importToDir(importPath, modulePath, repoRoot string) string {
	suffix := strings.TrimPrefix(importPath, modulePath+"/")
	return filepath.Join(repoRoot, suffix)
}
