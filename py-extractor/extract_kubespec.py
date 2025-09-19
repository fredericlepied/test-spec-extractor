#!/usr/bin/env python3

# py-extractor/extract_kubespec.py
import argparse, ast, json, os, re
from pathlib import Path

VERB_PREFIXES = ("create_", "patch_", "delete_", "read_", "list_", "replace_", "watch_")
CLI_RE = re.compile(r"\b(oc|kubectl)\b")
GOLDEN_RE = re.compile(r"(?i)testdata/[^\"']+")

PSA_KEYS = [
    "pod-security.kubernetes.io/enforce",
    "pod-security.kubernetes.io/audit",
    "pod-security.kubernetes.io/warn",
]
SCC_CLI_PATTERNS = [
    "oc adm policy add-scc-to-user",
    "oc adm policy add-scc-to-group",
]

# Purpose detection patterns - keywords that indicate test purpose
PURPOSE_PATTERNS = {
    "NETWORK_CONNECTIVITY": [
        "curl",
        "url",
        "frr",
        "routing",
        "connectivity",
        "reach",
        "ping",
        "network",
        "traffic",
    ],
    "POD_HEALTH": ["pods", "status", "running", "phase", "health", "ready", "condition", "state"],
    "POD_MANAGEMENT": ["create", "delete", "update", "pod", "deployment", "replica", "scale"],
    "NETWORK_POLICY": ["policy", "network", "multinetwork", "ingress", "egress", "security"],
    "RESOURCE_VALIDATION": ["count", "exist", "validation", "verify", "check", "assert"],
    "OPERATOR_MANAGEMENT": ["operator", "subscription", "csv", "catalogsource", "installplan"],
    "STORAGE_TESTING": ["storage", "volume", "pvc", "pv", "mount", "filesystem"],
    "SECURITY_TESTING": ["security", "rbac", "scc", "psa", "permission", "access"],
    "CONFIGURATION": ["config", "configuration", "settings", "parameters", "env"],
    "PERFORMANCE": ["performance", "load", "stress", "benchmark", "latency", "throughput"],
}

# Test type detection patterns
TEST_TYPE_PATTERNS = {
    "unit": ["test", "Test", "unit", "Unit", "mock", "Mock"],
    "integration": ["integration", "Integration", "e2e", "E2E", "pytest", "Pytest"],
    "performance": ["performance", "Performance", "benchmark", "Benchmark", "load", "Load", "stress", "Stress"],
    "conformance": ["conformance", "Conformance", "k8s", "K8s", "kubernetes", "Kubernetes"],
}

# Dependency detection patterns
DEPENDENCY_PATTERNS = {
    "operator": ["operator", "Operator", "csv", "CSV", "subscription", "Subscription"],
    "storage": ["storage", "Storage", "pvc", "PVC", "pv", "PV", "volume", "Volume"],
    "network": ["network", "Network", "cni", "CNI", "multus", "Multus", "sriov", "SR-IOV"],
    "security": ["security", "Security", "rbac", "RBAC", "scc", "SCC", "psa", "PSA"],
    "monitoring": ["monitoring", "Monitoring", "prometheus", "Prometheus", "grafana", "Grafana"],
    "logging": ["logging", "Logging", "fluentd", "Fluentd", "elasticsearch", "Elasticsearch"],
}

# Environment detection patterns
ENVIRONMENT_PATTERNS = {
    "single_node": ["sno", "SNO", "single", "Single", "standalone", "Standalone"],
    "multi_node": ["multi", "Multi", "cluster", "Cluster", "nodes", "Nodes"],
    "bare_metal": ["bare", "Bare", "metal", "Metal", "bmh", "BMH", "ironic", "Ironic"],
    "virtual": ["virtual", "Virtual", "vm", "VM", "kvm", "KVM", "libvirt", "Libvirt"],
    "cloud": ["cloud", "Cloud", "aws", "AWS", "azure", "Azure", "gcp", "GCP"],
    "edge": ["edge", "Edge", "remote", "Remote", "far", "Far"],
}


def detect_test_type(test_name: str, file_path: str, docstring: str) -> str:
    """Detect the type of test based on file path, test name, and content"""
    content = (test_name + " " + file_path + " " + (docstring or "")).lower()
    
    scores = {}
    for test_type, patterns in TEST_TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if pattern.lower() in content:
                score += 1
        scores[test_type] = score
    
    # Find the test type with highest score
    max_score = 0
    detected_type = "unknown"
    for test_type, score in scores.items():
        if score > max_score:
            max_score = score
            detected_type = test_type
    
    # Default to integration if it's a pytest test
    if detected_type == "unknown" and ("pytest" in content or "test" in content):
        detected_type = "integration"
    
    return detected_type


def detect_dependencies(test_name: str, file_path: str, docstring: str, actions: list) -> list:
    """Detect required dependencies based on test content"""
    content = (test_name + " " + file_path + " " + (docstring or "")).lower()
    
    # Add action-based dependencies
    for action in actions:
        gvk = action.get("gvk", "").lower()
        if "operator" in gvk or "subscription" in gvk or "csv" in gvk:
            content += " operator"
        if "pvc" in gvk or "pv" in gvk or "storage" in gvk:
            content += " storage"
        if "network" in gvk or "cni" in gvk or "multus" in gvk:
            content += " network"
        if "rbac" in gvk or "scc" in gvk or "security" in gvk:
            content += " security"
    
    dependencies = []
    for dep_type, patterns in DEPENDENCY_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in content:
                dependencies.append(dep_type)
                break
    
    return dependencies


def detect_environment(test_name: str, file_path: str, docstring: str) -> list:
    """Detect the target environment based on test content"""
    content = (test_name + " " + file_path + " " + (docstring or "")).lower()
    
    environment = []
    for env_type, patterns in ENVIRONMENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in content:
                environment.append(env_type)
                break
    
    # Default to multi_node if no environment detected
    if not environment:
        environment = ["multi_node"]
    
    return environment


def extract_assertion_expectation(assert_node: ast.Assert) -> dict:
    """Extract meaningful expectation information from assert statements"""
    if not assert_node.test:
        return None

    # Helper function to safely convert AST nodes to strings
    def ast_to_string(node):
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        else:
            return str(node)

    # Helper function to convert comparison operators to strings
    def op_to_string(op):
        op_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
        }
        return op_map.get(type(op), str(op))

    # Handle comparison operations (==, !=, <, >, etc.)
    if isinstance(assert_node.test, ast.Compare):
        left = ast_to_string(assert_node.test.left)
        ops = [op_to_string(op) for op in assert_node.test.ops]
        comparators = [ast_to_string(comp) for comp in assert_node.test.comparators]

        if len(ops) == 1 and len(comparators) == 1:
            condition = f"{left} {ops[0]} {comparators[0]}"
            # Standardized target classification for similarity search compatibility
            target = "test_condition"  # Default to match Go extractor
            if "len(" in left and any(
                resource in left.lower()
                for resource in [
                    "baremetalhost",
                    "clusterdeployment",
                    "namespace",
                    "pod",
                    "service",
                ]
            ):
                target = "resource_count"
            elif "online" in left.lower() or "status" in left.lower():
                target = "resource_status"
            elif "deleted" in left.lower() or "not in" in condition or "empty" in condition.lower():
                target = "resource_deletion"
            elif "version" in left.lower() or "image" in left.lower():
                target = "resource_version"

            return {"target": target, "condition": condition}

    # Handle membership tests (in, not in)
    elif isinstance(assert_node.test, ast.Compare) and len(assert_node.test.ops) == 1:
        op = assert_node.test.ops[0]
        if isinstance(op, (ast.In, ast.NotIn)):
            left = ast_to_string(assert_node.test.left)
            right = ast_to_string(assert_node.test.comparators[0])
            condition = f"{left} {'not in' if isinstance(op, ast.NotIn) else 'in'} {right}"
            return {
                "target": "resource_deletion" if isinstance(op, ast.NotIn) else "test_condition",
                "condition": condition,
            }

    # Handle boolean operations (and, or)
    elif isinstance(assert_node.test, ast.BoolOp):
        op = "and" if isinstance(assert_node.test.op, ast.And) else "or"
        values = [ast_to_string(v) for v in assert_node.test.values]
        condition = f" {op} ".join(values)
        return {"target": "compound_condition", "condition": condition}

    # Generic assertion
    condition = ast_to_string(assert_node.test)
    return {"target": "test_condition", "condition": condition}


class TestVisitor(ast.NodeVisitor):
    def __init__(self, path, root_path):
        self.path = path
        self.root_path = root_path
        self.specs = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not node.name.startswith("test"):
            return
        # Create test_id with basename substitution
        basename = os.path.basename(self.root_path)
        relative_path = os.path.relpath(self.path, self.root_path)
        spec = {
            "test_id": f"{basename}/{relative_path}:{node.name}",
            "test_type": "unknown",
            "dependencies": [],
            "environment": [],
            "actions": [],
            "expectations": [],
            "openshift_specific": [],
            "concurrency": [],
            "artifacts": [],
            "purpose": "",
        }
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "parametrize":
                spec["dependencies"].append("parametrized")
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func = n.func

                # Check for direct helper function calls (e.g., get_resource, get_resource_from_namespace)
                if isinstance(func, ast.Name) and func.id in [
                    "get_resource",
                    "get_resource_from_namespace",
                ]:
                    if n.args and isinstance(n.args[0], ast.Constant):
                        resource_name = n.args[0].value
                        gvk = map_resource_name_to_gvk(resource_name)
                        if gvk:
                            spec["actions"].append({"gvk": gvk, "verb": "get"})
                    elif n.args:
                        # For any arguments (including variables), add a generic action
                        spec["actions"].append({"gvk": "unknown/unknown", "verb": "get"})

                # Check for attribute-based calls
                elif isinstance(func, ast.Attribute):
                    mname = func.attr or ""
                    low = mname.lower()
                    if low.startswith(VERB_PREFIXES):
                        verb = low.split("_", 1)[0]
                        kind = low.split("_")[-1]
                        spec["actions"].append(
                            {
                                "gvk": guess_gvk_from_attr(func),
                                "verb": verb.replace("read", "get"),
                                "fields": {"kind_hint": kind.capitalize()},
                            }
                        )

                    # Check for openshift library patterns
                    if isinstance(func.value, ast.Name) and func.value.id == "oc":
                        if mname in ["selector"]:
                            # oc.selector(resource_name) - this is a get/list operation
                            if n.args and isinstance(n.args[0], ast.Constant):
                                resource_name = n.args[0].value
                                gvk = map_resource_name_to_gvk(resource_name)
                                if gvk:
                                    spec["actions"].append({"gvk": gvk, "verb": "get"})

                    # Check for imported helper function calls (e.g., from oc_helpers import get_resource)
                    if func.attr in [
                        "get_resource",
                        "get_resource_from_namespace",
                    ]:
                        if n.args and isinstance(n.args[0], ast.Constant):
                            resource_name = n.args[0].value
                            gvk = map_resource_name_to_gvk(resource_name)
                            if gvk:
                                spec["actions"].append({"gvk": gvk, "verb": "get"})
                        elif n.args:
                            # For any arguments (including variables), add a generic action
                            spec["actions"].append({"gvk": "unknown/unknown", "verb": "get"})
                if isinstance(func, ast.Attribute) and func.attr in {
                    "run",
                    "check_call",
                    "check_output",
                }:
                    if getattr(func.value, "id", "") == "subprocess":
                        args = []
                        for a in n.args:
                            if isinstance(a, ast.List):
                                for el in a.elts:
                                    if isinstance(el, ast.Constant):
                                        args.append(str(el.value))
                            elif isinstance(a, ast.Constant):
                                args.append(str(a.value))
                        cmd = " ".join(args)
                        if CLI_RE.search(cmd):
                            # Map CLI command to equivalent API operation
                            low = cmd.lower()
                            gvk, verb = map_command_to_api(cmd, low)
                            if gvk and verb:
                                spec["actions"].append({"gvk": gvk, "verb": verb})

                            # Handle PSA labels
                            if (" label " in low and " ns " in low) and any(
                                k in low for k in PSA_KEYS
                            ):
                                for k in PSA_KEYS:
                                    if k in low:
                                        m = re.search(k + r"=([^\s]+)", low)
                                        if m:
                                            spec["dependencies"].append(f"psa:{k}={m.group(1)}")
                            if any(p in low for p in [s.lower() for s in SCC_CLI_PATTERNS]):
                                spec["dependencies"].append("equiv:scc~psa")
                            if isinstance(func, ast.Attribute) and func.attr == "raises":
                                if getattr(func.value, "id", "") == "pytest":
                                    spec["expectations"].append(
                                        {"target": "exception", "condition": "raises"}
                                    )
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                m = GOLDEN_RE.search(n.value)
                if m:
                    spec["artifacts"].append(m.group(0))

            # Check for assert statements (expectations)
            if isinstance(n, ast.Assert):
                expectation = extract_assertion_expectation(n)
                if expectation:
                    spec["expectations"].append(expectation)
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                low = (n.func.attr or "").lower()
                if "ingress" in low:
                    verb = low.split("_", 1)[0]
                    spec["actions"].append(
                        {
                            "gvk": "networking.k8s.io/v1/Ingress",
                            "verb": verb.replace("read", "get"),
                        }
                    )

        if any(
            a.get("verb") in {"create", "delete", "patch", "replace", "watch"}
            for a in spec["actions"]
        ):
            spec["level"] = "integration"

        bridges = set()
        for a in spec["actions"]:
            g = (a.get("gvk") or "").lower()
            if "route.openshift.io" in g and "/route" in g:
                bridges.add("equiv:route~ingress")
            if "networking.k8s.io" in g and "/ingress" in g:
                bridges.add("equiv:route~ingress")
            if "security.openshift.io" in g:
                bridges.add("equiv:scc~psa")
        for p in spec["dependencies"]:
            if p.startswith("psa:"):
                bridges.add("equiv:scc~psa")
        for b in sorted(bridges):
            spec["dependencies"].append(b)

        # Detect test type, dependencies, and environment
        docstring = ast.get_docstring(node) or ""
        spec["test_type"] = detect_test_type(node.name, self.path, docstring)
        spec["dependencies"].extend(detect_dependencies(node.name, self.path, docstring, spec["actions"]))
        spec["environment"] = detect_environment(node.name, self.path, docstring)
        
        # Detect purpose based on test content
        spec["purpose"] = detect_purpose(
            node.name, docstring, spec["actions"], spec["expectations"]
        )

        self.specs.append(spec)


def guess_gvk_from_attr(attr: ast.Attribute) -> str:
    root = attr
    while isinstance(root, ast.Attribute):
        if isinstance(root.value, ast.Call) and isinstance(root.value.func, ast.Attribute):
            api = root.value.func.attr
            group, version = api_to_group_version(api)
            if group or version:
                return f"{group+'/'+version if group else version}"
        root = root.value if isinstance(root.value, ast.Attribute) else getattr(root, "value", None)
        if root is None:
            break
    return ""


def map_resource_name_to_gvk(resource_name: str) -> str:
    """Map Kubernetes resource name to GVK"""
    resource_name = resource_name.lower()

    # Common resource mappings
    resource_map = {
        "pods": "v1/Pod",
        "pod": "v1/Pod",
        "services": "v1/Service",
        "service": "v1/Service",
        "deployments": "apps/v1/Deployment",
        "deployment": "apps/v1/Deployment",
        "namespaces": "v1/Namespace",
        "namespace": "v1/Namespace",
        "configmaps": "v1/ConfigMap",
        "configmap": "v1/ConfigMap",
        "secrets": "v1/Secret",
        "secret": "v1/Secret",
        "routes": "route.openshift.io/v1/Route",
        "route": "route.openshift.io/v1/Route",
        "ingresses": "networking.k8s.io/v1/Ingress",
        "ingress": "networking.k8s.io/v1/Ingress",
        "baremetalhosts": "metal3.io/v1alpha1/BareMetalHost",
        "baremetalhost": "metal3.io/v1alpha1/BareMetalHost",
        "clusterdeployments": "hive.openshift.io/v1/ClusterDeployment",
        "clusterdeployment": "hive.openshift.io/v1/ClusterDeployment",
        "agentclusterinstalls": "extensions.hive.openshift.io/v1beta1/AgentClusterInstall",
        "agentclusterinstall": "extensions.hive.openshift.io/v1beta1/AgentClusterInstall",
    }

    return resource_map.get(resource_name, "")


def map_command_to_api(cmd: str, low: str) -> tuple[str, str]:
    """Map CLI command to equivalent API operation (GVK, verb)"""

    # kubectl/oc create patterns
    if " create " in low:
        if " pod " in low or " pods " in low:
            return "v1/Pod", "create"
        if " service " in low or " svc " in low:
            return "v1/Service", "create"
        if " deployment " in low or " deploy " in low:
            return "apps/v1/Deployment", "create"
        if " route " in low:
            return "route.openshift.io/v1/Route", "create"
        if " namespace " in low or " ns " in low:
            return "v1/Namespace", "create"
        if " configmap " in low:
            return "v1/ConfigMap", "create"
        if " secret " in low:
            return "v1/Secret", "create"
        if " ingress " in low:
            return "networking.k8s.io/v1/Ingress", "create"

    # kubectl/oc get patterns
    if " get " in low:
        if " pod " in low or " pods " in low:
            return "v1/Pod", "get"
        if " service " in low or " svc " in low:
            return "v1/Service", "get"
        if " deployment " in low or " deploy " in low:
            return "apps/v1/Deployment", "get"
        if " route " in low:
            return "route.openshift.io/v1/Route", "get"
        if " namespace " in low or " ns " in low:
            return "v1/Namespace", "get"
        if " configmap " in low:
            return "v1/ConfigMap", "get"
        if " secret " in low:
            return "v1/Secret", "get"
        if " ingress " in low:
            return "networking.k8s.io/v1/Ingress", "get"

    # kubectl/oc delete patterns
    if " delete " in low:
        if " pod " in low or " pods " in low:
            return "v1/Pod", "delete"
        if " service " in low or " svc " in low:
            return "v1/Service", "delete"
        if " deployment " in low or " deploy " in low:
            return "apps/v1/Deployment", "delete"
        if " route " in low:
            return "route.openshift.io/v1/Route", "delete"
        if " namespace " in low or " ns " in low:
            return "v1/Namespace", "delete"
        if " configmap " in low:
            return "v1/ConfigMap", "delete"
        if " secret " in low:
            return "v1/Secret", "delete"
        if " ingress " in low:
            return "networking.k8s.io/v1/Ingress", "delete"

    # kubectl/oc patch patterns
    if " patch " in low:
        if " pod " in low or " pods " in low:
            return "v1/Pod", "patch"
        if " service " in low or " svc " in low:
            return "v1/Service", "patch"
        if " deployment " in low or " deploy " in low:
            return "apps/v1/Deployment", "patch"
        if " route " in low:
            return "route.openshift.io/v1/Route", "patch"
        if " namespace " in low or " ns " in low:
            return "v1/Namespace", "patch"
        if " configmap " in low:
            return "v1/ConfigMap", "patch"
        if " secret " in low:
            return "v1/Secret", "patch"
        if " ingress " in low:
            return "networking.k8s.io/v1/Ingress", "patch"

    # kubectl/oc apply patterns
    if " apply " in low:
        return "unknown/unknown", "apply"

    return "", ""


def api_to_group_version(api: str):
    m = (api or "").lower()
    if "appsv1" in m:
        return ("apps", "v1")
    if m == "corev1api" or m == "v1api" or "corev1" in m:
        return ("", "v1")
    if "batchv1" in m:
        return ("batch", "v1")
    if "rbacauthorizationv1" in m or "rbacv1" in m:
        return ("rbac.authorization.k8s.io", "v1")
    if "networkingv1" in m:
        return ("networking.k8s.io", "v1")
    return ("", "")


def detect_purpose(test_name: str, docstring: str, actions: list, expectations: list) -> str:
    """Analyze test content to determine its purpose"""
    # Combine all text content for analysis
    content = test_name.lower()
    if docstring:
        content += " " + docstring.lower()

    for action in actions:
        if isinstance(action, dict):
            gvk = action.get("gvk", "")
            verb = action.get("verb", "")
            if gvk:
                content += " " + gvk.lower()
            if verb:
                content += " " + verb.lower()

    for exp in expectations:
        if isinstance(exp, dict):
            condition = exp.get("condition", "")
            if condition:
                content += " " + condition.lower()

    # Score each purpose category based on keyword matches
    scores = {}
    for purpose, keywords in PURPOSE_PATTERNS.items():
        score = 0
        for keyword in keywords:
            if keyword in content:
                score += 1
        scores[purpose] = score

    # Find the purpose with the highest score
    max_score = 0
    detected_purpose = "UNKNOWN"
    for purpose, score in scores.items():
        if score > max_score:
            max_score = score
            detected_purpose = purpose

    # If no keywords matched, try to infer from operations
    if max_score == 0:
        detected_purpose = infer_purpose_from_operations(actions)

    return detected_purpose


def infer_purpose_from_operations(actions: list) -> str:
    """Try to infer purpose from the types of operations performed"""
    pod_ops = 0
    network_ops = 0
    storage_ops = 0
    operator_ops = 0
    has_create_delete_update = False
    has_get_list = False

    for action in actions:
        if not isinstance(action, dict):
            continue

        gvk = action.get("gvk", "").lower()
        verb = action.get("verb", "").lower()

        # Count pod-related operations
        if "pod" in gvk or "deployment" in gvk or "replicaset" in gvk:
            pod_ops += 1
            if verb in ["create", "delete", "update"]:
                has_create_delete_update = True
            if verb in ["get", "list"]:
                has_get_list = True

        # Count network-related operations
        if "network" in gvk or "service" in gvk or "ingress" in gvk or "route" in gvk:
            network_ops += 1

        # Count storage-related operations
        if "pvc" in gvk or "pv" in gvk or "storage" in gvk:
            storage_ops += 1

        # Count operator-related operations
        if "operator" in gvk or "subscription" in gvk or "csv" in gvk:
            operator_ops += 1

    # Determine purpose based on operation counts
    if pod_ops > 0 and has_create_delete_update:
        return "POD_MANAGEMENT"
    if pod_ops > 0 and has_get_list:
        return "POD_HEALTH"
    if network_ops > 0:
        return "NETWORK_POLICY"
    if storage_ops > 0:
        return "STORAGE_TESTING"
    if operator_ops > 0:
        return "OPERATOR_MANAGEMENT"

    return "RESOURCE_VALIDATION"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    for root, _, files in os.walk(args.root):
        for fn in files:
            if not (fn.startswith("test_") and fn.endswith(".py")):
                continue
            path = os.path.join(root, fn)
            try:
                src = Path(path).read_text(encoding="utf-8")
                tree = ast.parse(src, filename=path)
                v = TestVisitor(path, args.root)
                v.visit(tree)
                for spec in v.specs:
                    print(json.dumps(spec, ensure_ascii=False))
            except Exception:
                continue


if __name__ == "__main__":
    main()
