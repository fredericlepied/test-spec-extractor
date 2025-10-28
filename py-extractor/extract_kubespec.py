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
    # Upgrade testing patterns - check first for specificity
    "UPGRADE_TESTING": [
        "upgrade",
        "upgrading",
        "upgraded",
        "version.*upgrade",
        "operator.*upgrade",
        "subscription.*upgrade",
        "csv.*upgrade",
        "upgrade.*successfully",
        "upgrade.*target",
        "upgrade.*version",
        "await.*upgrade",
        "upgrade.*await",
        "upgrade.*test",
        "upgrade.*suite",
    ],
    # IP Stack patterns - check first for specificity
    "DUAL_STACK_TESTING": [
        "dual",
        "stack",
        "dualstack",
        "ipv4.*ipv6",
        "ipv6.*ipv4",
        "both.*ip",
        "ipv4.*and.*ipv6",
        "ipv6.*and.*ipv4",
        "dualstack.*ipv4",
        "dualstack.*ipv6",
    ],
    "IPV4_ONLY_TESTING": [
        "ipv4.*only",
        "single.*stack.*ipv4",
        "ipv4.*single",
        "no.*ipv6",
        "ipv4.*no.*ipv6",
        "ipv4.*first",
        "ipv4.*preferred",
        "ipv4.*primary",
    ],
    "IPV6_ONLY_TESTING": [
        "ipv6.*only",
        "single.*stack.*ipv6",
        "ipv6.*single",
        "no.*ipv4",
        "ipv6.*no.*ipv4",
        "ipv6.*first",
        "ipv6.*preferred",
        "ipv6.*primary",
    ],
    # Networking technology patterns
    "SRIOV_TESTING": [
        "sriov",
        "sr-iov",
        "single",
        "root",
        "iov",
        "vf",
        "pf",
        "virtual",
        "function",
        "networkattachment",
    ],
    "PTP_TESTING": [
        "ptp",
        "precision",
        "time",
        "sync",
        "clock",
        "timing",
        "ptpoperator",
    ],
    # Domain-specific purpose patterns - check before general patterns
    "NETWORK_SERVICE_TESTING": [
        "service.*endpoint",
        "endpoint.*pod",
        "endpoint.*unready",
        "endpoint.*ready",
        "service.*connectivity",
        "loadbalancer.*service",
        "nodeport.*service",
        "clusterip.*service",
        "service.*discovery",
        "service.*selector",
    ],
    "CLUSTER_AUTOSCALING": [
        "cluster.*size.*autoscaling",
        "autoscaling.*scale.*up",
        "autoscaling.*scale.*down",
        "node.*autoscaler",
        "cluster.*autoscaler",
        "autoscaling.*mig",
        "managed.*instance.*group",
        "cluster.*size",
        "scale.*node",
        "add.*node",
    ],
    "ZTP_DEPLOYMENT": [
        "ztp",
        "zero.*touch",
        "cluster.*deployment",
        "cluster.*provisioning",
        "site.*config",
        "managed.*cluster",
        "cluster.*install",
        "agent.*cluster",
        "hive.*deployment",
    ],
    # General purpose patterns
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
    "POD_HEALTH": [
        "pods",
        "status",
        "running",
        "phase",
        "health",
        "ready",
        "condition",
        "state",
    ],
    "POD_MANAGEMENT": [
        "create",
        "delete",
        "update",
        "pod",
        "deployment",
        "replica",
        "scale",
    ],
    "NETWORK_POLICY": [
        "policy",
        "network",
        "multinetwork",
        "ingress",
        "egress",
        "security",
    ],
    "RESOURCE_VALIDATION": [
        "count",
        "exist",
        "validation",
        "verify",
        "check",
        "assert",
    ],
    "OPERATOR_MANAGEMENT": [
        "operator",
        "subscription",
        "csv",
        "catalogsource",
        "installplan",
    ],
    "STORAGE_TESTING": ["storage", "volume", "pvc", "pv", "mount", "filesystem"],
    "SECURITY_TESTING": ["security", "rbac", "scc", "psa", "permission", "access"],
    "CONFIGURATION": ["config", "configuration", "settings", "parameters", "env"],
    "PERFORMANCE": [
        "performance",
        "load",
        "stress",
        "benchmark",
        "latency",
        "throughput",
    ],
}

# Test type detection patterns
TEST_TYPE_PATTERNS = {
    "unit": ["test", "Test", "unit", "Unit", "mock", "Mock"],
    "integration": ["integration", "Integration", "e2e", "E2E", "pytest", "Pytest"],
    "performance": [
        "performance",
        "Performance",
        "benchmark",
        "Benchmark",
        "load",
        "Load",
        "stress",
        "Stress",
    ],
    "conformance": [
        "conformance",
        "Conformance",
        "k8s",
        "K8s",
        "kubernetes",
        "Kubernetes",
    ],
}

# Dependency detection patterns
DEPENDENCY_PATTERNS = {
    "operator": ["operator", "Operator", "csv", "CSV", "subscription", "Subscription"],
    "storage": ["storage", "Storage", "pvc", "PVC", "pv", "PV", "volume", "Volume"],
    "network": [
        "network",
        "Network",
        "cni",
        "CNI",
        "multus",
        "Multus",
        "sriov",
        "SR-IOV",
    ],
    "security": ["security", "Security", "rbac", "RBAC", "scc", "SCC", "psa", "PSA"],
    "monitoring": [
        "monitoring",
        "Monitoring",
        "prometheus",
        "Prometheus",
        "grafana",
        "Grafana",
    ],
    "logging": [
        "logging",
        "Logging",
        "fluentd",
        "Fluentd",
        "elasticsearch",
        "Elasticsearch",
    ],
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


def detect_tech(test_name: str, file_path: str, docstring: str) -> list:
    """Detect networking technologies from test name, file path, and docstring."""
    tech = []
    content = (test_name + " " + file_path + " " + (docstring or "")).lower()

    # Check for SR-IOV patterns
    sriov_patterns = [
        "sriov",
        "sr-iov",
        "single.*root.*iov",
        "vf",
        "pf",
        "virtual.*function",
        "networkattachment",
    ]
    for pattern in sriov_patterns:
        if re.search(pattern, content):
            tech.append("SR-IOV")
            break

    # Check for PTP patterns
    ptp_patterns = [
        "ptp",
        "precision.*time",
        "time.*sync",
        "clock.*sync",
        "timing",
        "ptpoperator",
    ]
    for pattern in ptp_patterns:
        if re.search(pattern, content):
            tech.append("PTP")
            break

    # Check for DPDK patterns
    dpdk_patterns = ["dpdk", "data.*plane.*development.*kit", "userspace.*networking"]
    for pattern in dpdk_patterns:
        if re.search(pattern, content):
            tech.append("DPDK")
            break

    # Check for MetalLB patterns
    metallb_patterns = [
        "metallb",
        "metal.*lb",
        "load.*balancer",
        "bfd",
        "bgp.*multiservice",
        "bgp.*unnumbered",
    ]
    for pattern in metallb_patterns:
        if re.search(pattern, content):
            tech.append("MetalLB")
            break

    # Check for GPU patterns
    gpu_patterns = [
        "gpu",
        "cuda",
        "nvidia",
        "amd.*gpu",
        "amdgpu",
        "kmm",
        "kernel.*module.*manager",
        "device.*plugin",
    ]
    for pattern in gpu_patterns:
        if re.search(pattern, content):
            tech.append("GPU")
            break

    # Check for RDMA patterns
    rdma_patterns = [
        "rdma",
        "infiniband",
        "roce",
        "rdma.*metrics",
        "rdma.*api",
        "remote.*direct.*memory.*access",
    ]
    for pattern in rdma_patterns:
        if re.search(pattern, content):
            tech.append("RDMA")
            break

    # Check for bonding patterns
    bond_patterns = ["bond", "bonding", "team", "link.*aggregation", "failover"]
    for pattern in bond_patterns:
        if re.search(pattern, content):
            tech.append("Bonding")
            break

    # Check for CNI patterns
    cni_patterns = [
        "cni",
        "container.*network.*interface",
        "tap",
        "macvlan",
        "bridge",
        "vlan",
    ]
    for pattern in cni_patterns:
        if re.search(pattern, content):
            tech.append("CNI")
            break

    # Check for power management patterns
    power_patterns = [
        "power.*save",
        "powersave",
        "power.*management",
        "cpu.*frequency",
        "energy.*efficiency",
    ]
    for pattern in power_patterns:
        if re.search(pattern, content):
            tech.append("Power Management")
            break

    # Check for virtualization patterns
    virt_patterns = [
        "kvm",
        "qemu",
        "libvirt",
        "virt",
        "vm",
        "virtualization",
        "hypervisor",
        "container",
        "docker",
        "podman",
        "crio",
        "containerd",
        "runc",
        "cri-o",
        "dockerfile",
        "buildah",
        "skopeo",
    ]
    for pattern in virt_patterns:
        if re.search(pattern, content):
            tech.append("Virtualization")
            break

    # Check for storage patterns
    storage_patterns = [
        "ceph",
        "gluster",
        "nfs",
        "iscsi",
        "lvm",
        "zfs",
        "snapshot",
        "backup",
        "storage",
        "volume",
        "pvc",
        "pv",
    ]
    for pattern in storage_patterns:
        if re.search(pattern, content):
            tech.append("Storage")
            break

    # Check for security patterns
    security_patterns = [
        "selinux",
        "apparmor",
        "seccomp",
        "firewall",
        "iptables",
        "nftables",
        "tls",
        "ssl",
        "certificate",
        "encryption",
        "rbac",
        "scc",
        "psa",
    ]
    for pattern in security_patterns:
        if re.search(pattern, content):
            tech.append("Security")
            break

    # Check for monitoring/observability patterns
    monitoring_patterns = [
        "prometheus",
        "grafana",
        "alertmanager",
        "metrics",
        "logging",
        "fluentd",
        "elasticsearch",
        "kibana",
        "jaeger",
        "tracing",
        "telemetry",
        "monitoring",
    ]
    for pattern in monitoring_patterns:
        if re.search(pattern, content):
            tech.append("Monitoring")
            break

    # Check for machine learning/AI patterns
    ml_patterns = [
        "tensorflow",
        "pytorch",
        "onnx",
        r"\bml\b",
        r"\bai\b",
        "inference",
        "model",
        "training",
        "neural",
        "deep.*learning",
    ]
    for pattern in ml_patterns:
        if re.search(pattern, content):
            tech.append("Machine Learning")
            break

    # Check for edge computing patterns
    edge_patterns = [
        "edge",
        "iot",
        "5g",
        "latency",
        "real.*time",
        "time.*sensitive",
        "industrial",
        "automation",
    ]
    for pattern in edge_patterns:
        if re.search(pattern, content):
            tech.append("Edge Computing")
            break

    # Check for networking protocols
    protocol_patterns = [
        "tcp",
        "udp",
        "http",
        "https",
        "grpc",
        "websocket",
        "mqtt",
        "coap",
        "snmp",
        "bgp",
        "ospf",
        "isis",
        "ldp",
        "rsvp",
        "mpls",
        "vxlan",
        "geneve",
        "gre",
        "ipsec",
        "tls",
        "ssl",
    ]
    for pattern in protocol_patterns:
        if re.search(pattern, content):
            tech.append("Networking Protocols")
            break

    # Check for service mesh patterns
    servicemesh_patterns = [
        "istio",
        "linkerd",
        "consul.*connect",
        "envoy",
        "service.*mesh",
        "sidecar",
        "proxy",
    ]
    for pattern in servicemesh_patterns:
        if re.search(pattern, content):
            tech.append("Service Mesh")
            break

    # Check for API gateway patterns
    apigateway_patterns = [
        "api.*gateway",
        "kong",
        "ambassador",
        "traefik",
        "nginx.*ingress",
        "haproxy",
        "api.*management",
    ]
    for pattern in apigateway_patterns:
        if re.search(pattern, content):
            tech.append("API Gateway")
            break

    # Check for database patterns
    database_patterns = [
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "cassandra",
        "elasticsearch",
        "influxdb",
        "timescaledb",
        "database",
        "db",
        "sql",
        "nosql",
    ]
    for pattern in database_patterns:
        if re.search(pattern, content):
            tech.append("Database")
            break

    # Check for messaging patterns
    messaging_patterns = [
        "kafka",
        "rabbitmq",
        "activemq",
        "nats",
        "pulsar",
        "message.*queue",
        "event.*streaming",
        "pub.*sub",
    ]
    for pattern in messaging_patterns:
        if re.search(pattern, content):
            tech.append("Messaging")
            break

    # Check for CI/CD patterns
    cicd_patterns = [
        "jenkins",
        "gitlab.*ci",
        "github.*actions",
        "tekton",
        "argo",
        "flux",
        "ci.*cd",
        "continuous.*integration",
        "continuous.*deployment",
        "pipeline",
    ]
    for pattern in cicd_patterns:
        if re.search(pattern, content):
            tech.append("CI/CD")
            break

    return tech


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
            # Enhanced target classification for similarity search compatibility
            target = "test_condition"  # Default to match Go extractor

            # Resource count patterns
            if "len(" in left and any(
                resource in left.lower()
                for resource in [
                    "baremetalhost",
                    "clusterdeployment",
                    "namespace",
                    "pod",
                    "service",
                    "pvc",
                    "pv",
                    "deployment",
                    "replicaset",
                    "statefulset",
                    "daemonset",
                    "configmap",
                    "secret",
                    "ingress",
                    "route",
                    "networkpolicy",
                    "scc",
                    "subscription",
                    "csv",
                    "operatorgroup",
                    "policy",
                    "managedcluster",
                ]
            ):
                target = "resource_count"
            # Resource status patterns
            elif any(
                status in left.lower()
                for status in [
                    "online",
                    "status",
                    "phase",
                    "state",
                    "ready",
                    "running",
                    "bound",
                    "available",
                    "active",
                    "healthy",
                    "succeeded",
                    "failed",
                    "pending",
                ]
            ):
                target = "resource_status"
            # Resource version patterns
            elif any(
                version in left.lower()
                for version in [
                    "version",
                    "generation",
                    "resourceversion",
                    "uid",
                    "creationtimestamp",
                ]
            ):
                target = "resource_version"
            # Resource deletion patterns
            elif any(
                deletion in left.lower()
                for deletion in [
                    "deleted",
                    "removed",
                    "not found",
                    "absent",
                    "empty",
                    "none",
                ]
            ):
                target = "resource_deletion"
            return {"target": target, "condition": condition}

    # Handle membership tests (in, not in)
    elif isinstance(assert_node.test, ast.Compare) and len(assert_node.test.ops) == 1:
        op = assert_node.test.ops[0]
        if isinstance(op, (ast.In, ast.NotIn)):
            left = ast_to_string(assert_node.test.left)
            right = ast_to_string(assert_node.test.comparators[0])
            condition = f"{left} {'not in' if isinstance(op, ast.NotIn) else 'in'} {right}"
            return {
                "target": ("resource_deletion" if isinstance(op, ast.NotIn) else "test_condition"),
                "condition": condition,
            }

    # Handle boolean operations (and, or)
    elif isinstance(assert_node.test, ast.BoolOp):
        op = "and" if isinstance(assert_node.test.op, ast.And) else "or"
        values = [ast_to_string(v) for v in assert_node.test.values]
        condition = f" {op} ".join(values)
        return {"target": "compound_condition", "condition": condition}

    # Handle unittest-style assertions (self.assertEqual, self.assertTrue, etc.)
    elif isinstance(assert_node.test, ast.Call) and isinstance(
        assert_node.test.func, ast.Attribute
    ):
        func_name = assert_node.test.func.attr
        args = [ast_to_string(arg) for arg in assert_node.test.args]

        if func_name in ["assertEqual", "assertEquals"] and len(args) >= 2:
            condition = f"{args[0]} == {args[1]}"
            target = "test_condition"
            if "len(" in args[0] and any(
                resource in args[0].lower()
                for resource in [
                    "baremetalhost",
                    "clusterdeployment",
                    "namespace",
                    "pod",
                    "service",
                    "pvc",
                    "pv",
                    "deployment",
                    "replicaset",
                    "statefulset",
                    "daemonset",
                ]
            ):
                target = "resource_count"
            return {"target": target, "condition": condition}
        elif func_name in ["assertTrue", "assertFalse"] and len(args) >= 1:
            condition = f"{args[0]} is {func_name == 'assertTrue'}"
            return {"target": "test_condition", "condition": condition}
        elif func_name in ["assertIn", "assertNotIn"] and len(args) >= 2:
            condition = f"{args[1]} {'in' if func_name == 'assertIn' else 'not in'} {args[0]}"
            return {"target": "test_condition", "condition": condition}
        elif func_name in ["assertIsNone", "assertIsNotNone"] and len(args) >= 1:
            condition = f"{args[0]} is {'None' if func_name == 'assertIsNone' else 'not None'}"
            return {"target": "test_condition", "condition": condition}
        else:
            condition = f"{func_name}({', '.join(args)})"
            return {"target": "test_condition", "condition": condition}

    # Handle pytest-style assertions (pytest.raises, etc.)
    elif isinstance(assert_node.test, ast.Call) and isinstance(
        assert_node.test.func, ast.Attribute
    ):
        if hasattr(assert_node.test.func, "value") and isinstance(
            assert_node.test.func.value, ast.Name
        ):
            if (
                assert_node.test.func.value.id == "pytest"
                and assert_node.test.func.attr == "raises"
            ):
                condition = f"pytest.raises({', '.join(ast_to_string(arg) for arg in assert_node.test.args)})"
                return {"target": "test_condition", "condition": condition}

    # Handle simple boolean assertions
    elif isinstance(assert_node.test, ast.Constant):
        condition = str(assert_node.test.value)
        return {"target": "test_condition", "condition": condition}

    # Handle call expressions (e.g., assert some_function())
    elif isinstance(assert_node.test, ast.Call):
        condition = ast_to_string(assert_node.test)
        return {"target": "test_condition", "condition": condition}

    # Generic assertion
    condition = ast_to_string(assert_node.test)
    return {"target": "test_condition", "condition": condition}


class TestVisitor(ast.NodeVisitor):
    def __init__(self, path, root_path):
        self.path = path
        self.root_path = root_path
        self.specs = []
        self.imports = {}  # Track imports for cross-file detection
        self.imported_modules = set()  # Track imported modules

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Enhanced test function detection patterns
        is_test_function = (
            node.name.startswith("test")  # pytest style
            or node.name.startswith("Test")  # unittest style
            or node.name.startswith("test_")  # explicit pytest style
            or node.name.startswith("test_")  # explicit test prefix
            or any(
                decorator.id == "pytest.mark.parametrize"
                or (hasattr(decorator, "attr") and decorator.attr == "parametrize")
                for decorator in node.decorator_list
                if hasattr(decorator, "id") or hasattr(decorator, "attr")
            )  # pytest parametrize
            or any(
                hasattr(decorator, "id")
                and decorator.id
                in [
                    "retry",
                    "retry_if_exception_type",
                    "stop_after_attempt",
                    "wait_fixed",
                ]
                for decorator in node.decorator_list
                if hasattr(decorator, "id")
            )  # retry decorators
        )

        # Check for Ginkgo patterns in the function
        has_ginkgo_patterns = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and is_ginkgo_call(child):
                has_ginkgo_patterns = True
                break

        # Check for step-like patterns (non-Ginkgo)
        has_step_patterns = False
        for child in ast.walk(node):
            if (
                (isinstance(child, ast.Call) and is_step_like_call(child))
                or (isinstance(child, ast.FunctionDef) and is_step_like_function(child))
                or (isinstance(child, ast.Expr) and is_step_comment(child))
            ):
                has_step_patterns = True
                break

        if not is_test_function and not has_ginkgo_patterns and not has_step_patterns:
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
            "tech": [],
            "by_steps": [],  # Track operations in By(...) steps
        }
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "parametrize":
                spec["dependencies"].append("parametrized")
        # First, detect cross-file actions
        cross_file_actions = self.detect_cross_file_actions(node)
        spec["actions"].extend(cross_file_actions)

        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func = n.func

                # Check for direct helper function calls
                if isinstance(func, ast.Name) and func.id in [
                    "get_resource",
                    "get_resource_from_namespace",
                    "get_namespaces",
                    "create_api_object",
                    "delete_object",
                ]:
                    if func.id == "get_namespaces":
                        # get_namespaces() -> v1/Namespace:list
                        spec["actions"].append({"gvk": "v1/Namespace", "verb": "list"})
                    elif func.id == "create_api_object":
                        # create_api_object() -> generic create operation
                        spec["actions"].append({"gvk": "unknown/unknown", "verb": "create"})
                    elif func.id == "delete_object":
                        # delete_object() -> generic delete operation
                        spec["actions"].append({"gvk": "unknown/unknown", "verb": "delete"})
                    elif func.id in ["get_resource", "get_resource_from_namespace"]:
                        if n.args and isinstance(n.args[0], ast.Constant):
                            resource_name = n.args[0].value
                            gvk = map_resource_name_to_gvk(resource_name)
                            if gvk:
                                # Determine verb based on function name and arguments
                                verb = "list" if func.id == "get_resource" else "get"
                                spec["actions"].append({"gvk": gvk, "verb": verb})
                        elif n.args:
                            # For any arguments (including variables), add a generic action
                            verb = "list" if func.id == "get_resource" else "get"
                            spec["actions"].append({"gvk": "unknown/unknown", "verb": verb})

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

                    # Check for imported helper function calls
                    if func.attr in [
                        "get_resource",
                        "get_resource_from_namespace",
                        "get_namespaces",
                        "create_api_object",
                        "delete_object",
                    ]:
                        if func.attr == "get_namespaces":
                            # get_namespaces() -> v1/Namespace:list
                            spec["actions"].append({"gvk": "v1/Namespace", "verb": "list"})
                        elif func.attr == "create_api_object":
                            # create_api_object() -> generic create operation
                            spec["actions"].append({"gvk": "unknown/unknown", "verb": "create"})
                        elif func.attr == "delete_object":
                            # delete_object() -> generic delete operation
                            spec["actions"].append({"gvk": "unknown/unknown", "verb": "delete"})
                        elif func.attr in [
                            "get_resource",
                            "get_resource_from_namespace",
                        ]:
                            if n.args and isinstance(n.args[0], ast.Constant):
                                resource_name = n.args[0].value
                                gvk = map_resource_name_to_gvk(resource_name)
                                if gvk:
                                    # Determine verb based on function name
                                    verb = "list" if func.attr == "get_resource" else "get"
                                    spec["actions"].append({"gvk": gvk, "verb": verb})
                            elif n.args:
                                # For any arguments (including variables), add a generic action
                                verb = "list" if func.attr == "get_resource" else "get"
                                spec["actions"].append({"gvk": "unknown/unknown", "verb": verb})
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
            spec["test_type"] = "integration"

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
        spec["dependencies"].extend(
            detect_dependencies(node.name, self.path, docstring, spec["actions"])
        )
        spec["environment"] = detect_environment(node.name, self.path, docstring)
        spec["tech"] = detect_tech(node.name, self.path, docstring)

        # Analyze By steps for more granular operation tracking
        if has_ginkgo_patterns:
            by_steps = analyze_by_steps_in_file(node)
            spec["by_steps"] = by_steps
        elif has_step_patterns:
            # Use non-Ginkgo step patterns
            step_patterns = analyze_step_patterns_in_file(node)
            spec["by_steps"] = step_patterns

        # Detect purpose based on test content
        spec["purpose"] = detect_purpose(
            node.name, docstring, spec["actions"], spec["expectations"]
        )

        self.specs.append(spec)

    def visit_ClassDef(self, node: ast.ClassDef):
        # Enhanced test class detection patterns
        is_test_class = (
            node.name.startswith("Test")  # unittest style
            or node.name.startswith("test")  # pytest style
            or any(
                base.id == "TestCase" for base in node.bases if hasattr(base, "id")
            )  # unittest.TestCase
            or any(
                hasattr(base, "id") and base.id in ["unittest.TestCase", "TestCase"]
                for base in node.bases
                if hasattr(base, "id")
            )  # unittest.TestCase
            or any(
                hasattr(base, "attr") and base.attr == "TestCase"
                for base in node.bases
                if hasattr(base, "attr")
            )  # unittest.TestCase
        )

        if not is_test_class:
            return

        # Process test methods in the class
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test"):
                # Create test_id with basename substitution
                basename = os.path.basename(self.root_path)
                relative_path = os.path.relpath(self.path, self.root_path)
                spec = {
                    "test_id": f"{basename}/{relative_path}:{node.name}.{item.name}",
                    "test_type": "unit",  # Class-based tests are typically unit tests
                    "dependencies": [],
                    "environment": [],
                    "actions": [],
                    "expectations": [],
                    "openshift_specific": [],
                    "concurrency": [],
                    "artifacts": [],
                    "purpose": "",
                    "tech": [],
                    "by_steps": [],  # Track operations in By(...) steps
                }

                # Extract docstring
                docstring = ast.get_docstring(item)
                if docstring:
                    spec["purpose"] = detect_purpose(
                        item.name, docstring, spec["actions"], spec["expectations"]
                    )
                else:
                    spec["purpose"] = "UNKNOWN"

                # Detect test type, dependencies, environment
                spec["test_type"] = detect_test_type(item.name, self.path, docstring)
                spec["dependencies"] = detect_dependencies(item.name, self.path, docstring)
                spec["environment"] = detect_environment(item.name, self.path, docstring)

                # Process function body for actions and expectations
                # First, detect cross-file actions
                cross_file_actions = self.detect_cross_file_actions(item)
                spec["actions"].extend(cross_file_actions)

                for n in ast.walk(item):
                    if isinstance(n, ast.Call):
                        # Check for direct helper function calls
                        if isinstance(n.func, ast.Name) and n.func.id in [
                            "get_resource",
                            "get_resource_from_namespace",
                            "get_namespaces",
                            "create_api_object",
                            "delete_object",
                        ]:
                            if n.func.id == "get_namespaces":
                                spec["actions"].append({"gvk": "v1/Namespace", "verb": "list"})
                            elif n.func.id == "create_api_object":
                                spec["actions"].append({"gvk": "unknown/unknown", "verb": "create"})
                            elif n.func.id == "delete_object":
                                spec["actions"].append({"gvk": "unknown/unknown", "verb": "delete"})
                            elif n.func.id in [
                                "get_resource",
                                "get_resource_from_namespace",
                            ]:
                                if n.args and isinstance(n.args[0], ast.Constant):
                                    resource_name = n.args[0].value
                                    gvk = map_resource_name_to_gvk(resource_name)
                                    if gvk:
                                        verb = "list" if n.func.id == "get_resource" else "get"
                                        spec["actions"].append({"gvk": gvk, "verb": verb})
                                elif n.args:
                                    verb = "list" if n.func.id == "get_resource" else "get"
                                    spec["actions"].append({"gvk": "unknown/unknown", "verb": verb})
                        # Check for imported helper function calls (e.g., oc_helpers.get_resource)
                        elif isinstance(n.func, ast.Attribute) and n.func.attr in [
                            "get_resource",
                            "get_resource_from_namespace",
                            "get_namespaces",
                            "create_api_object",
                            "delete_object",
                        ]:
                            if n.func.attr == "get_namespaces":
                                spec["actions"].append({"gvk": "v1/Namespace", "verb": "list"})
                            elif n.func.attr == "create_api_object":
                                spec["actions"].append({"gvk": "unknown/unknown", "verb": "create"})
                            elif n.func.attr == "delete_object":
                                spec["actions"].append({"gvk": "unknown/unknown", "verb": "delete"})
                            elif n.func.attr in [
                                "get_resource",
                                "get_resource_from_namespace",
                            ]:
                                if n.args and isinstance(n.args[0], ast.Constant):
                                    resource_name = n.args[0].value
                                    gvk = map_resource_name_to_gvk(resource_name)
                                    if gvk:
                                        verb = "list" if n.func.attr == "get_resource" else "get"
                                        spec["actions"].append({"gvk": gvk, "verb": verb})
                                elif n.args:
                                    verb = "list" if n.func.attr == "get_resource" else "get"
                                    spec["actions"].append({"gvk": "unknown/unknown", "verb": verb})
                        # Check for subprocess calls (CLI commands)
                        elif isinstance(n.func, ast.Attribute) and n.func.attr == "run":
                            if (
                                isinstance(n.func.value, ast.Name)
                                and n.func.value.id == "subprocess"
                            ):
                                if n.args and isinstance(n.args[0], ast.List):
                                    cmd_parts = []
                                    for arg in n.args[0].elts:
                                        if isinstance(arg, ast.Constant):
                                            cmd_parts.append(arg.value)
                                    if cmd_parts:
                                        cmd = " ".join(cmd_parts)
                                        gvk, verb = map_command_to_api(cmd)
                                        if gvk and verb:
                                            spec["actions"].append({"gvk": gvk, "verb": verb})
                    elif isinstance(n, ast.Assert):
                        expectation = extract_assertion_expectation(n)
                        if expectation:
                            spec["expectations"].append(expectation)

                # Add tech detection
                spec["tech"] = detect_tech(item.name, self.path, docstring)

                # Check for Ginkgo patterns and analyze By steps
                has_ginkgo_patterns = False
                for child in ast.walk(item):
                    if isinstance(child, ast.Call) and is_ginkgo_call(child):
                        has_ginkgo_patterns = True
                        break

                # Check for step-like patterns (non-Ginkgo)
                has_step_patterns = False
                for child in ast.walk(item):
                    if (
                        (isinstance(child, ast.Call) and is_step_like_call(child))
                        or (isinstance(child, ast.FunctionDef) and is_step_like_function(child))
                        or (isinstance(child, ast.Expr) and is_step_comment(child))
                    ):
                        has_step_patterns = True
                        break

                if has_ginkgo_patterns:
                    by_steps = analyze_by_steps_in_file(item)
                    spec["by_steps"] = by_steps
                elif has_step_patterns:
                    # Use non-Ginkgo step patterns
                    step_patterns = analyze_step_patterns_in_file(item)
                    spec["by_steps"] = step_patterns

                self.specs.append(spec)

    def visit_Import(self, node: ast.Import):
        """Track import statements for cross-file detection"""
        for alias in node.names:
            if alias.asname:
                self.imports[alias.asname] = alias.name
            else:
                self.imports[alias.name] = alias.name
            self.imported_modules.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from-import statements for cross-file detection"""
        module_name = node.module or ""
        for alias in node.names:
            if alias.asname:
                self.imports[alias.asname] = f"{module_name}.{alias.name}"
            else:
                self.imports[alias.name] = f"{module_name}.{alias.name}"
            self.imported_modules.add(module_name)

    def detect_cross_file_actions(self, node: ast.FunctionDef) -> list:
        """Detect actions from cross-file function calls"""
        actions = []

        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                # Check for calls to imported functions
                if isinstance(n.func, ast.Name) and n.func.id in self.imports:
                    imported_name = self.imports[n.func.id]
                    # Check if it's a known helper function
                    if any(
                        helper in imported_name
                        for helper in [
                            "get_resource",
                            "get_resource_from_namespace",
                            "get_namespaces",
                            "create_api_object",
                            "delete_object",
                            "oc_helpers",
                        ]
                    ):
                        if "get_namespaces" in imported_name:
                            actions.append({"gvk": "v1/Namespace", "verb": "list"})
                        elif "create_api_object" in imported_name:
                            actions.append({"gvk": "unknown/unknown", "verb": "create"})
                        elif "delete_object" in imported_name:
                            actions.append({"gvk": "unknown/unknown", "verb": "delete"})
                        elif "get_resource" in imported_name:
                            if n.args and isinstance(n.args[0], ast.Constant):
                                resource_name = n.args[0].value
                                gvk = map_resource_name_to_gvk(resource_name)
                                if gvk:
                                    verb = "list" if "get_resource" in imported_name else "get"
                                    actions.append({"gvk": gvk, "verb": verb})
                            else:
                                verb = "list" if "get_resource" in imported_name else "get"
                                actions.append({"gvk": "unknown/unknown", "verb": verb})

                # Check for calls to functions from imported modules
                elif isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and n.func.value.id in self.imports:
                        module_name = self.imports[n.func.value.id]
                        func_name = n.func.attr

                        # Check for common patterns in eco-pytests
                        if any(
                            pattern in module_name
                            for pattern in ["oc_helpers", "utils", "eco_pytests"]
                        ) and func_name in [
                            "get_resource",
                            "get_resource_from_namespace",
                            "get_namespaces",
                            "create_api_object",
                            "delete_object",
                        ]:
                            if func_name == "get_namespaces":
                                actions.append({"gvk": "v1/Namespace", "verb": "list"})
                            elif func_name == "create_api_object":
                                actions.append({"gvk": "unknown/unknown", "verb": "create"})
                            elif func_name == "delete_object":
                                actions.append({"gvk": "unknown/unknown", "verb": "delete"})
                            elif func_name in [
                                "get_resource",
                                "get_resource_from_namespace",
                            ]:
                                if n.args and isinstance(n.args[0], ast.Constant):
                                    resource_name = n.args[0].value
                                    gvk = map_resource_name_to_gvk(resource_name)
                                    if gvk:
                                        verb = "list" if func_name == "get_resource" else "get"
                                        actions.append({"gvk": gvk, "verb": verb})
                                else:
                                    verb = "list" if func_name == "get_resource" else "get"
                                    actions.append({"gvk": "unknown/unknown", "verb": verb})

        return actions


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
        # Additional resources found in tests
        "policies": "policy.open-cluster-management.io/v1/Policy",
        "policy": "policy.open-cluster-management.io/v1/Policy",
        "managedclusters": "cluster.open-cluster-management.io/v1/ManagedCluster",
        "managedcluster": "cluster.open-cluster-management.io/v1/ManagedCluster",
        "persistentvolumes": "v1/PersistentVolume",
        "persistentvolume": "v1/PersistentVolume",
        "persistentvolumeclaims": "v1/PersistentVolumeClaim",
        "persistentvolumeclaim": "v1/PersistentVolumeClaim",
        "nodes": "v1/Node",
        "node": "v1/Node",
        "daemonsets": "apps/v1/DaemonSet",
        "daemonset": "apps/v1/DaemonSet",
        "replicasets": "apps/v1/ReplicaSet",
        "replicaset": "apps/v1/ReplicaSet",
        "statefulsets": "apps/v1/StatefulSet",
        "statefulset": "apps/v1/StatefulSet",
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


def is_ginkgo_call(node: ast.Call) -> bool:
    """Check if a call expression is a Ginkgo test construct"""
    if isinstance(node.func, ast.Name):
        return node.func.id in [
            "Describe",
            "It",
            "By",
            "Context",
            "BeforeAll",
            "AfterAll",
            "BeforeEach",
            "AfterEach",
            "JustBeforeEach",
            "JustAfterEach",
            "Specify",
            "When",
            "Given",
            "Then",
            "And",
            "But",
            "FDescribe",
            "FIt",
            "FContext",
            "FWhen",
            "FSpecify",
            "PDescribe",
            "PIt",
            "PContext",
            "PWhen",
            "PSpecify",
            "XDescribe",
            "XIt",
            "XContext",
            "XWhen",
            "XSpecify",
        ]
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr in [
            "Describe",
            "It",
            "By",
            "Context",
            "BeforeAll",
            "AfterAll",
            "BeforeEach",
            "AfterEach",
            "JustBeforeEach",
            "JustAfterEach",
            "Specify",
            "When",
            "Given",
            "Then",
            "And",
            "But",
            "FDescribe",
            "FIt",
            "FContext",
            "FWhen",
            "FSpecify",
            "PDescribe",
            "PIt",
            "PContext",
            "PWhen",
            "PSpecify",
            "XDescribe",
            "XIt",
            "XContext",
            "XWhen",
            "XSpecify",
        ]
    return False


def is_step_like_call(node: ast.Call) -> bool:
    """Check if a call expression represents a step-like pattern (non-Ginkgo)"""
    if isinstance(node.func, ast.Name):
        # Common step-like function names
        step_names = [
            "step",
            "given",
            "when",
            "then",
            "and",
            "but",  # BDD patterns
            "setup",
            "teardown",
            "before",
            "after",  # Setup/teardown patterns
            "arrange",
            "act",
            "assert",  # AAA pattern
            "prepare",
            "execute",
            "verify",
            "cleanup",  # Test phases
            "precondition",
            "action",
            "postcondition",  # Formal testing
            "initialize",
            "configure",
            "deploy",
            "test",
            "validate",
            "cleanup",  # Common test steps
        ]
        return node.func.id.lower() in step_names

    elif isinstance(node.func, ast.Attribute):
        # Method calls that might be step-like
        attr_name = node.func.attr.lower()
        step_patterns = [
            "step",
            "given",
            "when",
            "then",
            "and",
            "but",
            "setup",
            "teardown",
            "before",
            "after",
            "arrange",
            "act",
            "assert",
            "prepare",
            "execute",
            "verify",
            "cleanup",
            "precondition",
            "action",
            "postcondition",
            "initialize",
            "configure",
            "deploy",
            "test",
            "validate",
            "cleanup",
        ]
        return attr_name in step_patterns

    return False


def is_step_comment(node: ast.Expr) -> bool:
    """Check if a comment represents a step description"""
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        comment = node.value.value.lower().strip()
        step_indicators = [
            "step",
            "given",
            "when",
            "then",
            "and",
            "but",
            "setup",
            "teardown",
            "before",
            "after",
            "arrange",
            "act",
            "assert",
            "prepare",
            "execute",
            "verify",
            "cleanup",
            "precondition",
            "action",
            "postcondition",
            "initialize",
            "configure",
            "deploy",
            "test",
            "validate",
            "cleanup",
        ]
        return any(indicator in comment for indicator in step_indicators)
    return False


def is_step_decorator(node: ast.Call) -> bool:
    """Check if a decorator represents a step pattern"""
    if isinstance(node.func, ast.Name):
        decorator_names = [
            "step",
            "given",
            "when",
            "then",
            "and",
            "but",
            "setup",
            "teardown",
            "before",
            "after",
            "arrange",
            "act",
            "assert",
            "prepare",
            "execute",
            "verify",
            "cleanup",
        ]
        return node.func.id.lower() in decorator_names

    elif isinstance(node.func, ast.Attribute):
        return node.func.attr.lower() in ["step", "given", "when", "then", "and", "but"]

    return False


def is_by_call(node: ast.Call) -> bool:
    """Check if a call expression is a By(...) call"""
    if isinstance(node.func, ast.Name):
        return node.func.id == "By"
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr == "By"
    return False


def extract_by_description(by_call: ast.Call) -> str:
    """Extract the description from a By(...) call"""
    if not by_call.args:
        return ""

    # Get the first argument (the description)
    arg = by_call.args[0]
    if isinstance(arg, ast.Constant):
        return str(arg.value)
    elif hasattr(ast, "unparse"):
        return ast.unparse(arg)
    else:
        return str(arg)


def extract_step_description(step_call: ast.Call) -> str:
    """Extract the description from a step-like call"""
    if not step_call.args:
        return ""

    # Get the first argument (the description)
    arg = step_call.args[0]
    if isinstance(arg, ast.Constant):
        return str(arg.value)
    elif hasattr(ast, "unparse"):
        return ast.unparse(arg)
    else:
        return str(arg)


def extract_function_name_as_description(func_def: ast.FunctionDef) -> str:
    """Extract function name as step description"""
    return func_def.name.replace("_", " ").replace("test", "").strip()


def extract_comment_as_description(comment_node: ast.Expr) -> str:
    """Extract comment text as step description"""
    if isinstance(comment_node.value, ast.Constant) and isinstance(comment_node.value.value, str):
        return comment_node.value.value.strip()
    return ""


def find_operations_in_by_step(by_call: ast.Call) -> list:
    """Find all operations within a By(...) step"""
    operations = []

    # For By(...) calls, we need to look at the function passed as the second argument
    # By("description", func() { ... })
    if len(by_call.args) >= 2:
        func_arg = by_call.args[1]
        if isinstance(func_arg, ast.Lambda):
            # Analyze the lambda body for operations
            operations.extend(analyze_node_for_operations(func_arg.body))
        elif isinstance(func_arg, ast.Call):
            # If it's a function call, analyze it
            operations.extend(analyze_node_for_operations(func_arg))

    return operations


def analyze_node_for_operations(node: ast.AST) -> list:
    """Analyze an AST node for Kubernetes operations"""
    operations = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # Check for direct helper function calls
            if isinstance(child.func, ast.Name) and child.func.id in [
                "get_resource",
                "get_resource_from_namespace",
                "get_namespaces",
                "create_api_object",
                "delete_object",
            ]:
                if child.func.id == "get_namespaces":
                    operations.append({"gvk": "v1/Namespace", "verb": "list"})
                elif child.func.id == "create_api_object":
                    operations.append({"gvk": "unknown/unknown", "verb": "create"})
                elif child.func.id == "delete_object":
                    operations.append({"gvk": "unknown/unknown", "verb": "delete"})
                elif child.func.id in ["get_resource", "get_resource_from_namespace"]:
                    if child.args and isinstance(child.args[0], ast.Constant):
                        resource_name = child.args[0].value
                        gvk = map_resource_name_to_gvk(resource_name)
                        if gvk:
                            verb = "list" if child.func.id == "get_resource" else "get"
                            operations.append({"gvk": gvk, "verb": verb})
                    else:
                        verb = "list" if child.func.id == "get_resource" else "get"
                        operations.append({"gvk": "unknown/unknown", "verb": verb})

            # Check for attribute-based calls
            elif isinstance(child.func, ast.Attribute):
                attr_name = child.func.attr or ""
                low = attr_name.lower()
                if low.startswith(VERB_PREFIXES):
                    verb = low.split("_", 1)[0]
                    kind = low.split("_")[-1]
                    operations.append(
                        {
                            "gvk": guess_gvk_from_attr(child.func),
                            "verb": verb.replace("read", "get"),
                            "fields": {"kind_hint": kind.capitalize()},
                        }
                    )

                # Check for subprocess calls (CLI commands)
                if attr_name in ["run", "check_call", "check_output"]:
                    if getattr(child.func.value, "id", "") == "subprocess":
                        args = []
                        for a in child.args:
                            if isinstance(a, ast.List):
                                for el in a.elts:
                                    if isinstance(el, ast.Constant):
                                        args.append(str(el.value))
                            elif isinstance(a, ast.Constant):
                                args.append(str(a.value))
                        cmd = " ".join(args)
                        if CLI_RE.search(cmd):
                            low_cmd = cmd.lower()
                            gvk, verb = map_command_to_api(cmd, low_cmd)
                            if gvk and verb:
                                operations.append({"gvk": gvk, "verb": verb})

    return operations


def analyze_by_steps_in_file(tree: ast.AST) -> list:
    """Analyze a file to find By(...) calls and their associated operations"""
    by_steps = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_by_call(node):
            description = extract_by_description(node)
            operations = find_operations_in_by_step(node)

            # Add By step information to each operation
            for operation in operations:
                operation["by_step"] = description

            by_step = {
                "description": description,
                "actions": operations,
                "line": getattr(node, "lineno", 0),
            }
            by_steps.append(by_step)

    return by_steps


def analyze_step_patterns_in_file(tree: ast.AST) -> list:
    """Analyze a file to find step-like patterns (non-Ginkgo) and their associated operations"""
    steps = []

    for node in ast.walk(tree):
        # Check for step-like function calls
        if isinstance(node, ast.Call) and is_step_like_call(node):
            description = extract_step_description(node)
            operations = find_operations_in_step(node)

            # Add step information to each operation
            for operation in operations:
                operation["step"] = description

            step = {
                "description": description,
                "actions": operations,
                "line": getattr(node, "lineno", 0),
                "type": "function_call",
            }
            steps.append(step)

        # Check for step-like function definitions
        elif isinstance(node, ast.FunctionDef) and is_step_like_function(node):
            description = extract_function_name_as_description(node)
            operations = find_operations_in_function(node)

            # Add step information to each operation
            for operation in operations:
                operation["step"] = description

            step = {
                "description": description,
                "actions": operations,
                "line": getattr(node, "lineno", 0),
                "type": "function_definition",
            }
            steps.append(step)

        # Check for step-like comments
        elif isinstance(node, ast.Expr) and is_step_comment(node):
            description = extract_comment_as_description(node)
            # Find operations in the next few statements
            operations = find_operations_after_comment(node, tree)

            # Add step information to each operation
            for operation in operations:
                operation["step"] = description

            step = {
                "description": description,
                "actions": operations,
                "line": getattr(node, "lineno", 0),
                "type": "comment",
            }
            steps.append(step)

    return steps


def is_step_like_function(func_def: ast.FunctionDef) -> bool:
    """Check if a function definition represents a step-like pattern"""
    func_name = func_def.name.lower()
    step_patterns = [
        "step",
        "given",
        "when",
        "then",
        "and",
        "but",
        "setup",
        "teardown",
        "before",
        "after",
        "arrange",
        "act",
        "assert",
        "prepare",
        "execute",
        "verify",
        "cleanup",
        "precondition",
        "action",
        "postcondition",
        "initialize",
        "configure",
        "deploy",
        "test",
        "validate",
        "cleanup",
    ]
    return any(pattern in func_name for pattern in step_patterns)


def find_operations_in_step(step_call: ast.Call) -> list:
    """Find all operations within a step-like call"""
    operations = []

    # For step calls, we need to look at the function passed as argument
    if len(step_call.args) >= 2:
        func_arg = step_call.args[1]
        if isinstance(func_arg, ast.Lambda):
            operations.extend(analyze_node_for_operations(func_arg.body))
        elif isinstance(func_arg, ast.Call):
            operations.extend(analyze_node_for_operations(func_arg))

    return operations


def find_operations_in_function(func_def: ast.FunctionDef) -> list:
    """Find all operations within a step-like function"""
    operations = []

    # Analyze the function body for operations
    for child in ast.walk(func_def):
        if isinstance(child, ast.Call):
            # Check for direct helper function calls
            if isinstance(child.func, ast.Name) and child.func.id in [
                "get_resource",
                "get_resource_from_namespace",
                "get_namespaces",
                "create_api_object",
                "delete_object",
            ]:
                if child.func.id == "get_namespaces":
                    operations.append({"gvk": "v1/Namespace", "verb": "list"})
                elif child.func.id == "create_api_object":
                    operations.append({"gvk": "unknown/unknown", "verb": "create"})
                elif child.func.id == "delete_object":
                    operations.append({"gvk": "unknown/unknown", "verb": "delete"})
                elif child.func.id in ["get_resource", "get_resource_from_namespace"]:
                    if child.args and isinstance(child.args[0], ast.Constant):
                        resource_name = child.args[0].value
                        gvk = map_resource_name_to_gvk(resource_name)
                        if gvk:
                            verb = "list" if child.func.id == "get_resource" else "get"
                            operations.append({"gvk": gvk, "verb": verb})
                    else:
                        verb = "list" if child.func.id == "get_resource" else "get"
                        operations.append({"gvk": "unknown/unknown", "verb": verb})

    return operations


def find_operations_after_comment(comment_node: ast.Expr, tree: ast.AST) -> list:
    """Find operations in statements following a step comment"""
    operations = []

    # This is a simplified implementation - in practice, you'd need to track
    # the AST structure to find the next few statements after the comment
    # For now, we'll search the entire tree for operations
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for direct helper function calls
            if isinstance(node.func, ast.Name) and node.func.id in [
                "get_resource",
                "get_resource_from_namespace",
                "get_namespaces",
                "create_api_object",
                "delete_object",
            ]:
                if node.func.id == "get_namespaces":
                    operations.append({"gvk": "v1/Namespace", "verb": "list"})
                elif node.func.id == "create_api_object":
                    operations.append({"gvk": "unknown/unknown", "verb": "create"})
                elif node.func.id == "delete_object":
                    operations.append({"gvk": "unknown/unknown", "verb": "delete"})
                elif node.func.id in ["get_resource", "get_resource_from_namespace"]:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        resource_name = node.args[0].value
                        gvk = map_resource_name_to_gvk(resource_name)
                        if gvk:
                            verb = "list" if node.func.id == "get_resource" else "get"
                            operations.append({"gvk": gvk, "verb": verb})
                    else:
                        verb = "list" if node.func.id == "get_resource" else "get"
                        operations.append({"gvk": "unknown/unknown", "verb": verb})

    return operations


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
