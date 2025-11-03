# match/build_index_and_match.py
import argparse, json, re
from typing import List, Dict, Any, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Operation categories for better similarity matching
OPERATION_CATEGORIES = {
    "read": ["get", "list", "watch"],
    "write": ["create", "update", "patch", "apply"],
    "delete": ["delete"],
}

# Verb groups for similarity detection
VERB_GROUPS = {
    "read_operations": ["get", "list", "watch"],
    "write_operations": ["create", "update", "patch", "apply"],
    "delete_operations": ["delete"],
}

# Purpose compatibility matrix - defines which purposes can match
# Made more restrictive to reduce false positives
PURPOSE_COMPATIBILITY = {
    "POD_MANAGEMENT": ["POD_HEALTH"],  # Only direct pod-related purposes
    "POD_HEALTH": ["POD_MANAGEMENT"],  # Only direct pod-related purposes
    "NETWORK_POLICY": ["NETWORK_CONNECTIVITY"],  # Only network-related purposes
    "NETWORK_CONNECTIVITY": ["NETWORK_POLICY"],  # Only network-related purposes
    "OPERATOR_MANAGEMENT": [
        "RESOURCE_VALIDATION",
        "UPGRADE_TESTING",
    ],  # Only with resource validation and upgrade
    "STORAGE_TESTING": ["RESOURCE_VALIDATION"],  # Only with resource validation
    "SECURITY_TESTING": ["RESOURCE_VALIDATION"],  # Only with resource validation
    "CONFIGURATION": [
        "RESOURCE_VALIDATION",
        "UPGRADE_TESTING",
    ],  # Only with resource validation and upgrade
    "PERFORMANCE": ["RESOURCE_VALIDATION"],  # Only with resource validation
    "WEBHOOK_TESTING": [],  # Only matches with itself - highly specialized
    "RESOURCE_VALIDATION": [
        "OPERATOR_MANAGEMENT",
        "STORAGE_TESTING",
        "SECURITY_TESTING",
        "CONFIGURATION",
        "PERFORMANCE",
    ],  # Removed network and pod purposes
    # Specialized testing purposes - only match with themselves or very specific others
    "UPGRADE_TESTING": [
        "UPGRADE_TESTING",
        "OPERATOR_MANAGEMENT",
        "CONFIGURATION",
    ],  # Upgrade can match with operator management and configuration
    "SRIOV_TESTING": [
        "SRIOV_TESTING",
        "NETWORK_POLICY",
    ],  # SR-IOV only matches SR-IOV or network policy
    "PTP_TESTING": [
        "PTP_TESTING",
        "NETWORK_CONNECTIVITY",
    ],  # PTP only matches PTP or network connectivity
    "DUAL_STACK_TESTING": [
        "DUAL_STACK_TESTING",
        "NETWORK_CONNECTIVITY",
    ],  # Dual stack only matches dual stack or network
    "IPV4_ONLY_TESTING": [
        "IPV4_ONLY_TESTING",
        "DUAL_STACK_TESTING",
        "NETWORK_CONNECTIVITY",
    ],
    "IPV6_ONLY_TESTING": [
        "IPV6_ONLY_TESTING",
        "DUAL_STACK_TESTING",
        "NETWORK_CONNECTIVITY",
    ],
    "UNKNOWN": [
        "POD_MANAGEMENT",
        "POD_HEALTH",
        "NETWORK_POLICY",
        "NETWORK_CONNECTIVITY",
        "OPERATOR_MANAGEMENT",
        "STORAGE_TESTING",
        "SECURITY_TESTING",
        "CONFIGURATION",
        "PERFORMANCE",
        "RESOURCE_VALIDATION",
    ],
}

# Resource compatibility matrix - defines which Kubernetes resources are compatible for matching
RESOURCE_COMPATIBILITY = {
    # Pod-related resources
    "v1/Pod": [
        "v1/Pod",
        "apps/v1/Deployment",
        "apps/v1/StatefulSet",
        "apps/v1/DaemonSet",
    ],
    "apps/v1/Deployment": ["v1/Pod", "apps/v1/Deployment", "apps/v1/StatefulSet"],
    "apps/v1/StatefulSet": ["v1/Pod", "apps/v1/Deployment", "apps/v1/StatefulSet"],
    "apps/v1/DaemonSet": ["v1/Pod", "apps/v1/DaemonSet"],
    # Namespace-related resources
    "v1/Namespace": [
        "v1/Namespace",
        "v1/Pod",
        "apps/v1/Deployment",
        "apps/v1/StatefulSet",
    ],
    # Cluster management resources (incompatible with each other)
    "config.openshift.io/v1/ClusterVersion": ["config.openshift.io/v1/ClusterVersion"],
    "hive.openshift.io/v1/ClusterDeployment": ["hive.openshift.io/v1/ClusterDeployment"],
    "metal3.io/v1alpha1/BareMetalHost": ["metal3.io/v1alpha1/BareMetalHost"],
    # Network-related resources
    "v1/Service": ["v1/Service", "route.openshift.io/v1/Route"],
    "route.openshift.io/v1/Route": ["v1/Service", "route.openshift.io/v1/Route"],
    # Storage-related resources
    "v1/PersistentVolume": ["v1/PersistentVolume", "v1/PersistentVolumeClaim"],
    "v1/PersistentVolumeClaim": ["v1/PersistentVolume", "v1/PersistentVolumeClaim"],
    # Operator-related resources
    "operators.coreos.com/v1alpha1/ClusterServiceVersion": [
        "operators.coreos.com/v1alpha1/ClusterServiceVersion"
    ],
    "operators.coreos.com/v1alpha1/Subscription": ["operators.coreos.com/v1alpha1/Subscription"],
}

# Technology compatibility matrix - expanded to include all technology domains
TECH_COMPATIBILITY = {
    # Networking technologies
    "SR-IOV": [
        "SR-IOV",
        "CNI",
        "Virtualization",
    ],  # SR-IOV can match with CNI and virtualization
    "PTP": [
        "PTP",
        "CNI",
        "Edge Computing",
    ],  # PTP can match with CNI and edge computing
    "DPDK": [
        "DPDK",
        "CNI",
        "Virtualization",
    ],  # DPDK can match with CNI and virtualization
    "MetalLB": ["MetalLB", "CNI"],  # MetalLB can match with CNI (both are networking)
    "RDMA": ["RDMA", "CNI", "Storage"],  # RDMA can match with CNI and storage
    "Bonding": ["Bonding", "CNI"],  # Bonding can match with CNI (both are networking)
    "CNI": [
        "SR-IOV",
        "PTP",
        "DPDK",
        "MetalLB",
        "RDMA",
        "Bonding",
        "CNI",
        "Virtualization",
    ],  # CNI is compatible with networking and virtualization
    # Hardware acceleration
    "GPU": [
        "GPU",
        "Machine Learning",
        "Virtualization",
    ],  # GPU can match with ML and virtualization
    # Virtualization
    "Virtualization": [
        "SR-IOV",
        "DPDK",
        "CNI",
        "GPU",
        "Storage",
        "Security",
    ],  # Virtualization is compatible with many technologies
    # Storage
    "Storage": [
        "RDMA",
        "Virtualization",
        "Security",
        "Monitoring",
    ],  # Storage can match with RDMA, virtualization, security, monitoring
    # Security
    "Security": [
        "Virtualization",
        "Storage",
        "Monitoring",
        "Edge Computing",
    ],  # Security is compatible with many domains
    # Monitoring/Observability
    "Monitoring": [
        "Storage",
        "Security",
        "Edge Computing",
        "Machine Learning",
    ],  # Monitoring is compatible with many domains
    # Machine Learning/AI
    "Machine Learning": [
        "GPU",
        "Edge Computing",
        "Monitoring",
    ],  # ML can match with GPU, edge computing, monitoring
    # Edge Computing
    "Edge Computing": [
        "PTP",
        "Security",
        "Monitoring",
        "Machine Learning",
    ],  # Edge computing is compatible with time-sensitive and monitoring tech
    # Power Management
    "Power Management": [
        "Power Management",
        "Edge Computing",
    ],  # Power management can match with edge computing
}


def is_purpose_compatible(purpose_a: str, purpose_b: str) -> bool:
    """Check if two test purposes are compatible for matching."""
    if not purpose_a or not purpose_b:
        return True  # Allow matches if purpose is unknown

    # Same purpose is always compatible
    if purpose_a == purpose_b:
        return True

    # Check compatibility matrix
    compatible_purposes = PURPOSE_COMPATIBILITY.get(purpose_a, [])
    return purpose_b in compatible_purposes


def is_resource_compatible(resources_a: List[str], resources_b: List[str]) -> bool:
    """Check if two sets of Kubernetes resources are compatible for matching."""
    if not resources_a or not resources_b:
        return True  # Allow matches if no resources specified

    # Check if any resource from test A is compatible with any resource from test B
    for res1 in resources_a:
        for res2 in resources_b:
            compatible_resources = RESOURCE_COMPATIBILITY.get(res1, [])
            if res2 in compatible_resources:
                return True

    return False


def is_tech_compatible(tech_a: List[str], tech_b: List[str]) -> bool:
    """Check if two sets of technologies are compatible for matching."""
    if not tech_a or not tech_b:
        return True  # Allow matches if no tech specified

    # If either test has no tech, allow the match
    if not tech_a or not tech_b:
        return True

    # Check if any technology from test A is compatible with any technology from test B
    for tech1 in tech_a:
        for tech2 in tech_b:
            compatible_techs = TECH_COMPATIBILITY.get(tech1, [])
            if tech2 in compatible_techs:
                return True

    return False


def calculate_functional_similarity(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> float:
    """Calculate functional similarity score based on test functionality overlap.
    Now includes setup and teardown operations from all phases."""
    actions_a = spec_a.get("actions") or []
    actions_b = spec_b.get("actions") or []
    expectations_a = spec_a.get("expectations") or []
    expectations_b = spec_b.get("expectations") or []

    if not actions_a or not actions_b:
        return 0.0

    # Extract operations and expectations, including phase information
    # Include operations from all phases: setup, test, teardown
    operations_a = set()
    operations_b = set()
    setup_operations_a = set()
    setup_operations_b = set()
    test_operations_a = set()
    test_operations_b = set()
    teardown_operations_a = set()
    teardown_operations_b = set()
    expectations_set_a = set()
    expectations_set_b = set()

    for action in actions_a:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "").lower()  # Normalize to lowercase
        phase = action.get("phase", "test")  # Default to "test" if not set
        if gvk and verb:
            operation_key = f"{gvk}:{verb}"
            operations_a.add(operation_key)
            # Also track by phase for weighted similarity
            if phase == "setup":
                setup_operations_a.add(operation_key)
            elif phase == "test":
                test_operations_a.add(operation_key)
            elif phase == "teardown":
                teardown_operations_a.add(operation_key)
            else:
                # Default to test operations
                test_operations_a.add(operation_key)

    for action in actions_b:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "").lower()  # Normalize to lowercase
        phase = action.get("phase", "test")  # Default to "test" if not set
        if gvk and verb:
            operation_key = f"{gvk}:{verb}"
            operations_b.add(operation_key)
            # Also track by phase for weighted similarity
            if phase == "setup":
                setup_operations_b.add(operation_key)
            elif phase == "test":
                test_operations_b.add(operation_key)
            elif phase == "teardown":
                teardown_operations_b.add(operation_key)
            else:
                # Default to test operations
                test_operations_b.add(operation_key)

    for exp in expectations_a:
        if isinstance(exp, dict):
            target = exp.get("target", "")
            condition = exp.get("condition", "")
            if target and condition:
                expectations_set_a.add(f"{target}:{condition}")

    for exp in expectations_b:
        if isinstance(exp, dict):
            target = exp.get("target", "")
            condition = exp.get("condition", "")
            if target and condition:
                expectations_set_b.add(f"{target}:{condition}")

    # Calculate overlap for all operations (setup, test, teardown)
    common_operations = operations_a & operations_b
    common_setup = setup_operations_a & setup_operations_b
    common_test = test_operations_a & test_operations_b
    common_teardown = teardown_operations_a & teardown_operations_b
    common_expectations = expectations_set_a & expectations_set_b

    # Weighted similarity: operations count more than expectations
    # Give more weight to test operations, but include setup/teardown
    if not operations_a or not operations_b:
        return 0.0

    # Calculate overlaps for each phase
    setup_overlap = (
        len(common_setup) / max(len(setup_operations_a), len(setup_operations_b))
        if setup_operations_a or setup_operations_b
        else 0.0
    )
    test_overlap = (
        len(common_test) / max(len(test_operations_a), len(test_operations_b))
        if test_operations_a or test_operations_b
        else 0.0
    )
    teardown_overlap = (
        len(common_teardown) / max(len(teardown_operations_a), len(teardown_operations_b))
        if teardown_operations_a or teardown_operations_b
        else 0.0
    )
    overall_operation_overlap = len(common_operations) / max(len(operations_a), len(operations_b))
    expectation_overlap = (
        len(common_expectations) / max(len(expectations_set_a), len(expectations_set_b))
        if expectations_set_a or expectations_set_b
        else 0.0
    )

    # Weighted combination:
    # - 50% overall operations (includes all phases)
    # - 20% test operations (most important)
    # - 15% setup operations (important for compatibility)
    # - 5% teardown operations (less important)
    # - 10% expectations
    # Ensure we don't double-count, so we use a combination approach
    similarity = (
        0.5 * overall_operation_overlap
        + 0.2 * test_overlap
        + 0.15 * setup_overlap
        + 0.05 * teardown_overlap
        + 0.1 * expectation_overlap
    )

    return similarity


def calculate_framework_bias_penalty(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> float:
    """Calculate penalty for framework-heavy matches to reduce false positives."""
    test_id_a = spec_a.get("test_id", "").lower()
    test_id_b = spec_b.get("test_id", "").lower()
    actions_a = spec_a.get("actions") or []
    actions_b = spec_b.get("actions") or []
    purpose_a = spec_a.get("purpose", "")
    purpose_b = spec_b.get("purpose", "")

    penalty = 0.0

    # Framework infrastructure operations that shouldn't drive similarity
    framework_ops = {
        "v1/Namespace:create",
        "v1/Namespace:delete",  # Basic test setup
        "v1/ConfigMap:create",
        "v1/ConfigMap:delete",  # Basic configuration
        "v1/Pod:create",
        "v1/Pod:delete",  # Basic pod operations
    }

    ops_a = set()
    ops_b = set()
    ops_with_gvk_a = 0
    ops_with_gvk_b = 0

    for action in actions_a:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            ops_a.add(f"{gvk}:{verb}")
            ops_with_gvk_a += 1

    for action in actions_b:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            ops_b.add(f"{gvk}:{verb}")
            ops_with_gvk_b += 1

    # CRITICAL: Heavy penalty if most operations don't have GVKs (too generic)
    if len(actions_a) > 0:
        gvk_ratio_a = ops_with_gvk_a / len(actions_a)
        if gvk_ratio_a < 0.4:  # Less than 40% of operations have GVKs
            penalty += 0.3

    if len(actions_b) > 0:
        gvk_ratio_b = ops_with_gvk_b / len(actions_b)
        if gvk_ratio_b < 0.4:  # Less than 40% of operations have GVKs
            penalty += 0.3

    # Calculate what percentage of operations are framework-heavy
    shared_ops = ops_a & ops_b
    framework_shared = shared_ops & framework_ops

    if shared_ops:
        framework_ratio = len(framework_shared) / len(shared_ops)
        # If more than 60% of shared operations are framework operations, apply penalty
        if framework_ratio > 0.6:
            penalty += 0.3 * framework_ratio

    # Additional penalty for RESOURCE_VALIDATION tests that only share generic expectations
    if purpose_a == "RESOURCE_VALIDATION" and purpose_b == "RESOURCE_VALIDATION":
        expectations_a = spec_a.get("expectations", [])
        expectations_b = spec_b.get("expectations", [])

        # Count specific vs generic expectations
        generic_expectations = {"test_condition"}
        specific_expectations_a = {
            exp.get("target", "") if isinstance(exp, dict) else ""
            for exp in expectations_a
            if isinstance(exp, dict) and exp.get("target", "") not in generic_expectations
        }
        specific_expectations_b = {
            exp.get("target", "") if isinstance(exp, dict) else ""
            for exp in expectations_b
            if isinstance(exp, dict) and exp.get("target", "") not in generic_expectations
        }

        # If both have mostly generic expectations and they don't share specific ones
        if not specific_expectations_a and not specific_expectations_b:
            # Both tests only have generic expectations - check if they validate different things
            # Check test names for different validation targets
            test_name_a = test_id_a.split(":")[-1] if ":" in test_id_a else test_id_a
            test_name_b = test_id_b.split(":")[-1] if ":" in test_id_b else test_id_b

            # If test names suggest different validation targets, apply penalty
            # This is a heuristic - webhook validation vs event generation are different
            if ("webhook" in test_name_a or "operator" in test_name_a) and (
                "event" in test_name_b or "generate" in test_name_b
            ):
                penalty += 0.5
            elif ("event" in test_name_a or "generate" in test_name_a) and (
                "webhook" in test_name_b or "operator" in test_name_b
            ):
                penalty += 0.5

    # Additional penalty for webhook tests with different types
    if "webhook" in test_id_a and "webhook" in test_id_b:
        # If both are webhook tests but one is much more complex than the other
        complexity_diff = abs(len(actions_a) - len(actions_b))
        if complexity_diff > 2:  # Significant complexity difference
            penalty += 0.4

    # Additional penalty for tests that only share basic operations
    if shared_ops and shared_ops.issubset(framework_ops):
        penalty += 0.5  # Heavy penalty for only sharing framework operations

    return min(penalty, 0.8)  # Cap penalty at 80%


def calculate_same_file_penalty(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> float:
    """Calculate penalty for tests from the same file that have different test names.

    Tests from the same file that have very different names (e.g., 'generate_events' vs 'deploy_image')
    likely test different functionality and shouldn't be matched unless they're very similar.
    """
    test_id_a = spec_a.get("test_id", "")
    test_id_b = spec_b.get("test_id", "")

    # Extract file paths
    file_a = get_file_path_from_test_id(test_id_a)
    file_b = get_file_path_from_test_id(test_id_b)

    penalty = 0.0

    # Only apply penalty if tests are from the same file
    if file_a == file_b and file_a:
        # Extract test names (everything after the last colon)
        test_name_a = test_id_a.split(":")[-1] if ":" in test_id_a else test_id_a
        test_name_b = test_id_b.split(":")[-1] if ":" in test_id_b else test_id_b

        # Normalize to lowercase for comparison
        name_a_lower = test_name_a.lower()
        name_b_lower = test_name_b.lower()

        # If test names are identical, no penalty (they might be the same test)
        if name_a_lower == name_b_lower:
            return 0.0

        # Different keywords that suggest different functionality
        functional_keywords_a = set()
        functional_keywords_b = set()

        # Extract meaningful keywords from test names
        keywords_patterns = [
            "event",
            "events",
            "generate",
            "generation",
            "deploy",
            "deployment",
            "create",
            "delete",
            "remove",
            "load",
            "loaded",
            "module",
            "modules",
            "image",
            "images",
            "build",
            "prebuild",
            "pre-build",
            "webhook",
            "webhooks",
            "operator",
            "operators",
            "validate",
            "validation",
            "verify",
            "check",
            "install",
            "uninstall",
            "update",
            "upgrade",
            "start",
            "stop",
            "run",
            "execute",
        ]

        for keyword in keywords_patterns:
            if keyword in name_a_lower:
                functional_keywords_a.add(keyword)
            if keyword in name_b_lower:
                functional_keywords_b.add(keyword)

        # If test names have no shared functional keywords, they're likely testing different things
        shared_keywords = functional_keywords_a & functional_keywords_b

        if not shared_keywords and (functional_keywords_a or functional_keywords_b):
            # No shared keywords suggests very different functionality
            penalty += 0.4

            # Additional penalty if they're testing completely different things
            # e.g., "generate_events" vs "deploy_image"
            event_related = any(
                kw in functional_keywords_a for kw in ["event", "events", "generate", "generation"]
            )
            deployment_related = any(
                kw in functional_keywords_a
                for kw in ["deploy", "deployment", "image", "build", "prebuild"]
            )
            event_related_b = any(
                kw in functional_keywords_b for kw in ["event", "events", "generate", "generation"]
            )
            deployment_related_b = any(
                kw in functional_keywords_b
                for kw in ["deploy", "deployment", "image", "build", "prebuild"]
            )

            if (event_related and deployment_related_b) or (deployment_related and event_related_b):
                # One is about events, the other about deployment - very different
                penalty += 0.3

        elif shared_keywords:
            # Some shared keywords - smaller penalty based on how different they are
            unique_a = functional_keywords_a - functional_keywords_b
            unique_b = functional_keywords_b - functional_keywords_a

            if len(unique_a) > 2 or len(unique_b) > 2:
                # Many unique keywords suggests different functionality
                penalty += 0.2

    return min(penalty, 0.7)  # Cap penalty at 70%


def calculate_cross_language_penalty(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> float:
    """Calculate penalty for cross-language matches to reduce false positives."""
    test_id_a = spec_a.get("test_id", "")
    test_id_b = spec_b.get("test_id", "")

    # Extract language from test_id
    lang_a = get_language_from_test_id(test_id_a)
    lang_b = get_language_from_test_id(test_id_b)

    # No penalty if same language
    if lang_a == lang_b:
        return 0.0

    # Cross-language match detected
    purpose_a = spec_a.get("purpose", "")
    purpose_b = spec_b.get("purpose", "")

    # Base penalty for cross-language matches
    penalty = 0.1  # Reduced base penalty

    # Increase penalty if purposes are incompatible for cross-language matching
    if not is_cross_language_purpose_compatible(purpose_a, purpose_b):
        penalty += 0.2  # Reduced incompatibility penalty

    # Additional penalty for very different test types
    if is_different_test_domain(spec_a, spec_b):
        penalty += 0.2  # Reduced domain penalty

    # Reduce penalty if tests have strong operational similarity (avoid recursion)
    # Calculate basic functional similarity without cross-language penalty
    actions_a = spec_a.get("actions") or []
    actions_b = spec_b.get("actions") or []

    operations_a = set()
    operations_b = set()

    for action in actions_a:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            operations_a.add(f"{gvk}:{verb}")

    for action in actions_b:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            operations_b.add(f"{gvk}:{verb}")

    # Calculate basic operational overlap
    shared_ops = operations_a & operations_b
    total_ops = len(operations_a | operations_b)

    if total_ops > 0:
        operational_similarity = len(shared_ops) / total_ops
        if operational_similarity > 0.6:  # Strong operational similarity
            penalty *= 0.5  # Reduce penalty for truly similar tests

    return min(penalty, 0.5)  # Cap penalty at 50%


def get_language_from_test_id(test_id: str) -> str:
    """Extract programming language from test ID."""
    if ".go:" in test_id or "/tests/" in test_id:
        return "go"
    elif ".py:" in test_id or "test_" in test_id:
        return "python"
    return "unknown"


def is_cross_language_purpose_compatible(purpose_a: str, purpose_b: str) -> bool:
    """Check if two purposes are compatible for cross-language matching."""
    # Purposes that are more suitable for cross-language comparison
    cross_language_friendly = {
        "RESOURCE_VALIDATION",
        "POD_HEALTH",
        "NETWORK_CONNECTIVITY",
        "STORAGE_TESTING",
        "CONFIGURATION",
        "OPERATOR_MANAGEMENT",
    }

    # Purposes that are more language/framework specific
    language_specific = {"WEBHOOK_TESTING", "POD_MANAGEMENT", "PTP_TESTING", "SRIOV_TESTING"}

    # Same purpose is compatible
    if purpose_a == purpose_b:
        return purpose_a in cross_language_friendly

    # Different purposes from language-specific categories are incompatible
    if purpose_a in language_specific or purpose_b in language_specific:
        return False

    # Check general purpose compatibility
    return is_purpose_compatible(purpose_a, purpose_b)


def is_different_test_domain(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> bool:
    """Check if two tests belong to significantly different domains."""
    test_id_a = spec_a.get("test_id", "").lower()
    test_id_b = spec_b.get("test_id", "").lower()

    # Define test domains based on path patterns
    domains_a = set()
    domains_b = set()

    domain_patterns = {
        "network": ["network", "service", "endpoint", "ingress", "route"],
        "autoscaling": ["autoscaling", "cluster_size", "hpa", "vpa", "scaling"],
        "deployment": ["deployment", "deploy", "ztp", "cluster_deployment"],
        "storage": ["storage", "pv", "pvc", "volume", "ceph"],
        "security": ["security", "webhook", "admission", "rbac", "scc"],
        "monitoring": ["monitor", "metrics", "prometheus", "alert"],
    }

    # Classify both tests
    for domain, patterns in domain_patterns.items():
        if any(pattern in test_id_a for pattern in patterns):
            domains_a.add(domain)
        if any(pattern in test_id_b for pattern in patterns):
            domains_b.add(domain)

    # If both tests have no clear domain, they're not from different domains
    if not domains_a or not domains_b:
        return False

    # Check if domains overlap
    return not bool(domains_a & domains_b)


def calculate_context_penalty(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> float:
    """Calculate penalty based on test file context differences."""
    test_id_a = spec_a.get("test_id", "")
    test_id_b = spec_b.get("test_id", "")

    # Extract file paths and context
    file_path_a = get_file_path_from_test_id(test_id_a)
    file_path_b = get_file_path_from_test_id(test_id_b)

    penalty = 0.0

    # Context similarity based on file path structure
    path_parts_a = file_path_a.split("/")
    path_parts_b = file_path_b.split("/")

    # Check if tests are from completely different modules
    if len(path_parts_a) >= 2 and len(path_parts_b) >= 2:
        module_a = path_parts_a[-2] if len(path_parts_a) > 1 else ""
        module_b = path_parts_b[-2] if len(path_parts_b) > 1 else ""

        # Different test modules suggest different contexts
        if module_a != module_b:
            # Check for related modules
            related_modules = {
                ("deployment", "undeploy"): 0.1,  # Related: deploy vs undeploy
                ("cu", "du"): 0.15,  # Related: CU vs DU
                ("network", "service"): 0.1,  # Related: network tests
                ("storage", "pv"): 0.1,  # Related: storage tests
                ("policies", "operator"): 0.1,  # Related: policy tests
            }

            # Check if modules are related
            is_related = False
            for (mod1, mod2), related_penalty in related_modules.items():
                if (mod1 in module_a.lower() and mod2 in module_b.lower()) or (
                    mod2 in module_a.lower() and mod1 in module_b.lower()
                ):
                    penalty += related_penalty
                    is_related = True
                    break

            if not is_related:
                penalty += 0.25  # Penalty for unrelated modules

    # Check for environment context differences
    env_a = spec_a.get("environment", [])
    env_b = spec_b.get("environment", [])

    if env_a and env_b:
        env_overlap = set(env_a) & set(env_b)
        if not env_overlap:
            penalty += 0.2  # Different environments
        elif len(env_overlap) < min(len(env_a), len(env_b)):
            penalty += 0.1  # Partial environment overlap

    # Check for test type context differences
    test_type_a = spec_a.get("test_type", "")
    test_type_b = spec_b.get("test_type", "")

    if test_type_a != test_type_b:
        penalty += 0.05  # Different test types

    # Check dependencies context
    deps_a = set(spec_a.get("dependencies", []))
    deps_b = set(spec_b.get("dependencies", []))

    if deps_a and deps_b:
        deps_overlap = deps_a & deps_b
        if not deps_overlap:
            penalty += 0.15  # No shared dependencies

    return min(penalty, 0.6)  # Cap context penalty at 60%


def get_file_path_from_test_id(test_id: str) -> str:
    """Extract file path from test ID."""
    if ":" in test_id:
        return test_id.split(":")[0]
    return test_id


def has_meaningful_operations(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> bool:
    """Check if two specs have meaningful operational overlap beyond just resource-level similarity."""
    actions_a = spec_a.get("actions") or []
    actions_b = spec_b.get("actions") or []

    # CRITICAL: Both tests must have at least one operation to be considered meaningful
    if len(actions_a) == 0 or len(actions_b) == 0:
        return False

    functional_score = calculate_functional_similarity(spec_a, spec_b)
    return functional_score > 0.3  # Threshold for meaningful operations


def is_framework_pattern_match(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> bool:
    """Check if two specs match mainly due to testing framework patterns rather than functional similarity."""
    test_id_a = spec_a.get("test_id", "")
    test_id_b = spec_b.get("test_id", "")

    # Framework pattern indicators
    framework_patterns = [
        "webhook",  # webhook testing infrastructure
        "ginkgo_tests",  # ginkgo suite setup
        "Test",  # basic unit test wrappers
        "Suite",  # test suite setup
        "e2e",  # e2e framework tests
    ]

    # Check if both test IDs contain framework patterns
    patterns_a = [p for p in framework_patterns if p.lower() in test_id_a.lower()]
    patterns_b = [p for p in framework_patterns if p.lower() in test_id_b.lower()]

    # If both have the same framework pattern, they might be framework matches
    if patterns_a and patterns_b and any(p in patterns_b for p in patterns_a):
        # Additional checks for webhook-specific patterns
        if "webhook" in patterns_a and "webhook" in patterns_b:
            return is_webhook_framework_match(spec_a, spec_b)

    return False


def is_webhook_framework_match(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> bool:
    """Check if two webhook tests are matching mainly due to shared framework rather than functionality."""
    test_id_a = spec_a.get("test_id", "")
    test_id_b = spec_b.get("test_id", "")

    # Common webhook test patterns that should be differentiated
    webhook_types = {
        "testWebhook": "comprehensive_validation",
        "testFailClosedWebhook": "fail_closed_testing",
        "testAttachingPodWebhook": "pod_attachment",
        "testMutatingWebhook": "mutation_testing",
        "testValidatingWebhook": "validation_testing",
    }

    # Extract webhook type from test names
    type_a = None
    type_b = None

    for pattern, webhook_type in webhook_types.items():
        if pattern in test_id_a:
            type_a = webhook_type
        if pattern in test_id_b:
            type_b = webhook_type

    # If both are webhook tests but different types, it's a framework match (false positive)
    if type_a and type_b and type_a != type_b:
        return True

    # Check operations overlap - if they share basic webhook operations but have different complexity
    actions_a = spec_a.get("actions") or []
    actions_b = spec_b.get("actions") or []

    # If one test is much more complex than the other, likely framework match
    if abs(len(actions_a) - len(actions_b)) > 3:  # Significant difference in complexity
        return True

    return False


def filter_by_purpose_compatibility(
    matches: List[Dict[str, Any]],
    specs_a: List[Dict[str, Any]],
    specs_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter matches based on purpose compatibility, resource compatibility, networking tech compatibility, and meaningful operations."""
    filtered = []
    for match in matches:
        idx_a = match["idx_a"]
        idx_b = match["idx_b"]

        purpose_a = specs_a[idx_a].get("purpose", "")
        purpose_b = specs_b[idx_b].get("purpose", "")
        tech_a = specs_a[idx_a].get("tech", [])
        tech_b = specs_b[idx_b].get("tech", [])
        resources_a = get_resources_from_spec(specs_a[idx_a])
        resources_b = get_resources_from_spec(specs_b[idx_b])

        # CRITICAL: Both specs must have at least one action to be considered for matching
        actions_a = specs_a[idx_a].get("actions") or []
        actions_b = specs_b[idx_b].get("actions") or []
        if len(actions_a) == 0 or len(actions_b) == 0:
            continue

        # CRITICAL: Exclude suite setup functions from matching (these are infrastructure, not real tests)
        test_id_a = specs_a[idx_a].get("test_id", "")
        test_id_b = specs_b[idx_b].get("test_id", "")
        if ":ginkgo_tests" in test_id_a or ":ginkgo_tests" in test_id_b:
            continue
        if ":Test" in test_id_a or ":Test" in test_id_b:
            continue

        # Check purpose compatibility
        if not is_purpose_compatible(purpose_a, purpose_b):
            continue

        # Check resource compatibility
        if not is_resource_compatible(resources_a, resources_b):
            continue

        # Check technology compatibility
        if not is_tech_compatible(tech_a, tech_b):
            continue

        # For high-similarity matches, also check for meaningful operations
        if match.get("base_score", 0) > 0.7:  # High similarity threshold
            if not has_meaningful_operations(specs_a[idx_a], specs_b[idx_b]):
                continue

        # Check for framework pattern matches (false positives)
        if is_framework_pattern_match(specs_a[idx_a], specs_b[idx_b]):
            continue

        filtered.append(match)

    return filtered


def load_specs(path: str) -> List[Dict[str, Any]]:
    specs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                specs.append(json.loads(line))
            except Exception:
                pass
    return specs


def is_utility_test(spec: Dict[str, Any]) -> bool:
    """Check if a test spec is a utility/helper test (not a real test case)."""
    test_id = spec.get("test_id", "")
    actions = spec.get("actions") or []
    expectations = spec.get("expectations") or []

    # Check for suite setup/test infrastructure functions (Ginkgo test suite setup)
    if ":ginkgo_tests" in test_id:
        return True

    # Check for unit test wrappers (not integration tests)
    if ":Test" in test_id and ":ginkgo_tests" not in test_id:
        return True

    # Check for utility test patterns in test_id
    utility_patterns = [
        "test_test",  # TestTestsToScript, test_test_*
        "helper",  # helper functions
        "util",  # utility functions
        "ref.go",  # reference/utility files
        "common.go",  # common utility files
        "setup",  # setup functions
        "teardown",  # teardown functions
        "mock",  # mock functions
        "stub",  # stub functions
        "fixture",  # test fixtures
        "helper_test",  # helper test files
        "util_test",  # utility test files
    ]

    # Check if test_id contains utility patterns
    for pattern in utility_patterns:
        if pattern in test_id.lower():
            return True

    # Check for utility test content patterns
    # 1. No actions (no Kubernetes operations)
    # 2. Only generic test_condition expectations
    # 3. Very simple expectation conditions (single strings, basic values)
    if len(actions) == 0 and len(expectations) > 0:
        # Check if all expectations are generic test_condition with simple values
        all_generic = True
        for exp in expectations:
            target = exp.get("target", "")
            condition = exp.get("condition", "")

            # Not a utility test if it has specific targets
            if target not in ["test_condition"]:
                all_generic = False
                break

            # Not a utility test if condition is complex (contains variables, function calls, etc.)
            if any(
                char in condition
                for char in [
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "==",
                    "!=",
                    ">",
                    "<",
                    "len(",
                    "count",
                ]
            ):
                all_generic = False
                break

            # Not a utility test if condition is too long (likely complex logic)
            if len(condition) > 50:
                all_generic = False
                break

        if all_generic:
            return True

    return False


def is_empty_test(spec: Dict[str, Any]) -> bool:
    """Check if a test spec is empty (has no meaningful content)."""
    # Handle None values by converting to empty lists
    actions = spec.get("actions") or []
    expectations = spec.get("expectations") or []
    dependencies = spec.get("dependencies") or []
    openshift_specific = spec.get("openshift_specific") or []
    concurrency = spec.get("concurrency") or []
    artifacts = spec.get("artifacts") or []

    # A test is considered empty if it has:
    # 1. No actions AND no expectations
    # 2. No other meaningful content (preconditions, openshift_specific, concurrency, artifacts)
    has_actions = len(actions) > 0
    has_expectations = len(expectations) > 0
    has_other_content = (
        len(dependencies) > 0
        or len(openshift_specific) > 0
        or len(concurrency) > 0
        or len(artifacts) > 0
    )

    # Empty if no actions, no expectations, and no other content
    # CRITICAL: Tests with no actions are not useful for similarity matching
    return not (has_actions or has_expectations or has_other_content)


def filter_empty_tests(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out empty tests and utility tests from the specs list."""
    filtered = []
    empty_count = 0
    utility_count = 0

    for spec in specs:
        if is_empty_test(spec):
            empty_count += 1
        elif is_utility_test(spec):
            utility_count += 1
        else:
            filtered.append(spec)

    print(
        f"Filtered out {empty_count} empty tests and {utility_count} utility tests, kept {len(filtered)} meaningful tests"
    )
    return filtered


def spec_to_text(spec: Dict[str, Any]) -> str:
    """Generate structured text representation that preserves JSON semantics"""
    # Don't include test_id in the content - it's used as document ID
    parts = []

    # Test metadata section
    test_type = spec.get("test_type", "unknown")
    if test_type:
        parts.append(f"TEST_TYPE: {test_type}")

    # Technology section (once, not repeated)
    tech = spec.get("tech", [])
    if tech:
        parts.append(f"TECHNOLOGY: {', '.join(tech)}")

    # Purpose section
    purpose = spec.get("purpose", "")
    if purpose:
        parts.append(f"PURPOSE: {purpose}")

    # Environment section
    environment = spec.get("environment", [])
    if environment:
        parts.append(f"ENVIRONMENT: {', '.join(environment)}")

    # Dependencies section
    dependencies = spec.get("dependencies") or []
    if dependencies:
        parts.append("DEPENDENCIES:")
        for dep in dependencies:
            parts.append(f"  - {dep}")

    # Operations section with structure, grouped by phase
    actions = spec.get("actions") or []
    if actions:
        # Group actions by phase for better representation
        setup_actions = [a for a in actions if a.get("phase") == "setup"]
        test_actions = [a for a in actions if a.get("phase") == "test" or not a.get("phase")]
        teardown_actions = [a for a in actions if a.get("phase") == "teardown"]

        if setup_actions:
            parts.append("SETUP_OPERATIONS:")
            for action in setup_actions:
                gvk = action.get("gvk", "")
                kind_hint = (action.get("fields") or {}).get("kind_hint", "")
                verb = action.get("verb", "").lower()  # Normalize to lowercase
                if gvk and kind_hint and "/" not in gvk:
                    gvk = f"{gvk}/{kind_hint}"
                if gvk and verb:
                    parts.append(f"  - {gvk}:{verb}")
                elif gvk:
                    parts.append(f"  - {gvk}")
                elif verb:
                    parts.append(f"  - verb:{verb}")

        if test_actions:
            parts.append("OPERATIONS:")
            for action in test_actions:
                gvk = action.get("gvk", "")
                kind_hint = (action.get("fields") or {}).get("kind_hint", "")
                verb = action.get("verb", "").lower()  # Normalize to lowercase
                if gvk and kind_hint and "/" not in gvk:
                    gvk = f"{gvk}/{kind_hint}"
                if gvk and verb:
                    parts.append(f"  - {gvk}:{verb}")
                elif gvk:
                    parts.append(f"  - {gvk}")
                elif verb:
                    parts.append(f"  - verb:{verb}")

        if teardown_actions:
            parts.append("TEARDOWN_OPERATIONS:")
            for action in teardown_actions:
                gvk = action.get("gvk", "")
                kind_hint = (action.get("fields") or {}).get("kind_hint", "")
                verb = action.get("verb", "").lower()  # Normalize to lowercase
                if gvk and kind_hint and "/" not in gvk:
                    gvk = f"{gvk}/{kind_hint}"
                if gvk and verb:
                    parts.append(f"  - {gvk}:{verb}")
                elif gvk:
                    parts.append(f"  - {gvk}")
                elif verb:
                    parts.append(f"  - verb:{verb}")

    # Expectations section
    expectations = spec.get("expectations") or []
    if expectations:
        parts.append("EXPECTATIONS:")
        for exp in expectations:
            target = exp.get("target", "")
            condition = exp.get("condition", "")
            if target and condition:
                parts.append(f"  - {target}={condition}")

    # External dependencies section
    externals = spec.get("externals") or []
    if externals:
        parts.append("EXTERNAL_DEPENDENCIES:")
        for ext in externals:
            parts.append(f"  - {ext}")

    # OpenShift specific section
    openshift_specific = spec.get("openshift_specific") or []
    if openshift_specific:
        parts.append("OPENSHIFT_SPECIFIC:")
        for openshift in openshift_specific:
            parts.append(f"  - {openshift}")

    # Concurrency section
    concurrency = spec.get("concurrency") or []
    if concurrency:
        parts.append("CONCURRENCY:")
        for conc in concurrency:
            parts.append(f"  - {conc}")

    # Artifacts section
    artifacts = spec.get("artifacts") or []
    if artifacts:
        parts.append("ARTIFACTS:")
        for artifact in artifacts:
            parts.append(f"  - {artifact}")

    # Steps section with hierarchical structure
    by_steps = spec.get("by_steps", [])
    if by_steps:
        parts.append("STEPS:")
        for step in by_steps:
            description = step.get("description", "")
            if description:
                parts.append(f"  - {description}")

                # Step operations
                step_actions = step.get("actions") or []
                if step_actions:
                    for action in step_actions:
                        gvk = action.get("gvk", "")
                        verb = action.get("verb", "")
                        if gvk and verb:
                            parts.append(f"    - {gvk}:{verb}")
                        elif gvk:
                            parts.append(f"    - {gvk}")
                        elif verb:
                            parts.append(f"    - verb:{verb}")

                # Step type
                step_type = step.get("type", "")
                if step_type:
                    parts.append(f"    - type:{step_type}")

    # Prerequisites section (resources created in setup phase)
    prereq = spec.get("prereq", [])
    if prereq:
        parts.append("PREREQUISITES:")
        for p in prereq:
            parts.append(f"  - {p}")

    # Context section (language-agnostic test context hierarchy)
    context = spec.get("context", [])
    if context:
        parts.append("CONTEXT:")
        for c in context:
            parts.append(f"  - {c}")

    return "\n".join(parts)


def expand_equivalents(tokens: set) -> set:
    toks = set(tokens)
    has_route = any(("route.openshift.io" in t) or ("/Route:" in t) for t in toks)
    has_ing = any(("networking.k8s.io" in t) or ("/Ingress:" in t) for t in toks)
    if has_route and not has_ing:
        toks.add("networking.k8s.io/v1/Ingress:create")
    if has_ing and not has_route:
        toks.add("route.openshift.io/v1/Route:create")
    return toks


def build_embeddings(specs: List[Dict[str, Any]], model) -> np.ndarray:
    texts = [spec_to_text(s) for s in specs]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return np.array(embs, dtype="float32")


def tokens_from_spec(s: Dict[str, Any]) -> set:
    toks = set()
    actions = s.get("actions") or []
    for act in actions:
        gvk = act.get("gvk", "")
        kind_hint = (act.get("fields") or {}).get("kind_hint", "")
        verb = act.get("verb", "")
        if gvk and kind_hint and "/" not in gvk:
            gvk = f"{gvk}/{kind_hint}"
        if gvk and verb:
            toks.add(f"{gvk}:{verb}")
    return toks


def get_resource_tokens(s: Dict[str, Any]) -> Set[str]:
    """Extract resource-level tokens (just GVK) from a spec's actions."""
    actions = s.get("actions") or []
    tokens = set()
    for a in actions:
        gvk = a.get("gvk", "")
        if gvk:
            tokens.add(gvk)
    return tokens


def get_category_tokens(s: Dict[str, Any]) -> Set[str]:
    """Extract category-level tokens (resource:category) from a spec's actions."""
    actions = s.get("actions") or []
    tokens = set()
    for a in actions:
        gvk = a.get("gvk", "")
        verb = a.get("verb", "")
        if gvk and verb:
            # Find the category for this verb
            category = None
            for cat, verbs in OPERATION_CATEGORIES.items():
                if verb in verbs:
                    category = cat
                    break
            if category:
                tokens.add(f"{gvk}:{category}")
    return tokens


def get_verb_group_tokens(s: Dict[str, Any]) -> Set[str]:
    """Extract verb group tokens (resource:verb_group) from a spec's actions."""
    actions = s.get("actions") or []
    tokens = set()
    for a in actions:
        gvk = a.get("gvk", "")
        verb = a.get("verb", "")
        if gvk and verb:
            # Find the verb group for this verb
            verb_group = None
            for group, verbs in VERB_GROUPS.items():
                if verb in verbs:
                    verb_group = group
                    break
            if verb_group:
                tokens.add(f"{gvk}:{verb_group}")
    return tokens


def get_resources_from_spec(spec: Dict[str, Any]) -> List[str]:
    """Extract unique resource types (GVKs) from a spec.
    Resources are extracted from:
    1. prereq - resources created in setup phase
    2. actions - resources from all actions
    3. expectations - resources from expectations
    """
    resources = set()

    # 1. Add resources from prereq (setup phase resources)
    prereq = spec.get("prereq", [])
    if isinstance(prereq, list):
        for gvk in prereq:
            if gvk:
                resources.add(gvk)

    # 2. Add resources from actions
    actions = spec.get("actions", [])
    for action in actions:
        if isinstance(action, dict):
            # Add GVK from action
            gvk = action.get("gvk", "")
            if gvk:
                resources.add(gvk)
            # Also check resources field in action (if present)
            action_resources = action.get("resources", [])
            if isinstance(action_resources, list):
                for res in action_resources:
                    if res:
                        resources.add(res)

    # 3. Add resources from expectations
    expectations = spec.get("expectations", [])
    for expectation in expectations:
        if isinstance(expectation, dict):
            exp_resources = expectation.get("resources", [])
            if isinstance(exp_resources, list):
                for res in exp_resources:
                    if res:
                        resources.add(res)

    return list(resources)


def shared_signals(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Enhanced shared signals detection with multiple similarity levels."""
    signals = []

    # 1. Exact operation matches (GVK:verb)
    ta, tb = expand_equivalents(tokens_from_spec(a)), expand_equivalents(tokens_from_spec(b))
    exact_matches = sorted(ta & tb)
    if exact_matches:
        signals.extend([f"exact:{match}" for match in exact_matches])

    # 2. Resource-level matches (same GVK, different verbs)
    ra, rb = get_resource_tokens(a), get_resource_tokens(b)
    resource_matches = sorted(ra & rb)
    if resource_matches:
        signals.extend([f"resource:{match}" for match in resource_matches])

    # 3. Category-level matches (same GVK and operation category)
    ca, cb = get_category_tokens(a), get_category_tokens(b)
    category_matches = sorted(ca & cb)
    if category_matches:
        signals.extend([f"category:{match}" for match in category_matches])

    # 4. Verb group matches (same GVK and verb group)
    va, vb = get_verb_group_tokens(a), get_verb_group_tokens(b)
    verb_group_matches = sorted(va & vb)
    if verb_group_matches:
        signals.extend([f"verb_group:{match}" for match in verb_group_matches])

    # 5. Purpose compatibility
    purpose_a = a.get("purpose", "")
    purpose_b = b.get("purpose", "")
    if purpose_a and purpose_b:
        if purpose_a == purpose_b:
            signals.append(f"purpose:{purpose_a}")
        elif is_purpose_compatible(purpose_a, purpose_b):
            signals.append(f"purpose_compatible:{purpose_a}~{purpose_b}")

    # 6. Resource compatibility
    resources_a = get_resources_from_spec(a)
    resources_b = get_resources_from_spec(b)
    if resources_a and resources_b:
        common_resources = set(resources_a) & set(resources_b)
        if common_resources:
            signals.append(f"resources:{','.join(sorted(common_resources))}")
        elif is_resource_compatible(resources_a, resources_b):
            signals.append(
                f"resources_compatible:{','.join(sorted(resources_a))}~{','.join(sorted(resources_b))}"
            )

    # 7. Networking technology compatibility
    tech_a = a.get("tech", [])
    tech_b = b.get("tech", [])
    if tech_a and tech_b:
        common_techs = set(tech_a) & set(tech_b)
        if common_techs:
            signals.append(f"tech:{','.join(sorted(common_techs))}")
        elif is_tech_compatible(tech_a, tech_b):
            signals.append(f"tech_compatible:{','.join(sorted(tech_a))}~{','.join(sorted(tech_b))}")

    return ";".join(signals)


def cross_match(specs_a, embs_a, specs_b, embs_b, topk=5):
    idx = faiss.IndexFlatIP(embs_b.shape[1])
    idx.add(embs_b)
    sims, nbrs = idx.search(embs_a, topk)
    pairs = []
    for i, (scores, nbr) in enumerate(zip(sims, nbrs)):
        for j, sc in zip(nbr, scores):
            # Calculate shared signals
            shared = shared_signals(specs_a[i], specs_b[j])

            # Enhanced scoring based on different types of shared signals, purpose compatibility, and functional similarity
            boosted_score = float(sc)

            # Purpose-based scoring
            purpose_a = specs_a[i].get("purpose", "")
            purpose_b = specs_b[j].get("purpose", "")
            purpose_boost = 0.0

            if purpose_a and purpose_b:
                if purpose_a == purpose_b:
                    purpose_boost = 0.05  # Same purpose gets small boost
                elif is_purpose_compatible(purpose_a, purpose_b):
                    purpose_boost = 0.025  # Compatible purposes get tiny boost
                else:
                    purpose_boost = -0.08  # Incompatible purposes get small penalty

            # Networking technology compatibility scoring
            tech_a = specs_a[i].get("tech", [])
            tech_b = specs_b[j].get("tech", [])
            tech_boost = 0.0

            # ENHANCED: Technology-aware penalty for mismatches
            if tech_a and not tech_b:
                # Technology-specific test vs generic test - heavy penalty
                tech_boost = -0.4
            elif tech_b and not tech_a:
                # Generic test vs technology-specific test - heavy penalty
                tech_boost = -0.4
            elif tech_a and tech_b:
                common_techs = set(tech_a) & set(tech_b)
                if common_techs:
                    tech_boost = 0.04 * min(
                        len(common_techs), 2
                    )  # Smaller boost for common technologies
                elif is_tech_compatible(tech_a, tech_b):
                    tech_boost = 0.02  # Tiny boost for compatible technologies
                else:
                    tech_boost = -0.2  # Medium penalty for incompatible technologies

            # Functional similarity scoring
            functional_score = calculate_functional_similarity(specs_a[i], specs_b[j])
            functional_boost = (
                functional_score * 0.08
            )  # Much smaller boost based on functional similarity

            if shared:
                # Count different types of shared signals
                exact_count = len([s for s in shared.split(";") if s.startswith("exact:")])
                resource_count = len([s for s in shared.split(";") if s.startswith("resource:")])
                category_count = len([s for s in shared.split(";") if s.startswith("category:")])
                verb_group_count = len(
                    [s for s in shared.split(";") if s.startswith("verb_group:")]
                )

                # Boost based on signal types (exact > category > verb_group > resource)
                signal_boost = 0.0
                if exact_count > 0:
                    signal_boost += 0.03 * min(
                        exact_count, 2
                    )  # Much smaller boost for exact matches, capped at 2
                if category_count > 0:
                    signal_boost += 0.025 * min(
                        category_count, 2
                    )  # Much smaller boost for category matches, capped at 2
                if verb_group_count > 0:
                    signal_boost += 0.02 * min(
                        verb_group_count, 2
                    )  # Much smaller boost for verb group matches, capped at 2
                if resource_count > 0:
                    signal_boost += 0.015 * min(
                        resource_count, 2
                    )  # Much smaller boost for resource matches, capped at 2

                boosted_score = min(
                    1.0,
                    float(sc) + signal_boost + purpose_boost + tech_boost + functional_boost,
                )
            else:
                boosted_score = min(
                    1.0,
                    max(0.0, float(sc) + purpose_boost + tech_boost + functional_boost),
                )

            # Apply cross-language threshold filtering
            test_id_a = specs_a[i]["test_id"]
            test_id_b = specs_b[j]["test_id"]
            lang_a = get_language_from_test_id(test_id_a)
            lang_b = get_language_from_test_id(test_id_b)

            # Apply context-aware filtering
            context_penalty = calculate_context_penalty(specs_a[i], specs_b[j])

            # Apply framework bias penalty for generic operations and false positives
            framework_penalty = calculate_framework_bias_penalty(specs_a[i], specs_b[j])

            # Apply same-file penalty for tests from the same file with different names
            same_file_penalty = calculate_same_file_penalty(specs_a[i], specs_b[j])

            # Combine penalties (multiplicative for context, framework bias, and same-file)
            final_score = (
                boosted_score
                * (1.0 - context_penalty)
                * (1.0 - framework_penalty)
                * (1.0 - same_file_penalty)
            )

            # Apply higher threshold for cross-language matches
            if lang_a != lang_b:
                # Cross-language matches need higher similarity to be included
                cross_lang_threshold = (
                    0.5  # Reduced threshold to allow meaningful cross-language matches
                )
                if final_score < cross_lang_threshold:
                    continue  # Skip low-scoring cross-language matches

            # Update the final blended score with context penalty
            boosted_score = final_score

            pairs.append(
                {
                    "idx_a": i,
                    "idx_b": int(j),
                    "base_score": float(sc),
                    "blended_score": boosted_score,
                    "a_test": specs_a[i]["test_id"],
                    "b_test": specs_b[j]["test_id"],
                    "shared_signals": shared,
                }
            )
    return pairs


def validate_high_similarity_matches(pairs, specs_a, specs_b, threshold=0.8):
    """Validate that high-similarity matches actually share operations and compatible purposes."""
    high_sim_matches = [p for p in pairs if p["base_score"] >= threshold]
    shared_ops_matches = [p for p in high_sim_matches if p["shared_signals"]]

    # Count different types of shared signals
    exact_matches = [
        p
        for p in high_sim_matches
        if any(s.startswith("exact:") for s in p["shared_signals"].split(";"))
    ]
    resource_matches = [
        p
        for p in high_sim_matches
        if any(s.startswith("resource:") for s in p["shared_signals"].split(";"))
    ]
    category_matches = [
        p
        for p in high_sim_matches
        if any(s.startswith("category:") for s in p["shared_signals"].split(";"))
    ]
    verb_group_matches = [
        p
        for p in high_sim_matches
        if any(s.startswith("verb_group:") for s in p["shared_signals"].split(";"))
    ]

    # Count purpose compatibility, networking tech compatibility, and functional similarity
    purpose_compatible_matches = []
    purpose_same_matches = []
    tech_compatible_matches = []
    tech_same_matches = []
    functional_matches = []
    for p in high_sim_matches:
        idx_a = p["idx_a"]
        idx_b = p["idx_b"]

        # Check bounds to avoid index errors
        if idx_a >= len(specs_a) or idx_b >= len(specs_b):
            continue

        purpose_a = specs_a[idx_a].get("purpose", "")
        purpose_b = specs_b[idx_b].get("purpose", "")
        if purpose_a and purpose_b:
            if purpose_a == purpose_b:
                purpose_same_matches.append(p)
            elif is_purpose_compatible(purpose_a, purpose_b):
                purpose_compatible_matches.append(p)

        # Check networking technology compatibility
        tech_a = specs_a[idx_a].get("tech", [])
        tech_b = specs_b[idx_b].get("tech", [])
        if tech_a and tech_b:
            common_techs = set(tech_a) & set(tech_b)
            if common_techs:
                tech_same_matches.append(p)
            elif is_tech_compatible(tech_a, tech_b):
                tech_compatible_matches.append(p)

        # Check functional similarity
        functional_score = calculate_functional_similarity(specs_a[idx_a], specs_b[idx_b])
        if functional_score > 0.3:  # Threshold for meaningful functional similarity
            functional_matches.append(p)

    print(f"High similarity matches (>{threshold}): {len(high_sim_matches)}")
    print(f"  - Exact operation matches: {len(exact_matches)}")
    print(f"  - Resource-level matches: {len(resource_matches)}")
    print(f"  - Category-level matches: {len(category_matches)}")
    print(f"  - Verb group matches: {len(verb_group_matches)}")
    print(f"  - Any shared signals: {len(shared_ops_matches)}")
    print(f"  - Same purpose: {len(purpose_same_matches)}")
    print(f"  - Compatible purpose: {len(purpose_compatible_matches)}")
    print(f"  - Same networking tech: {len(tech_same_matches)}")
    print(f"  - Compatible networking tech: {len(tech_compatible_matches)}")
    print(f"  - Functional similarity: {len(functional_matches)}")

    if len(high_sim_matches) > 0:
        validation_rate = len(shared_ops_matches) / len(high_sim_matches) * 100
        purpose_rate = (
            (len(purpose_same_matches) + len(purpose_compatible_matches))
            / len(high_sim_matches)
            * 100
        )
        tech_rate = (
            (len(tech_same_matches) + len(tech_compatible_matches)) / len(high_sim_matches) * 100
        )
        functional_rate = len(functional_matches) / len(high_sim_matches) * 100
        print(f"Operation validation rate: {validation_rate:.1f}%")
        print(f"Purpose compatibility rate: {purpose_rate:.1f}%")
        print(f"Networking tech compatibility rate: {tech_rate:.1f}%")
        print(f"Functional similarity rate: {functional_rate:.1f}%")

        if validation_rate < 50:
            print(
                f"⚠️  {len(high_sim_matches) - len(shared_ops_matches)} high-similarity matches lack shared operations!"
            )
            print("   These may be false positives based on text similarity alone.")
        else:
            print("✅ Good validation rate - most high-similarity matches have shared operations!")

    return shared_ops_matches


def write_report(pairs_ab, pairs_ba, go_specs, py_specs, out_csv):
    import pandas as pd

    df = pd.DataFrame(pairs_ab + pairs_ba)
    score_col = "blended_score" if "blended_score" in df.columns else "base_score"
    df.sort_values(score_col, ascending=False, inplace=True)

    # Validate high similarity matches
    print("\n=== VALIDATION RESULTS ===")
    validate_high_similarity_matches(pairs_ab + pairs_ba, go_specs, py_specs)

    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")


def write_comprehensive_report(pairs, all_specs, outfile):
    """Write comprehensive similarity report to CSV (language-agnostic)."""
    import pandas as pd

    rows = []
    for p in pairs:
        spec_a = all_specs[p["idx_a"]]
        spec_b = all_specs[p["idx_b"]]

        rows.append(
            {
                "idx_a": p["idx_a"],
                "idx_b": p["idx_b"],
                "a_test": spec_a["test_id"],
                "b_test": spec_b["test_id"],
                "a_language": spec_a["_language"],
                "b_language": spec_b["_language"],
                "a_repo": spec_a["_repo"],
                "b_repo": spec_b["_repo"],
                "base_score": p["base_score"],
                "blended_score": p["blended_score"],
                "shared_signals": p["shared_signals"],
                "match_type": f"{spec_a['_language']}->{spec_b['_language']}",
            }
        )

    df = pd.DataFrame(rows)
    score_col = "blended_score" if "blended_score" in df.columns else "base_score"
    df.sort_values(score_col, ascending=False, inplace=True)

    # Validate high similarity matches
    print("\n=== VALIDATION RESULTS ===")
    validate_high_similarity_matches(pairs, all_specs, all_specs)

    df.to_csv(outfile, index=False)
    print(f"Wrote {outfile} ({len(df)} rows)")

    # Print summary by match type
    match_types = df["match_type"].value_counts()
    print("Match type distribution:")
    for match_type, count in match_types.items():
        print(f"  {match_type}: {count} matches")


def coverage_matrix(specs, repo_label):
    from collections import Counter

    cv = Counter()
    for s in specs:
        actions = s.get("actions") or []
        for a in actions:
            gvk = a.get("gvk", "")
            kind_hint = (a.get("fields") or {}).get("kind_hint", "")
            verb = a.get("verb", "")
            if gvk and kind_hint and "/" not in gvk:
                gvk = f"{gvk}/{kind_hint}"
            if gvk and verb:
                cv[(gvk, verb)] += 1
    rows = []
    for (gvk, verb), cnt in sorted(cv.items()):
        rows.append({"repo": repo_label, "gvk": gvk, "verb": verb, "count": cnt})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", required=True, help="JSONL from Go extractor")
    ap.add_argument("--py", required=True, help="JSONL from Python extractor")
    ap.add_argument("--out", default="report.csv")
    ap.add_argument("--cov", default="coverage_matrix.csv")
    ap.add_argument("--llm", action="store_true", help="use LLM re-ranking (env vars required)")
    args = ap.parse_args()

    # Load and combine all specs regardless of language
    go_specs = load_specs(args.go)
    for s in go_specs:
        s["_repo"] = "go"
        s["_language"] = "go"

    py_specs = load_specs(args.py)
    for s in py_specs:
        s["_repo"] = "py"
        s["_language"] = "py"

    # Filter out empty tests
    print("Filtering Go specs...")
    go_specs = filter_empty_tests(go_specs)
    print("Filtering Python specs...")
    py_specs = filter_empty_tests(py_specs)

    # Combine all specs for comprehensive analysis
    all_specs = go_specs + py_specs
    print(
        f"Total specs for analysis: {len(all_specs)} (Go: {len(go_specs)}, Python: {len(py_specs)})"
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    all_embs = build_embeddings(all_specs, model)

    # Find similar tests across all specs (including within same language)
    print("Finding similar tests across all languages...")
    pairs = cross_match(all_specs, all_embs, all_specs, all_embs, topk=10)

    # Remove self-matches and duplicates
    filtered_pairs = []
    seen_pairs = set()
    for pair in pairs:
        idx_a, idx_b = pair["idx_a"], pair["idx_b"]
        # Skip self-matches
        if idx_a == idx_b:
            continue
        # Skip duplicate pairs (A-B and B-A)
        pair_key = tuple(sorted([idx_a, idx_b]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            filtered_pairs.append(pair)

    print(f"Found {len(filtered_pairs)} unique similar test pairs")

    # Apply purpose-based filtering to reduce false positives
    print(f"Before purpose filtering: {len(filtered_pairs)} matches")
    filtered_pairs = filter_by_purpose_compatibility(filtered_pairs, all_specs, all_specs)
    print(f"After purpose filtering: {len(filtered_pairs)} matches")

    if args.llm:
        from llm_rerank import rerank_batch

        print("Re-ranking with LLM...")
        filtered_pairs = rerank_batch(filtered_pairs, all_specs, all_specs)

    # Write comprehensive report
    write_comprehensive_report(filtered_pairs, all_specs, args.out)

    # Generate coverage matrix for all specs
    df_all = coverage_matrix(all_specs, "all")
    df_all.to_csv(args.cov, index=False)
    print(f"Wrote {args.cov} ({len(df_all)} rows)")


if __name__ == "__main__":
    main()
