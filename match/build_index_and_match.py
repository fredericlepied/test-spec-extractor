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
    "OPERATOR_MANAGEMENT": ["RESOURCE_VALIDATION", "UPGRADE_TESTING"],  # Only with resource validation and upgrade
    "STORAGE_TESTING": ["RESOURCE_VALIDATION"],  # Only with resource validation
    "SECURITY_TESTING": ["RESOURCE_VALIDATION"],  # Only with resource validation
    "CONFIGURATION": ["RESOURCE_VALIDATION", "UPGRADE_TESTING"],  # Only with resource validation and upgrade
    "PERFORMANCE": ["RESOURCE_VALIDATION"],  # Only with resource validation
    "RESOURCE_VALIDATION": [
        "OPERATOR_MANAGEMENT",
        "STORAGE_TESTING",
        "SECURITY_TESTING",
        "CONFIGURATION",
        "PERFORMANCE",
    ],  # Removed network and pod purposes
    # Specialized testing purposes - only match with themselves or very specific others
    "UPGRADE_TESTING": ["UPGRADE_TESTING", "OPERATOR_MANAGEMENT", "CONFIGURATION"],  # Upgrade can match with operator management and configuration
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
    "v1/Pod": ["v1/Pod", "apps/v1/Deployment", "apps/v1/StatefulSet", "apps/v1/DaemonSet"],
    "apps/v1/Deployment": ["v1/Pod", "apps/v1/Deployment", "apps/v1/StatefulSet"],
    "apps/v1/StatefulSet": ["v1/Pod", "apps/v1/Deployment", "apps/v1/StatefulSet"],
    "apps/v1/DaemonSet": ["v1/Pod", "apps/v1/DaemonSet"],
    
    # Namespace-related resources
    "v1/Namespace": ["v1/Namespace", "v1/Pod", "apps/v1/Deployment", "apps/v1/StatefulSet"],
    
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
    "operators.coreos.com/v1alpha1/ClusterServiceVersion": ["operators.coreos.com/v1alpha1/ClusterServiceVersion"],
    "operators.coreos.com/v1alpha1/Subscription": ["operators.coreos.com/v1alpha1/Subscription"],
}

# Technology compatibility matrix - expanded to include all technology domains
TECH_COMPATIBILITY = {
    # Networking technologies
    "SR-IOV": ["SR-IOV", "CNI", "Virtualization"],  # SR-IOV can match with CNI and virtualization
    "PTP": ["PTP", "CNI", "Edge Computing"],  # PTP can match with CNI and edge computing
    "DPDK": ["DPDK", "CNI", "Virtualization"],  # DPDK can match with CNI and virtualization
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
    "GPU": ["GPU", "Machine Learning", "Virtualization"],  # GPU can match with ML and virtualization
    
    # Virtualization
    "Virtualization": ["SR-IOV", "DPDK", "CNI", "GPU", "Storage", "Security"],  # Virtualization is compatible with many technologies
    
    # Storage
    "Storage": ["RDMA", "Virtualization", "Security", "Monitoring"],  # Storage can match with RDMA, virtualization, security, monitoring
    
    # Security
    "Security": ["Virtualization", "Storage", "Monitoring", "Edge Computing"],  # Security is compatible with many domains
    
    # Monitoring/Observability
    "Monitoring": ["Storage", "Security", "Edge Computing", "Machine Learning"],  # Monitoring is compatible with many domains
    
    # Machine Learning/AI
    "Machine Learning": ["GPU", "Edge Computing", "Monitoring"],  # ML can match with GPU, edge computing, monitoring
    
    # Edge Computing
    "Edge Computing": ["PTP", "Security", "Monitoring", "Machine Learning"],  # Edge computing is compatible with time-sensitive and monitoring tech
    
    # Power Management
    "Power Management": ["Power Management", "Edge Computing"],  # Power management can match with edge computing
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


def calculate_functional_similarity(
    spec_a: Dict[str, Any], spec_b: Dict[str, Any]
) -> float:
    """Calculate functional similarity score based on test functionality overlap."""
    actions_a = spec_a.get("actions") or []
    actions_b = spec_b.get("actions") or []
    expectations_a = spec_a.get("expectations") or []
    expectations_b = spec_b.get("expectations") or []

    if not actions_a or not actions_b:
        return 0.0

    # Extract operations and expectations
    operations_a = set()
    operations_b = set()
    expectations_set_a = set()
    expectations_set_b = set()

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

    # Calculate similarity scores
    exact_ops = operations_a & operations_b
    category_ops = set()
    resource_ops = set()

    # Category-level operations
    for action in actions_a:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            for cat, verbs in OPERATION_CATEGORIES.items():
                if verb in verbs:
                    category_ops.add(f"{gvk}:{cat}")
                    break

    for action in actions_b:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            for cat, verbs in OPERATION_CATEGORIES.items():
                if verb in verbs:
                    category_ops.add(f"{gvk}:{cat}")
                    break

    category_matches = set()
    for cat_a in category_ops:
        for cat_b in category_ops:
            if cat_a == cat_b:
                category_matches.add(cat_a)

    # Resource-level operations
    resources_a = set()
    resources_b = set()
    for action in actions_a:
        gvk = action.get("gvk", "")
        if gvk:
            resources_a.add(gvk)

    for action in actions_b:
        gvk = action.get("gvk", "")
        if gvk:
            resources_b.add(gvk)

    resource_matches = resources_a & resources_b

    # Calculate weighted similarity score
    exact_score = len(exact_ops) * 1.0
    category_score = len(category_matches) * 0.7
    resource_score = len(resource_matches) * 0.3
    expectation_score = len(expectations_set_a & expectations_set_b) * 0.5

    total_ops = len(operations_a | operations_b)
    total_expectations = len(expectations_set_a | expectations_set_b)

    if total_ops == 0:
        return 0.0

    functional_score = (exact_score + category_score + resource_score) / total_ops
    if total_expectations > 0:
        functional_score += expectation_score / total_expectations * 0.2

    return min(1.0, functional_score)


def has_meaningful_operations(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> bool:
    """Check if two specs have meaningful operational overlap beyond just resource-level similarity."""
    functional_score = calculate_functional_similarity(spec_a, spec_b)
    return functional_score > 0.3  # Threshold for meaningful operations


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
    # Don't include test_id in the content - it's used as document ID
    parts = []

    # Include test_type field to differentiate empty tests
    test_type = spec.get("test_type", "unknown")
    if test_type:
        parts.append(f"test_type:{test_type}")

    # Handle None values by converting to empty lists
    dependencies = spec.get("dependencies") or []
    actions = spec.get("actions") or []
    expectations = spec.get("expectations") or []
    externals = spec.get("externals") or []
    openshift_specific = spec.get("openshift_specific") or []
    concurrency = spec.get("concurrency") or []
    artifacts = spec.get("artifacts") or []

    parts += dependencies
    for a in actions:
        gvk = a.get("gvk", "")
        kind_hint = (a.get("fields") or {}).get("kind_hint", "")
        verb = a.get("verb", "")
        if gvk and kind_hint and "/" not in gvk:
            gvk = f"{gvk}/{kind_hint}"
        if gvk and verb:
            parts.append(f"{gvk}:{verb}")
        elif gvk:
            parts.append(gvk)
        elif verb:
            parts.append(f"verb:{verb}")
    parts += [
        f"expect:{e.get('target','')}={e.get('condition','')}" for e in expectations
    ]
    parts += [f"ext:{x}" for x in externals]
    parts += openshift_specific
    parts += concurrency
    parts += artifacts
    return "\n".join(map(str, parts))


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
    embs = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
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
    """Extract unique resource types (GVKs) from a spec."""
    resources = set()
    actions = spec.get("actions", [])
    for action in actions:
        if isinstance(action, dict) and "gvk" in action:
            resources.add(action["gvk"])
    return list(resources)


def shared_signals(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Enhanced shared signals detection with multiple similarity levels."""
    signals = []

    # 1. Exact operation matches (GVK:verb)
    ta, tb = expand_equivalents(tokens_from_spec(a)), expand_equivalents(
        tokens_from_spec(b)
    )
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
            signals.append(
                f"tech_compatible:{','.join(sorted(tech_a))}~{','.join(sorted(tech_b))}"
            )

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
                    purpose_boost = 0.20  # Same purpose gets significant boost
                elif is_purpose_compatible(purpose_a, purpose_b):
                    purpose_boost = 0.10  # Compatible purposes get moderate boost
                else:
                    purpose_boost = -0.30  # Incompatible purposes get penalty

            # Networking technology compatibility scoring
            tech_a = specs_a[i].get("tech", [])
            tech_b = specs_b[j].get("tech", [])
            tech_boost = 0.0

            if tech_a and tech_b:
                common_techs = set(tech_a) & set(tech_b)
                if common_techs:
                    tech_boost = 0.15 * len(
                        common_techs
                    )  # Boost for common technologies
                elif is_tech_compatible(tech_a, tech_b):
                    tech_boost = 0.08  # Moderate boost for compatible technologies
                else:
                    tech_boost = -0.20  # Penalty for incompatible technologies

            # Functional similarity scoring
            functional_score = calculate_functional_similarity(specs_a[i], specs_b[j])
            functional_boost = (
                functional_score * 0.25
            )  # Boost based on functional similarity

            if shared:
                # Count different types of shared signals
                exact_count = len(
                    [s for s in shared.split(";") if s.startswith("exact:")]
                )
                resource_count = len(
                    [s for s in shared.split(";") if s.startswith("resource:")]
                )
                category_count = len(
                    [s for s in shared.split(";") if s.startswith("category:")]
                )
                verb_group_count = len(
                    [s for s in shared.split(";") if s.startswith("verb_group:")]
                )

                # Boost based on signal types (exact > category > verb_group > resource)
                signal_boost = 0.0
                if exact_count > 0:
                    signal_boost += (
                        0.15 * exact_count
                    )  # Highest boost for exact matches
                if category_count > 0:
                    signal_boost += (
                        0.12 * category_count
                    )  # High boost for category matches
                if verb_group_count > 0:
                    signal_boost += (
                        0.10 * verb_group_count
                    )  # Medium boost for verb group matches
                if resource_count > 0:
                    signal_boost += (
                        0.08 * resource_count
                    )  # Lower boost for resource matches

                boosted_score = min(
                    1.0,
                    float(sc)
                    + signal_boost
                    + purpose_boost
                    + tech_boost
                    + functional_boost,
                )
            else:
                boosted_score = min(
                    1.0,
                    max(0.0, float(sc) + purpose_boost + tech_boost + functional_boost),
                )

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
        functional_score = calculate_functional_similarity(
            specs_a[idx_a], specs_b[idx_b]
        )
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
            (len(tech_same_matches) + len(tech_compatible_matches))
            / len(high_sim_matches)
            * 100
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
            print(
                "✅ Good validation rate - most high-similarity matches have shared operations!"
            )

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
    ap.add_argument(
        "--llm", action="store_true", help="use LLM re-ranking (env vars required)"
    )
    args = ap.parse_args()

    go_specs = load_specs(args.go)
    for s in go_specs:
        s["_repo"] = "go"
    py_specs = load_specs(args.py)
    for s in py_specs:
        s["_repo"] = "py"

    # Filter out empty tests
    print("Filtering Go specs...")
    go_specs = filter_empty_tests(go_specs)
    print("Filtering Python specs...")
    py_specs = filter_empty_tests(py_specs)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    go_embs = build_embeddings(go_specs, model)
    py_embs = build_embeddings(py_specs, model)

    pairs_ab = cross_match(go_specs, go_embs, py_specs, py_embs, topk=5)
    pairs_ba = cross_match(py_specs, py_embs, go_specs, go_embs, topk=5)

    # Apply purpose-based filtering to reduce false positives
    print(
        f"Before purpose filtering: {len(pairs_ab)} A->B matches, {len(pairs_ba)} B->A matches"
    )
    pairs_ab = filter_by_purpose_compatibility(pairs_ab, go_specs, py_specs)
    pairs_ba = filter_by_purpose_compatibility(pairs_ba, py_specs, go_specs)
    print(
        f"After purpose filtering: {len(pairs_ab)} A->B matches, {len(pairs_ba)} B->A matches"
    )

    if args.llm:
        from llm_rerank import rerank_batch

        print("Re-ranking A->B with LLM...")
        pairs_ab = rerank_batch(pairs_ab, go_specs, py_specs)
        print("Re-ranking B->A with LLM...")
        pairs_ba = rerank_batch(pairs_ba, py_specs, go_specs)

    write_report(pairs_ab, pairs_ba, go_specs, py_specs, args.out)

    df_go = coverage_matrix(go_specs, "go")
    df_py = coverage_matrix(py_specs, "py")
    cov = pd.concat([df_go, df_py], ignore_index=True)
    cov.to_csv(args.cov, index=False)
    print(f"Wrote {args.cov} ({len(cov)} rows)")


if __name__ == "__main__":
    main()
