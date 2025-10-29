# Comprehensive Test Similarity Analyzer (Go + Python)

A sophisticated toolkit that extracts KubeSpecs from Go and Python test files, builds semantic embeddings, and performs **comprehensive language-agnostic similarity analysis** with **purpose-based filtering** to reduce false positives. OpenShift-aware with automatic equivalence detection (Route↔Ingress, SCC↔PSA) and optional LLM re-ranking.

## 🎯 Key Features

### **Comprehensive Language-Agnostic Analysis**
- **All Tests Compared**: Go↔Go, Python↔Python, and Go↔Python similarity analysis
- **Intra-Language Duplicates**: Finds exact duplicates within the same language (95.4% of matches)
- **Cross-Language Opportunities**: Identifies patterns that can be shared between languages
- **True Duplicate Detection**: 8 exact duplicates found with 100% similarity scores

### **Purpose-Based Filtering System**
- **Intelligent Purpose Detection**: Automatically categorizes tests by their purpose (POD_HEALTH, NETWORK_CONNECTIVITY, POD_MANAGEMENT, etc.)
- **Compatibility Matrix**: Only matches tests with compatible purposes, eliminating false positives
- **78% Reduction in False Positives**: From 1898 to 416 matches with much higher quality

### **Advanced Test Analysis**
- **Multi-Level Similarity**: Exact operations, resource-level, category-level, and verb-group matching
- **Enhanced Scoring**: Purpose-based boosts (+0.20 same purpose, +0.10 compatible, -0.30 incompatible)
- **OpenShift Awareness**: Automatic detection of Route↔Ingress and SCC↔PSA equivalents
- **Utility Test Filtering**: Automatically filters out helper functions and utility tests

### **Granular Test Extraction**
- **By(...) Step Extraction**: Tracks operations within individual `By(...)` steps for fine-grained analysis
- **Individual It(...) Block Extraction**: Extracts separate specs for each `It(...)` test block instead of file-level consolidation
- **Step-Level Operation Mapping**: Maps Kubernetes operations to specific test steps for better traceability

### **Comprehensive Extraction**
- **Go Tests**: Supports Ginkgo/Gomega, eco-goinfra, standard Go tests
- **Python Tests**: Supports pytest, openshift library, subprocess calls
- **Cross-File Detection**: Detects operations in helper functions across files
- **Rich Metadata**: Extracts actions, expectations, dependencies, environment, tech

## 🚀 Quick Start

### **Automated Pipeline (Recommended)**

```bash
# Single repositories
./extract-and-match.sh -g /path/to/eco-gotests -p /path/to/eco-pytests

# Multiple repositories
./extract-and-match.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -p /path/to/eco-pytests

# Custom output directory
./extract-and-match.sh -g /path/to/go-tests -p /path/to/py-tests --output-dir my_results
```

### **What You Get**

The pipeline generates:
- `test_report.csv`: Comprehensive similarity matches (Go↔Go, Python↔Python, Go↔Python) with scores and shared operations
- `test_coverage.csv`: Coverage matrix of operations across all test suites
- `go_specs.jsonl` / `py_specs.jsonl`: Raw extracted test specifications
- `*_analysis.json`: Individual test suite analyses (one per repository)
- `*_report.md`: High-level reports for test suite owners (one per repository)
- `similarity_report.md`: Comprehensive similarity analysis report with match type distribution
- `all_suite_comparisons.json`: Comprehensive cross-suite comparisons

## 🔍 New Analysis Capabilities

### **Language-Agnostic Similarity Analysis**

The tool now performs comprehensive similarity analysis across all test languages:

**Match Types:**
- **Go↔Go (81.0%)**: Intra-language Go test similarities
- **Python↔Python (14.4%)**: Intra-language Python test similarities  
- **Go↔Python (4.6%)**: Cross-language test opportunities

**Key Benefits:**
- **Duplicate Detection**: Find exact duplicates within the same language
- **Consolidation Opportunities**: Identify tests that can be parameterized or merged
- **Cross-Pollination**: Discover patterns that work well in one language for another
- **Quality Improvement**: Reduce test maintenance overhead by eliminating redundancy

**Example Findings:**
- **8 Exact Duplicates**: 100% identical tests found (e.g., `test_agent_cluster_creation` in CU vs DU)
- **Functional Duplicates**: 96.2% similar tests that differ only in parameters
- **Cross-Language Patterns**: Similar namespace lifecycle patterns in Go and Python

## 📊 Analyzing Results

### **Understanding the Test Report (`test_report.csv`)**

The test report contains comprehensive similarity matches across all languages with detailed scoring:

```csv
idx_a,idx_b,a_test,b_test,a_language,b_language,a_repo,b_repo,base_score,blended_score,shared_signals,match_type
240,283,eco-pytests/cu/test.py:test_cu_pods_count,eco-pytests/du/test.py:test_du_pods_count,py,py,py,py,0.962,1.0,exact:v1/Pod:get;exact:v1/Pod:list,py->py
```

**Column Descriptions:**

- `idx_a`, `idx_b`: Index references to the original spec files
- `a_test`, `b_test`: Test identifiers (format: `repo/path/file:function`)
- `a_language`, `b_language`: Programming language (go/py)
- `a_repo`, `b_repo`: Repository identifier
- `base_score`: Raw semantic similarity score (0.0-1.0)
- `blended_score`: Final score after purpose-based filtering and boosts
- `shared_signals`: Types of shared operations between tests
- `match_type`: Match category (go->go, py->py, py->go, go->py)

**Shared Signal Types:**

- `exact:gvk:verb`: Exact operation match (e.g., `exact:v1/Pod:get`)
- `resource:gvk`: Resource-level match (e.g., `resource:v1/Pod`)
- `category:gvk:category`: Operation category match (e.g., `category:v1/Pod:read`)
- `verb_group:gvk:group`: Verb group match (e.g., `verb_group:v1/Pod:read_operations`)

**How to Analyze:**

```bash
# View top matches by blended score
sort -t',' -k4 -nr test_report.csv | head -10

# Find matches with exact operations
grep "exact:" test_report.csv

# Count matches by shared signal type
cut -d',' -f7 test_report.csv | tr ';' '\n' | cut -d':' -f1 | sort | uniq -c

# Filter by specific test
grep "test_function_name" test_report.csv

# Analyze match types
cut -d',' -f12 test_report.csv | sort | uniq -c

# Find intra-language duplicates (same language)
grep -E "(go->go|py->py)" test_report.csv

# Find cross-language opportunities
grep -E "(py->go|go->py)" test_report.csv
```

### **Understanding the Coverage Matrix (`test_coverage.csv`)**

The coverage matrix shows which operations are tested across different test suites:

```csv
operation,go_count,py_count,total_count,coverage_ratio
v1/Pod:get,45,23,68,0.68
v1/Namespace:create,32,15,47,0.47
```

**Column Descriptions:**

- `operation`: Kubernetes operation (GVK:verb format)
- `go_count`: Number of Go tests using this operation
- `py_count`: Number of Python tests using this operation
- `total_count`: Total unique tests using this operation
- `coverage_ratio`: Ratio of tests using this operation

**How to Analyze:**

```bash
# Find most common operations
sort -t',' -k4 -nr test_coverage.csv | head -10

# Find operations only tested in Go
awk -F',' '$3==0 {print $0}' test_coverage.csv

# Find operations only tested in Python
awk -F',' '$2==0 {print $0}' test_coverage.csv

# Find operations with high coverage
awk -F',' '$5>0.5 {print $0}' test_coverage.csv
```

### **Understanding the Similarity Report (`similarity_report.md`)**

The similarity report provides comprehensive analysis of all test relationships:

**Key Sections:**

- **Executive Summary**: Overview of matches, quality indicators, and duplicate ratios
- **Match Type Analysis**: Distribution of intra-language vs cross-language matches
- **Score Distribution**: Visual analysis of similarity score patterns
- **Shared Signals Analysis**: Breakdown of different signal types
- **Potential Duplicates**: High-similarity matches (≥0.95) with consolidation recommendations
- **Complementary Tests**: Medium-similarity matches (0.6-0.8) with different purposes
- **Top Similarity Matches**: Most similar test pairs with detailed analysis
- **Strategic Recommendations**: Actionable insights for test optimization

**How to Use:**

```bash
# View the full report
cat similarity_report.md

# Find duplicate recommendations
grep -A 5 "Potential Duplicates" similarity_report.md

# Check match type distribution
grep -A 10 "Match Type Distribution" similarity_report.md

# Review strategic recommendations
grep -A 20 "Strategic Recommendations" similarity_report.md
```

### **Understanding the Spec Files (`*_specs.jsonl`)**

Each line in the spec files contains a complete test specification in JSON format:

```json
{
  "test_id": "eco-gotests/tests/pod_test.go:TestPodHealth",
  "test_type": "integration",
  "dependencies": ["network", "storage"],
  "environment": ["multi_node"],
  "actions": [{"gvk": "v1/Pod", "verb": "get"}],
  "expectations": [{"target": "resource_status", "condition": "pod.status.phase == 'Running'"}],
  "openshift_specific": ["route.openshift.io/v1/Route"],
  "concurrency": [],
  "artifacts": ["testdata/pod.yaml"],
  "purpose": "POD_HEALTH"
}
```

**Field Descriptions:**

- `test_id`: Unique identifier (repo/path/file:function)
- `test_type`: Test classification (unit, integration, e2e, performance, conformance)
- `dependencies`: Required components (network, storage, operator, etc.)
- `environment`: Target environment (single_node, multi_node, bare_metal, cloud, edge)
- `tech`: Technologies detected (SR-IOV, GPU, Storage, Security, etc.)
- `actions`: Kubernetes operations performed (GVK:verb pairs)
- `expectations`: Test assertions and validations
- `openshift_specific`: OpenShift-specific resources used
- `concurrency`: Concurrency-related patterns
- `artifacts`: Test data files and golden files
- `purpose`: Primary test intent (POD_HEALTH, NETWORK_CONNECTIVITY, etc.)
- `by_steps`: Detailed breakdown of operations within `By(...)` steps (Ginkgo tests)

**How to Analyze:**

```bash
# Count tests by purpose
jq -r '.purpose' go_specs.jsonl | sort | uniq -c | sort -nr

# Find tests with specific operations
jq -r 'select(.actions[].gvk == "v1/Pod") | .test_id' go_specs.jsonl

# Find tests by environment
jq -r 'select(.environment[] == "multi_node") | .test_id' go_specs.jsonl

# Find tests with specific dependencies
jq -r 'select(.dependencies[] == "network") | .test_id' go_specs.jsonl

# Compare test types between Go and Python
echo "Go test types:"; jq -r '.test_type' go_specs.jsonl | sort | uniq -c
echo "Python test types:"; jq -r '.test_type' py_specs.jsonl | sort | uniq -c
```

### **Quality Assessment Workflow**

1. **Check Match Quality:**

   ```bash
   # Look for exact operation matches (highest quality)
   grep "exact:" test_report.csv | head -5
   
   # Check purpose compatibility
   awk -F',' 'NR>1 {print $5, $6}' test_report.csv | head -10
   ```

2. **Identify Coverage Gaps:**

   ```bash
   # Find operations with low coverage
   awk -F',' '$5<0.3 {print $0}' test_coverage.csv
   
   # Find operations only in one language
   awk -F',' '$2==0 || $3==0 {print $0}' test_coverage.csv
   ```

3. **Validate Similarity:**
   ```bash
   # Get details of top matches
   head -5 test_report.csv | while IFS=',' read -r idx_a idx_b base_score blended_score a_test b_test shared_signals; do
     echo "Match: $a_test ↔ $b_test"
     echo "Score: $blended_score"
     echo "Shared: $shared_signals"
     echo "---"
   done
   ```

### **Advanced Analysis Examples**

**Find Cross-Language Test Pairs:**

```bash
# Extract test pairs with their details
awk -F',' 'NR>1 {print $5, $6, $4, $7}' test_report.csv | head -10
```

**Analyze Purpose Distribution:**

```bash
# Go test purposes
jq -r '.purpose' go_specs.jsonl | sort | uniq -c | sort -nr

# Python test purposes  
jq -r '.purpose' py_specs.jsonl | sort | uniq -c | sort -nr
```

**Find Tests by Operation Type:**

```bash
# Tests that create resources
jq -r 'select(.actions[].verb == "create") | .test_id' go_specs.jsonl

# Tests that validate resource status
jq -r 'select(.expectations[].target == "resource_status") | .test_id' py_specs.jsonl
```

## 🔍 Test Suite Analysis

### **Understanding Suite Analysis Files**

The pipeline automatically generates comprehensive analyses for each individual test suite (repository):

#### **Individual Suite Analysis (`*_analysis.json`)**

Each analysis provides detailed insights into a specific test suite:

```json
{
  "suite_name": "Go Tests",
  "total_tests": 1086,
  "coverage_metrics": {
    "unique_operations": 909,
    "unique_resources": 184,
    "avg_operations_per_test": 5.9
  },
  "test_distribution": {
    "by_type": {"integration": 756, "unit": 197, "conformance": 133},
    "by_purpose": {"POD_MANAGEMENT": 422, "RESOURCE_VALIDATION": 261},
    "by_environment": {"multi_node": 334, "cloud": 10}
  },
  "key_insights": [
    "Primary test type: integration (756 tests, 69.6%)",
    "Main focus: POD_MANAGEMENT (422 tests, 38.9%)",
    "Most tested resources: meta/v1/GetOptions (728), v1/Pod (337)"
  ]
}
```

#### **Cross-Suite Comparison (`all_suite_comparisons.json`)**

Identifies gaps and differences between all test suites:

```json
{
  "suite1": "Go Tests",
  "suite2": "Python Tests",
  "test_count_diff": 975,
  "operation_gaps": {
    "unique_to_suite1": ["v1/Pod:create", "v1/Namespace:delete"],
    "unique_to_suite2": ["hive.openshift.io/v1/ClusterDeployment:get"],
    "common": ["v1/Pod:get", "v1/Namespace:get"]
  },
  "recommendations": [
    "Consider adding 907 operations from Go Tests to Python Tests",
    "Consider adding tests for 182 resource types from Go Tests to Python Tests"
  ]
}
```

### **Analyzing Suite Characteristics**

**Test Type Distribution:**
```bash
# View test type breakdown for a specific suite
jq '.test_distribution.by_type' eco-gotests_analysis.json

# Compare test types between all suites
for file in *_analysis.json; do
  echo "=== $(basename $file _analysis.json) ==="
  jq '.test_distribution.by_type' "$file"
done
```

**Purpose Analysis:**
```bash
# Find most common purposes for a specific suite
jq '.test_distribution.by_purpose' eco-gotests_analysis.json

# Identify purpose gaps between all suites
jq '.purpose_gaps' all_suite_comparisons.json
```

**Resource Coverage:**
```bash
# Top resources by usage for a specific suite
jq '.resources | to_entries | sort_by(.value) | reverse | .[0:10]' eco-gotests_analysis.json

# Resource gaps between all suites
jq '.resource_gaps' all_suite_comparisons.json
```

**Operation Analysis:**
```bash
# Most common operations for a specific suite
jq '.operations | to_entries | sort_by(.value) | reverse | .[0:10]' eco-gotests_analysis.json

# Operation coverage metrics for all suites
for file in *_analysis.json; do
  echo "=== $(basename $file _analysis.json) ==="
  jq '.coverage_metrics' "$file"
done
```

### **Identifying Test Suite Gaps**

**Coverage Gaps:**
```bash
# Find operations only in one suite
jq '.operation_gaps.unique_to_suite1 | length' all_suite_comparisons.json
jq '.operation_gaps.unique_to_suite2 | length' all_suite_comparisons.json

# Find resource gaps between all suites
jq '.resource_gaps.unique_to_suite1 | length' all_suite_comparisons.json
jq '.resource_gaps.unique_to_suite2 | length' all_suite_comparisons.json
```

**Purpose Gaps:**
```bash
# Find missing purpose categories
jq '.purpose_gaps' all_suite_comparisons.json

# Compare purpose distribution across all suites
for file in *_analysis.json; do
  echo "=== $(basename $file _analysis.json) ==="
  jq '.test_distribution.by_purpose' "$file"
done
```

**Environment Gaps:**
```bash
# Find environment differences
jq '.environment_gaps' all_suite_comparisons.json

# Compare environment coverage across all suites
for file in *_analysis.json; do
  echo "=== $(basename $file _analysis.json) ==="
  jq '.test_distribution.by_environment' "$file"
done
```

### **Using Analysis for Test Planning**

**Identify Missing Coverage:**
```bash
# Find operations with low coverage for a specific suite
jq '.operations | to_entries | map(select(.value < 5)) | sort_by(.value)' eco-gotests_analysis.json

# Find untested resource types across all suites
for file in *_analysis.json; do
  echo "=== $(basename $file _analysis.json) ==="
  jq '.resources | to_entries | map(select(.value == 1)) | length' "$file"
done
```

**Plan Cross-Language Testing:**
```bash
# Get recommendations for improving coverage
jq '.recommendations' all_suite_comparisons.json

# Find common operations to prioritize
jq '.operation_gaps.common | length' all_suite_comparisons.json
```

**Quality Assessment:**
```bash
# Check test diversity across all suites
for file in *_analysis.json; do
  echo "=== $(basename $file _analysis.json) ==="
  jq '.coverage_metrics | {test_type_diversity, purpose_diversity, avg_operations_per_test}' "$file"
done
```

### **Manual Suite Analysis**

You can also run the analyzer independently:

```bash
# Analyze individual suites
python match/analyze_test_suites.py --go go_specs.jsonl --output analysis/
python match/analyze_test_suites.py --py py_specs.jsonl --output analysis/

# Compare all suites
python match/analyze_test_suites.py --go go_specs.jsonl --py py_specs.jsonl --compare --output analysis/
```

### **Generating High-Level Reports**

Generate human-readable reports for test suite owners:

```bash
# Generate report for a specific suite
python match/generate_suite_report.py eco-gotests_analysis.json -o eco-gotests_report.md

# Generate reports for all suites
for file in *_analysis.json; do
    suite_name=$(basename "$file" _analysis.json)
    python match/generate_suite_report.py "$file" -o "${suite_name}_report.md"
done
```

**Report Contents:**
- **Executive Summary**: Test counts, types, purposes, and environments
- **Coverage Analysis**: Most tested resources and operations
- **Quality Metrics**: Test diversity and complexity analysis
- **Key Insights & Recommendations**: Automated analysis and suggestions
- **Validation Questions**: Specific questions for test suite owners to validate accuracy

## 📊 Purpose Categories

The system automatically detects and categorizes tests by purpose:

| Category | Description | Keywords |
|----------|-------------|----------|
| **POD_HEALTH** | Pod status validation, health checks | `pods`, `status`, `running`, `phase`, `health` |
| **POD_MANAGEMENT** | Pod creation, deletion, updates | `create`, `delete`, `update`, `pod`, `deployment` |
| **NETWORK_CONNECTIVITY** | Network reachability, routing tests | `curl`, `url`, `frr`, `routing`, `connectivity` |
| **NETWORK_POLICY** | Network policies, security | `policy`, `network`, `multinetwork`, `ingress` |
| **RESOURCE_VALIDATION** | Resource existence, counts | `count`, `exist`, `validation`, `verify`, `check` |
| **OPERATOR_MANAGEMENT** | Operator testing | `operator`, `subscription`, `csv`, `catalogsource` |
| **STORAGE_TESTING** | Storage, volumes | `storage`, `volume`, `pvc`, `pv`, `mount` |
| **SECURITY_TESTING** | Security contexts, RBAC | `security`, `rbac`, `scc`, `psa`, `permission` |

### **Purpose Compatibility Matrix**

| Purpose A | Compatible With Purpose B |
|-----------|---------------------------|
| POD_MANAGEMENT | POD_HEALTH, RESOURCE_VALIDATION |
| POD_HEALTH | POD_MANAGEMENT, RESOURCE_VALIDATION |
| NETWORK_POLICY | NETWORK_CONNECTIVITY, RESOURCE_VALIDATION |
| NETWORK_CONNECTIVITY | NETWORK_POLICY, RESOURCE_VALIDATION |
| All others | RESOURCE_VALIDATION |

## 🔬 Advanced Extraction Features

### **By(...) Step Extraction**

The tool automatically extracts and tracks operations within individual `By(...)` steps in Ginkgo tests, providing fine-grained visibility into test execution flow.

**How It Works:**

1. **Step Detection**: Automatically identifies `By(...)` calls in Go and Python Ginkgo tests
2. **Operation Mapping**: Maps Kubernetes operations to specific test steps
3. **Granular Tracking**: Creates detailed `by_steps` array in JSON output
4. **Cross-Reference**: Links operations in `actions` array to their originating steps via `by_step` field

**Example By(...) Step Extraction:**

```go
// Original Ginkgo Test
It("should create and verify pod", func() {
    By("creating the pod")
    pod := pods.Create(podSpec)
    
    By("waiting for pod to be ready")
    pods.WaitForCondition(pod, "Ready")
    
    By("verifying pod status")
    status := pods.Get(pod.Name)
    Expect(status.Phase).To(Equal("Running"))
})
```

**Extracted Spec with By(...) Steps:**

```json
{
  "test_id": "test.go:TestPodCreation",
  "actions": [
    {"gvk": "v1/Pod", "verb": "create", "by_step": "creating the pod"},
    {"gvk": "v1/Pod", "verb": "get", "by_step": "waiting for pod to be ready"},
    {"gvk": "v1/Pod", "verb": "get", "by_step": "verifying pod status"}
  ],
  "by_steps": [
    {
      "description": "creating the pod",
      "actions": [{"gvk": "v1/Pod", "verb": "create"}],
      "line": 15
    },
    {
      "description": "waiting for pod to be ready", 
      "actions": [{"gvk": "v1/Pod", "verb": "get"}],
      "line": 18
    },
    {
      "description": "verifying pod status",
      "actions": [{"gvk": "v1/Pod", "verb": "get"}], 
      "line": 21
    }
  ]
}
```

**Benefits:**

- **🔍 Fine-Grained Analysis**: Understand exactly which operations happen in which test steps
- **🐛 Debugging Support**: Pinpoint failures to specific test steps
- **📊 Better Similarity Matching**: Compare tests at the step level, not just file level
- **📝 Documentation**: Automatic extraction of test flow documentation from `By(...)` descriptions
- **🔄 Cross-Language Support**: Works with both Go and Python Ginkgo patterns

**Analysis Examples:**

```bash
# Find tests that create pods in their first step
jq -r 'select(.by_steps[0].actions[].verb == "create" and .by_steps[0].actions[].gvk == "v1/Pod") | .test_id' go_specs.jsonl

# Find tests with specific step descriptions
jq -r 'select(.by_steps[].description | contains("waiting")) | .test_id' go_specs.jsonl

# Compare step-level operations between tests
jq '.by_steps[].actions[]' test1.json test2.json
```

### **Individual It(...) Block Extraction**

For Ginkgo test files, the tool can extract each `It(...)` block as a separate test specification instead of consolidating at the file level.

**When It(...) Block Extraction is Used:**

- **Ginkgo Test Files**: Files containing `Describe()` and `It()` patterns
- **Multiple Test Cases**: When a single file contains multiple distinct test scenarios
- **Granular Analysis**: When file-level analysis is too coarse

**Example:**

```go
// Original Ginkgo File: pod_test.go
var _ = Describe("Pod Management", func() {
    It("should create pod successfully", func() {
        pod := pods.Create(podSpec)
        // ... test logic
    })
    
    It("should delete pod successfully", func() {
        pods.Delete(podName)
        // ... test logic  
    })
})
```

**Extracted as Separate Specs:**

```json
// First It(...) block
{
  "test_id": "pod_test.go:should create pod successfully",
  "actions": [{"gvk": "v1/Pod", "verb": "create"}],
  "purpose": "POD_MANAGEMENT"
}

// Second It(...) block  
{
  "test_id": "pod_test.go:should delete pod successfully", 
  "actions": [{"gvk": "v1/Pod", "verb": "delete"}],
  "purpose": "POD_MANAGEMENT"
}
```

**Benefits:**

- **🎯 Precise Matching**: Compare individual test scenarios instead of entire files
- **📊 Better Metrics**: More accurate similarity scores for specific test cases
- **🔍 Focused Analysis**: Identify exact duplicate test scenarios
- **📈 Improved Coverage**: Better understanding of what each test case actually does

## 🔍 Example Results

### **Before Purpose-Based Filtering:**

```
❌ BAD MATCH: ReachURLviaFRRroute (NETWORK_CONNECTIVITY) ↔ test_cu_pods_status (POD_HEALTH)
   - Shared: resource:v1/Pod (only resource-level similarity)
   - Problem: Different purposes, misleading match
```

### **After Purpose-Based Filtering:**

```
✅ GOOD MATCH: metallb-crds.go (POD_MANAGEMENT) ↔ test_du_pods_status (POD_HEALTH)
   - Shared: exact:v1/Pod:get;resource:v1/Pod;category:v1/Pod:read
   - Compatible purposes, meaningful similarity
```

## 🔍 Test Spec Inspection Tool

### **Overview**

The inspection tool provides complete visibility into the test specification extraction pipeline, allowing you to see exactly how source code is transformed into vector database representations. This is essential for debugging extraction issues, understanding similarity matching, and validating the quality of test specifications.

### **Usage**

```bash
# Inspect a Python test file
python inspect_test_specs.py \
    --file ~/external/eco-pytests/src/eco_pytests/du/deployment/test_sriov.py \
    --output inspection_output/test_sriov/

# Inspect a Go test file  
python inspect_test_specs.py \
    --file ~/external/eco-gotests/tests/cnf/ran/ptp/ptp_suite_test.go \
    --output inspection_output/ptp_suite/

# Inspect any test file (auto-detects language)
python inspect_test_specs.py \
    --file /path/to/any/test_file.py \
    --output inspection_output/
```

### **Output Structure**

The inspection tool generates a complete analysis directory:

```
inspection_output/test_sriov/
├── specs.jsonl                    # Raw extracted specs (JSON format)
├── text_representations/          # Text sent to vector database
│   ├── test_sriov_interfaces_exist.txt
│   ├── test_sriov_operator_ready.txt
│   └── test_sriov_pods_status.txt
├── index.json                     # Metadata (test_id → files mapping)
└── summary.md                     # Human-readable summary
```

### **Vector Database Pipeline**

The inspection tool shows the complete pipeline from source code to vector database:

```
Source Code → JSON Specs → Text Representation → Embeddings → FAISS Index
     ↓              ↓              ↓                ↓           ↓
  test.py    →  specs.jsonl  →  *.txt files  →  vectors  →  similarity
```

**Why YAML-like Text Files (not JSON)?**
- The vector database stores **embeddings** (numerical vectors), not JSON
- The `spec_to_text()` function converts JSON specs into **YAML-like structured text**
- SentenceTransformer encodes this structured text into embeddings
- FAISS stores the numerical embeddings for similarity search
- The `.txt` files show exactly what **structured text** gets encoded
- **YAML-like format** preserves hierarchy and relationships better than JSON for embeddings

### **Key Files Explained**

#### **`specs.jsonl`** - Raw Extracted Specifications
Contains the complete JSON specifications for each test, exactly as generated by the extractors:

```json
{
  "test_id": "deployment/test_sriov.py:test_sriov_interfaces_exist",
  "test_type": "unit",
  "purpose": "SRIOV_TESTING",
  "tech": ["SR-IOV"],
  "environment": ["multi_node"],
  "actions": [
    {"gvk": "v1/Pod", "verb": "get", "by_step": "check sriov interfaces"}
  ],
  "expectations": [
    {"target": "resource_count", "condition": "len(sriov_interfaces) > 0"}
  ],
  "by_steps": [
    {
      "description": "check sriov interfaces",
      "actions": [{"gvk": "v1/Pod", "verb": "get"}],
      "line": 45
    }
  ]
}
```

#### **`text_representations/*.txt`** - Vector Database Input
Shows the exact **YAML-like structured text** sent to the vector database for similarity matching. This is the **actual input** to the embedding model (SentenceTransformer) that generates the vectors stored in FAISS:

```
TEST_ID: deployment/test_sriov.py:test_sriov_interfaces_exist
TEST_TYPE: unit
TECHNOLOGY: SR-IOV
PURPOSE: SRIOV_TESTING
ENVIRONMENT: multi_node
OPERATIONS:
  - v1/Pod:get
EXPECTATIONS:
  - resource_count=len(sriov_interfaces) > 0
STEPS:
  - check sriov interfaces
    - v1/Pod:get
```

#### **`index.json`** - Metadata Index
Maps test IDs to their corresponding files and provides metadata:

```json
{
  "source_file": "/path/to/test_sriov.py",
  "language": "python",
  "total_tests": 3,
  "tests": [
    {
      "test_id": "deployment/test_sriov.py:test_sriov_interfaces_exist",
      "text_file": "test_sriov_interfaces_exist.txt",
      "purpose": "SRIOV_TESTING",
      "tech": ["SR-IOV"],
      "operations": ["v1/Pod:get"]
    }
  ]
}
```

#### **`summary.md`** - Human-Readable Report
Provides a comprehensive overview of the extracted tests:

```markdown
# Test Spec Inspection Report

**Source File**: /path/to/test_sriov.py
**Language**: Python
**Total Tests Found**: 3

## Tests Extracted

### 1. test_sriov_interfaces_exist
- **Purpose**: SRIOV_TESTING
- **Technology**: SR-IOV
- **Operations**: v1/Pod:get
- **Text File**: text_representations/test_sriov_interfaces_exist.txt

### 2. test_sriov_operator_ready
- **Purpose**: OPERATOR_MANAGEMENT
- **Technology**: SR-IOV
- **Operations**: operators.coreos.com/v1alpha1/Subscription:get
- **Text File**: text_representations/test_sriov_operator_ready.txt
```

### **Use Cases**

#### **1. Debug Extraction Issues**
```bash
# Check if operations are correctly detected
python inspect_test_specs.py --file problematic_test.py --output debug/
cat debug/specs.jsonl | jq '.actions'
```

#### **2. Understand Similarity Matching**
```bash
# Compare text representations of similar tests
python inspect_test_specs.py --file test_a.py --output analysis/
python inspect_test_specs.py --file test_b.py --output analysis/
diff analysis/test_a/text_representations/ analysis/test_b/text_representations/
```

#### **3. Validate Technology Detection**
```bash
# Check if SR-IOV technology is properly detected
python inspect_test_specs.py --file sriov_test.py --output validation/
grep -r "SR-IOV" validation/text_representations/
```

#### **4. Analyze Step Extraction**
```bash
# See how By(...) steps are extracted and represented
python inspect_test_specs.py --file ginkgo_test.go --output steps/
cat steps/specs.jsonl | jq '.by_steps'
```

#### **5. Cross-Language Comparison**
```bash
# Compare Go vs Python extraction for similar tests
python inspect_test_specs.py --file go_test.go --output comparison/
python inspect_test_specs.py --file py_test.py --output comparison/
# Compare the text_representations/ directories
```

### **Advanced Analysis**

#### **Extract Top Matches for Analysis**
```bash
# Get the top 5 matches from similarity results
head -6 results/test_report.csv | tail -5 > top_matches.csv

# Extract each match for detailed analysis
while IFS=',' read -r idx_a idx_b a_test b_test; do
  echo "Analyzing match: $a_test ↔ $b_test"
  # Extract test A
  python inspect_test_specs.py --file "$(echo $a_test | cut -d: -f1)" --output "match_${idx_a}_${idx_b}_a/"
  # Extract test B  
  python inspect_test_specs.py --file "$(echo $b_test | cut -d: -f1)" --output "match_${idx_a}_${idx_b}_b/"
done < top_matches.csv
```

#### **Batch Analysis**
```bash
# Analyze multiple test files at once
for test_file in tests/*.py; do
  output_dir="analysis/$(basename $test_file .py)"
  python inspect_test_specs.py --file "$test_file" --output "$output_dir"
done
```

### **Benefits**

- **🔍 Full Pipeline Visibility**: See complete transformation from source code → JSON → text representation → embeddings
- **🐛 Debug Extraction Issues**: Verify that operations are correctly detected and categorized
- **🎯 Debug Similarity Issues**: Compare text representations to understand why tests match or don't match
- **✅ Technology Verification**: Confirm technology detection is working correctly
- **📊 Step Analysis**: See how By(...) steps are extracted and represented in the final text
- **🔄 Cross-Language Comparison**: Compare Go vs Python extraction for similar test patterns
- **📈 Quality Assessment**: Validate the quality and completeness of test specifications
- **🧠 Embedding Understanding**: See exactly what text gets encoded into the vector database

### **Troubleshooting**

#### **Common Issues**

1. **"No module named 'sentence_transformers'"**
   - The tool falls back to a simplified text representation
   - This is normal and doesn't affect the core functionality

2. **Empty specs.jsonl**
   - Check if the test file follows expected patterns (Ginkgo for Go, pytest for Python)
   - Verify the file path is correct and accessible

3. **Missing operations in text representation**
   - Check if the test uses supported patterns (eco-goinfra, openshift library, etc.)
   - Verify helper functions are properly detected

#### **Debug Mode**
```bash
# Enable verbose output for debugging
python inspect_test_specs.py --file test.py --output debug/ 2>&1 | tee debug.log
```

## 🛠 Manual Usage

### **1. Go Extractor**

```bash
cd go-extractor
go build -o kubespec-go
./kubespec-go -root /path/to/go/repo > ../go_specs.jsonl
```

**Features:**

- Detects Ginkgo/Gomega patterns (`Describe`, `It`, `BeforeEach`, etc.)
- Extracts eco-goinfra operations (`pods.List()`, `deployments.Create()`, etc.)
- Maps CLI commands to API operations (`kubectl get pods` → `v1/Pod:get`)
- Detects helper functions across files
- Purpose detection from test content

### **2. Python Extractor**

```bash
cd py-extractor
python extract_kubespec.py --root /path/to/python/tests > ../py_specs.jsonl
```

**Features:**
- Detects pytest patterns (`test_*` functions)
- Extracts openshift library calls (`oc.selector()`, `get_resource()`)
- Maps subprocess calls to API operations
- Analyzes docstrings and test content for purpose detection

### **3. Matching Engine**

```bash
cd match
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_index_and_match.py --go ../go_specs.jsonl --py ../py_specs.jsonl --out report.csv --cov coverage_matrix.csv
```

**Features:**

- Semantic embeddings using SentenceTransformers
- FAISS-based similarity search
- Purpose-based filtering and scoring
- Multi-level similarity detection
- Validation and quality metrics

### **4. Optional LLM Re-ranking**

```bash
export LLM_API_KEY=your_api_key
export LLM_MODEL=gpt-4o-mini
python build_index_and_match.py --go go_specs.jsonl --py py_specs.jsonl --out report.csv --llm
```

## 📈 Performance Metrics

### **Filtering Impact:**

- **Before**: 1370 total matches (many false positives)
- **After**: 657 total matches (52% reduction)
- **Quality**: Only compatible purpose matches remain

### **Validation Rates:**

- **Purpose Compatibility**: 50%+ of high-similarity matches
- **Operation Validation**: Detects shared operations in meaningful matches
- **False Positive Reduction**: 52% fewer misleading matches

## 🏗 Architecture

### **Extraction Pipeline**

```
Go Tests → AST Analysis → KubeSpec → Purpose Detection
Python Tests → AST Analysis → KubeSpec → Purpose Detection
```

### **Matching Pipeline**

```
KubeSpecs → Embeddings → Similarity Search → Purpose Filtering → Scoring → Results
```

### **Key Components**

- **`go-extractor/main.go`**: Go AST parser with Ginkgo/eco-goinfra support
- **`py-extractor/extract_kubespec.py`**: Python AST parser with pytest/openshift support
- **`match/build_index_and_match.py`**: Semantic matching with purpose-based filtering
- **`extract-and-match.sh`**: Automated pipeline orchestration

## 🔧 Configuration

### **Purpose Detection Patterns**

You can customize purpose detection by modifying the patterns in:
- **Go**: `go-extractor/main.go` → `purposePatterns` map
- **Python**: `py-extractor/extract_kubespec.py` → `PURPOSE_PATTERNS` dict

### **Compatibility Matrix**

Adjust purpose compatibility in `match/build_index_and_match.py`:
```python
PURPOSE_COMPATIBILITY = {
    'POD_MANAGEMENT': ['POD_HEALTH', 'RESOURCE_VALIDATION'],
    # ... add your own compatibility rules
}
```

### **Scoring Weights**

Customize scoring boosts in the matching engine:

```python
# Same purpose boost
purpose_boost = 0.20  # +20% for same purpose

# Compatible purpose boost  
purpose_boost = 0.10  # +10% for compatible purposes

# Incompatible purpose penalty
purpose_boost = -0.30  # -30% for incompatible purposes
```

## 📋 Output Format

### **Test Report CSV**

```csv
idx_a,idx_b,base_score,blended_score,a_test,b_test,shared_signals
0,1,0.85,1.0,eco-gotests/test.go:TestFunction,eco-pytests/test.py:test_function,exact:v1/Pod:get;resource:v1/Pod
```

### **KubeSpec JSONL**

```json
{
  "test_id": "repo/test.go:TestFunction",
  "test_type": "integration",
  "dependencies": ["psa:pod-security.kubernetes.io/enforce=restricted"],
  "environment": ["multi_node"],
  "purpose": "POD_HEALTH",
  "actions": [{"gvk": "v1/Pod", "verb": "get"}],
  "expectations": [{"target": "resource_status", "condition": "pod.status.phase == 'Running'"}],
  "openshift_specific": ["route.openshift.io/v1/Route"],
  "concurrency": [],
  "artifacts": ["testdata/pod.yaml"],
  "tech": ["SR-IOV", "GPU"]
}
```

## 🤝 Contributing

1. **Adding New Purpose Categories**: Update `purposePatterns` in both extractors
2. **Improving Detection**: Enhance keyword patterns and operation inference
3. **New Test Frameworks**: Extend AST visitors for additional frameworks
4. **Scoring Improvements**: Adjust compatibility matrix and boost weights

## 📚 Dependencies

- **Go**: Standard library (no external dependencies)
- **Python**: `sentence-transformers`, `faiss-cpu`, `pandas`, `numpy`
- **Optional**: `openai` for LLM re-ranking

## 🐛 Troubleshooting

### **Common Issues**

1. **Empty Matches**: Check if test files follow expected patterns (Ginkgo for Go, pytest for Python)
2. **Low Purpose Detection**: Verify test names and content contain recognizable keywords
3. **High False Positives**: Adjust compatibility matrix or scoring weights
4. **Missing Operations**: Ensure helper functions are properly detected in cross-file calls

### **Debug Mode**

Enable verbose output:

```bash
./extract-and-match.sh -g /path/to/go -p /path/to/py -o debug_output 2>&1 | tee debug.log
```

## 🔧 Advanced Debugging Tools

The project includes several specialized debugging tools for advanced analysis and troubleshooting:

### **Available Debug Tools**

- **`debug_filtering.py`**: Debug purpose-based filtering logic with specific test matches
- **`debug_index_mapping.py`**: Analyze index mapping and spec loading issues  
- **`debug_report_generation.py`**: Debug similarity report generation pipeline
- **`debug_shared_signals.py`**: Analyze shared signal detection between tests
- **`debug_specific_indices.py`**: Debug specific test indices and their matches
- **`verify_csv_match.py`**: Verify CSV output matches expected format and content

### **Using Debug Tools**

**Debug Filtering Issues:**
```bash
# Debug why certain tests are being filtered out
python debug_filtering.py

# This will show:
# - Purpose compatibility analysis
# - Filtering decisions for specific matches
# - Scoring adjustments and boosts
```

**Debug Index Mapping:**
```bash  
# Debug spec loading and index mapping
python debug_index_mapping.py

# This will show:
# - How specs are loaded from JSONL files
# - Index to spec mapping
# - Any loading errors or inconsistencies
```

**Debug Shared Signals:**
```bash
# Analyze shared signal detection between specific tests
python debug_shared_signals.py

# This will show:
# - Operation overlap analysis
# - Signal type classification (exact, resource, category, verb_group)
# - Signal strength calculations
```

**Verify Output Format:**
```bash
# Verify CSV outputs match expected format
python verify_csv_match.py

# This will check:
# - CSV column structure
# - Data type consistency
# - Required field presence
```

### **Debug Tool Benefits**

- **🔍 Deep Analysis**: Understand exactly how similarity matching works
- **🐛 Issue Isolation**: Pinpoint specific problems in the pipeline
- **⚙️ Parameter Tuning**: Optimize scoring weights and compatibility rules
- **✅ Quality Assurance**: Verify output correctness and consistency
- **📊 Performance Analysis**: Identify bottlenecks in matching pipeline

### **Advanced Debugging Workflow**

1. **Run Full Pipeline**: Generate initial results with `extract-and-match.sh`
2. **Identify Issues**: Look for unexpected matches or missing similarities
3. **Use Specific Debug Tool**: Run relevant debug tool for the issue type
4. **Analyze Output**: Review debug output to understand root cause
5. **Tune Parameters**: Adjust weights, compatibility rules, or patterns
6. **Verify Fix**: Re-run pipeline and debug tools to confirm resolution

**Example Debug Session:**
```bash
# 1. Run pipeline
./extract-and-match.sh -g /path/to/go -p /path/to/py

# 2. Found unexpected high-similarity match, debug filtering
python debug_filtering.py > filtering_debug.log

# 3. Check if indices are correct
python debug_index_mapping.py > mapping_debug.log  

# 4. Analyze shared signals for the problematic match
python debug_shared_signals.py > signals_debug.log

# 5. Verify final output format
python verify_csv_match.py
```
