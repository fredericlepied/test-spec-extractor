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
# Single Go repository
./extract-spec-md.sh -g /path/to/eco-gotests

# Multiple Go repositories
./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -g /path/to/cnf-gotests

# With Python repository
./extract-spec-md.sh -g /path/to/eco-gotests -p /path/to/eco-pytests

# Custom output directory
./extract-spec-md.sh -g /path/to/go-tests -o my_analysis
```

### **What You Get**

The pipeline generates:
- `markdown_similarity_results.csv`: Comprehensive similarity matches (Go↔Go, Python↔Python, Go↔Python) with scores and shared operations
- `similarity_analysis.md`: Comprehensive similarity analysis report with match type distribution, executive summary, and strategic recommendations
- `go_specs_per_it.jsonl` / `py_specs_per_it.jsonl` / `all_specs_per_it.jsonl`: Per-test extracted specifications
- `markdown/`: Directory containing markdown documentation for each test file
- Repository symlinks for easy navigation to source files

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

### **Understanding the Test Report (`markdown_similarity_results.csv`)**

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
sort -t',' -k4 -nr markdown_similarity_results.csv | head -10

# Find matches with exact operations
grep "exact:" markdown_similarity_results.csv

# Count matches by shared signal type
cut -d',' -f7 markdown_similarity_results.csv | tr ';' '\n' | cut -d':' -f1 | sort | uniq -c

# Filter by specific test
grep "test_function_name" markdown_similarity_results.csv

# Analyze match types
cut -d',' -f12 markdown_similarity_results.csv | sort | uniq -c

# Find intra-language duplicates (same language)
grep -E "(go->go|py->py)" markdown_similarity_results.csv

# Find cross-language opportunities
grep -E "(py->go|go->py)" markdown_similarity_results.csv
```

### **Understanding the Similarity Analysis Report (`similarity_analysis.md`)**

The similarity analysis report provides a comprehensive analysis of test relationships and includes:

- **Executive Summary**: Overview of matches, quality indicators, and duplicate ratios
- **Match Type Analysis**: Distribution of Go↔Go, Python↔Python, and cross-language matches
- **Score Distribution**: Visual analysis of similarity score patterns
- **Shared Signals Analysis**: Breakdown of exact operations, resource matches, and category matches
- **Potential Duplicates**: High-similarity tests (≥0.95) with consolidation recommendations
- **Complementary Tests**: Medium-similarity tests (0.6-0.8) with different purposes
- **Top Similarity Matches**: Most similar test pairs with detailed analysis
- **Strategic Recommendations**: Actionable insights for test optimization

**How to Use:**

```bash
# View the full report
cat similarity_analysis.md

# Find duplicate recommendations
grep -A 5 "Potential Duplicates" similarity_analysis.md

# Check match type distribution
grep -A 10 "Match Type Distribution" similarity_analysis.md

# Review strategic recommendations
grep -A 20 "Strategic Recommendations" similarity_analysis.md
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
   grep "exact:" markdown_similarity_results.csv | head -5

   # Check purpose compatibility
   awk -F',' 'NR>1 {print $5, $6}' markdown_similarity_results.csv | head -10
   ```

2. **Review Similarity Analysis:**

   ```bash
   # View comprehensive analysis report
   cat similarity_analysis.md

   # Check for high-similarity duplicates
   grep -A 10 "Potential Duplicates" similarity_analysis.md
   ```

3. **Validate Similarity:**
   ```bash
   # Get details of top matches
   head -5 markdown_similarity_results.csv | while IFS=',' read -r idx_a idx_b base_score blended_score a_test b_test shared_signals; do
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
awk -F',' 'NR>1 {print $5, $6, $4, $7}' markdown_similarity_results.csv | head -10
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

## 📊 Understanding Markdown Documentation

The `markdown/` directory contains auto-generated documentation for each test file, organized by repository:

```
spec-md/
├── eco-gotests/          # Go test repository
│   ├── tests/
│   │   └── cnf/
│   │       └── ran/
│   │           └── ptp/
│   │               └── ptp_suite_test.md
├── eco-pytests/          # Python test repository
│   └── src/
│       └── eco_pytests/
│           └── du/
│               └── deployment/
│                   └── test_sriov.md
└── symlinks/             # Easy access to source repos
    ├── eco-gotests -> /path/to/eco-gotests
    └── eco-pytests -> /path/to/eco-pytests
```

Each markdown file contains:
- **Container hierarchy** (Describe/Context/When blocks)
- **Test cases** (It blocks) with descriptions
- **Test steps** (By(...) calls) with line numbers
- **Setup/Teardown** (BeforeEach/AfterEach)
- **Parametrized tests** (Entry blocks)

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

### **3. Similarity Analysis**

```bash
# Run similarity analysis on markdown specs
python match/markdown-similarity.py \
  --go-jsonl go_specs_per_it.jsonl \
  --py-jsonl py_specs_per_it.jsonl \
  --output similarity_results.csv \
  --report similarity_analysis.md

# Or use the automated pipeline
./extract-spec-md.sh -g /path/to/go-tests -p /path/to/py-tests
```

**Features:**

- Semantic embeddings using SentenceTransformers (all-mpnet-base-v2)
- FAISS-based similarity search with BDD-aware hierarchical context
- Purpose-based filtering and scoring
- Match type analysis (Go↔Go, Python↔Python, Python↔Go)
- Generic assertion filtering for better cross-repository matching
- Comprehensive markdown reports with strategic recommendations

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
- **`go-extractor/spec-extractor/`**: Go spec markdown extractor (generates markdown + JSONL)
- **`py-extractor/extract_kubespec.py`**: Python AST parser with pytest/openshift support
- **`py-extractor/spec_extractor/`**: Python spec markdown extractor (generates markdown + JSONL)
- **`match/markdown-similarity.py`**: Advanced semantic matching with BDD-aware context and purpose-based filtering
- **`extract-spec-md.sh`**: Automated pipeline for markdown generation and similarity analysis

## 📝 Markdown Spec Extraction

### **Overview**

The project includes spec extractors that generate human-readable markdown documentation and JSONL files from test files. These are compatible with the similarity matching system and can be used for cross-language test analysis.

### **Go Spec Extractor**

```bash
# Generate markdown specs from Go test repositories
./extract-spec-md.sh -g /path/to/eco-gotests

# Multiple repositories
./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/openshift-tests
```

**Output:**
- `spec-md/{repo_name}/` - Markdown files for each test file
- `go_specs_per_it.jsonl` - Per-test JSONL records for similarity matching

**Features:**
- Extracts Ginkgo BDD structure (Describe, Context, When, It blocks)
- Extracts `By(...)` steps as test actions
- Handles BeforeEach/AfterEach setup/teardown
- Supports parametrized tests (Entry)
- Generates markdown matching exact format for compatibility

### **Python Spec Extractor**

```bash
# Generate markdown specs from Python test repositories
./extract-spec-md.sh -p /path/to/eco-pytests

# Both Go and Python for cross-language similarity
./extract-spec-md.sh -g /path/to/eco-gotests -p /path/to/eco-pytests
```

**Output:**
- `spec-md/{repo_name}/` - Markdown files for each test file
- `py_specs_per_it.jsonl` - Per-test JSONL records for similarity matching
- `all_specs_per_it.jsonl` - Combined Go and Python specs (when both are provided)

**Features:**
- Extracts pytest test functions (`test_*`)
- Extracts test classes as containers
- Handles pytest fixtures as setup/teardown
- Extracts function body operations as steps
- Supports parametrized tests
- Generates markdown matching Go extractor format exactly

**Structure Compatibility:**
- Uses same Container/TestCase/TestStep data structures as Go extractor
- Generates JSONL in same PerItRecord format
- Compatible with `match/markdown-similarity.py` for semantic analysis
- Enables cross-language similarity matching (Go↔Python)

### **Similarity Analysis with Markdown Specs**

The `extract-spec-md.sh` script automatically runs similarity analysis:

```bash
./extract-spec-md.sh -g /path/to/eco-gotests -p /path/to/eco-pytests
```

This generates:
- Markdown specs for all repositories
- JSONL files for similarity matching
- `markdown_similarity_results.csv` - Detailed similarity analysis
- `similarity_analysis.md` - Human-readable similarity report

The similarity analysis uses semantic embeddings to find similar tests across languages, enabling discovery of:
- Duplicate tests in different languages
- Similar test patterns that could be consolidated
- Cross-language test opportunities

## 🔧 Configuration

### **Purpose Detection Patterns**

You can customize purpose detection by modifying the patterns in:
- **Go**: `go-extractor/main.go` → `purposePatterns` map
- **Python**: `py-extractor/extract_kubespec.py` → `PURPOSE_PATTERNS` dict

### **Compatibility Matrix**

Adjust purpose compatibility in `match/markdown-similarity.py`:
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
./extract-spec-md.sh -g /path/to/go -p /path/to/py -o debug_output 2>&1 | tee debug.log
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

1. **Run Full Pipeline**: Generate initial results with `extract-spec-md.sh`
2. **Identify Issues**: Look for unexpected matches or missing similarities
3. **Use Specific Debug Tool**: Run relevant debug tool for the issue type
4. **Analyze Output**: Review debug output to understand root cause
5. **Tune Parameters**: Adjust weights, compatibility rules, or patterns
6. **Verify Fix**: Re-run pipeline and debug tools to confirm resolution

**Example Debug Session:**
```bash
# 1. Run pipeline
./extract-spec-md.sh -g /path/to/go -p /path/to/py

# 2. Found unexpected high-similarity match, debug filtering
python debug_filtering.py > filtering_debug.log

# 3. Check if indices are correct
python debug_index_mapping.py > mapping_debug.log

# 4. Analyze shared signals for the problematic match
python debug_shared_signals.py > signals_debug.log

# 5. Verify final output format
python verify_csv_match.py
```
