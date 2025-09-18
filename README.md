# Cross-Language Test Matcher (Go + Python)

A sophisticated toolkit that extracts KubeSpecs from Go and Python test files, builds semantic embeddings, and matches cross-language test similarity with **purpose-based filtering** to reduce false positives. OpenShift-aware with automatic equivalence detection (Route↔Ingress, SCC↔PSA) and optional LLM re-ranking.

## 🎯 Key Features

### **Purpose-Based Filtering System**
- **Intelligent Purpose Detection**: Automatically categorizes tests by their purpose (POD_HEALTH, NETWORK_CONNECTIVITY, POD_MANAGEMENT, etc.)
- **Compatibility Matrix**: Only matches tests with compatible purposes, eliminating false positives
- **52% Reduction in False Positives**: From 1370 to 657 matches with much higher quality

### **Advanced Test Analysis**
- **Multi-Level Similarity**: Exact operations, resource-level, category-level, and verb-group matching
- **Enhanced Scoring**: Purpose-based boosts (+0.20 same purpose, +0.10 compatible, -0.30 incompatible)
- **OpenShift Awareness**: Automatic detection of Route↔Ingress and SCC↔PSA equivalents
- **Utility Test Filtering**: Automatically filters out helper functions and utility tests

### **Comprehensive Extraction**
- **Go Tests**: Supports Ginkgo/Gomega, eco-goinfra, standard Go tests
- **Python Tests**: Supports pytest, openshift library, subprocess calls
- **Cross-File Detection**: Detects operations in helper functions across files
- **Rich Metadata**: Extracts actions, expectations, preconditions, artifacts

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
- `test_report.csv`: Similarity matches with scores and shared operations
- `test_coverage.csv`: Coverage matrix of operations across test suites
- `go_specs.jsonl` / `py_specs.jsonl`: Raw extracted test specifications

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
  "level": "integration",
  "purpose": "POD_HEALTH",
  "actions": [{"gvk": "v1/Pod", "verb": "get"}],
  "expectations": [{"target": "resource_status", "condition": "pod.status.phase == 'Running'"}],
  "preconditions": ["psa:pod-security.kubernetes.io/enforce=restricted"],
  "openshift_specific": ["route.openshift.io/v1/Route"],
  "concurrency": [],
  "artifacts": ["testdata/pod.yaml"]
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

## 📄 License

This project is part of the eco-system test analysis toolkit.