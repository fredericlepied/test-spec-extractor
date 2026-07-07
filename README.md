# Test Similarity Analyzer (Go + Python)

Finds duplicate and similar tests across Go and Python test repositories using semantic analysis with FAISS embeddings and K8s resource detection.

## Key Features

- **Cross-repository duplicate detection** - Finds identical tests across different repositories
- **Language-agnostic analysis** - Go↔Go, Python↔Python, and Python↔Go similarity matching
- **K8s resource detection** - Identifies which Kubernetes resources each test exercises
- **BDD-aware context** - Uses Describe/Context/It hierarchy for better matching
- **Quality metrics** - Per-repo coverage reporting for K8s resource detection

## Quick Start

```bash
# Single repository
./extract-spec-md.sh -g /path/to/go-tests

# Multiple repositories for cross-repo duplicate detection
./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/cnf-gotests -g /path/to/openshift-tests -p /path/to/eco-pytests

# Custom output directory
./extract-spec-md.sh -g /path/to/tests -o my-analysis
```

## What You Get

The pipeline generates:

- **similarity_analysis.md** - Executive summary with duplicates, recommendations, and match type analysis
- **markdown_similarity_results.csv** - Detailed similarity matches with scores and file paths
- **markdown/** - Auto-generated test documentation organized by repository
- **go_specs_per_it.jsonl** / **py_specs_per_it.jsonl** - Extracted test specifications with K8s resources

## K8s Resource Detection

Each test file is annotated with the K8s resources it exercises. Detection works differently per language:

**Go tests** (eco-gotests, cnf-gotests, openshift-tests):
- eco-goinfra package imports (`pkg/pod` → Pod, `pkg/deployment` → Deployment)
- `k8s.io/api` type references (`corev1.Pod`, `appsv1.Deployment`)
- `oc.Run("get").Args("pod")` CLI patterns (openshift-tests)
- Transitive scanning through helper packages

**Python tests** (eco-pytests):
- `oc.selector("pod")`, `get_resource("node")` call arguments
- `create_api_object({"kind": "PVC"})` dict literals
- Well-known helper functions (`get_pods_list`, `await_all_nodes_ready`, etc.)
- Full AST walking including nested try/except blocks

Resource names are normalized across languages for cross-language matching (e.g., `csv` → `OLM`, `baremetalhosts` → `BMC`).

The pipeline reports per-repo coverage:
```
eco-gotests (Go):            213/213 files with k8s resources (100%)
openshift-tests (Go):        152/157 files with k8s resources (96%)
openshift-tests-private (Go): 186/247 files with k8s resources (75%)
eco-pytests (Python):          33/56 files with k8s resources (58%)
```

## Understanding Results

### Similarity Scores

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.95-1.00 | Near-identical / duplicate tests | Consolidate immediately |
| 0.85-0.94 | Same test pattern, minor variations | Review for consolidation |
| 0.75-0.84 | Related tests in same domain | Understand coverage overlap |

### Example Output

```
Score: 0.933 [cnf-gotests <-> eco-gotests]
  Q: should delete a worker node from the cluster
  M: should delete a worker node from the cluster
  ztp_argocd_node_delete.go <-> ztp-argocd-node-delete.go
```

### JSONL Test Spec Format

```json
{
  "desc": "Check pods state",
  "test_id": "54548",
  "line_number": 88,
  "k8s_resources": ["Deployment", "NFD", "Namespace", "Node", "Pod"],
  "steps": ["Check that pods are in running state"],
  "validations": ["operation succeeds without error"],
  "file_path": "/path/to/features-test.go"
}
```

## Web Explorer

The project includes an interactive web UI to browse test specs, similarity matches, and K8s resource coverage.

```bash
# Build (requires Node.js)
cd web
npm install
npm run build

# Serve the result
cd dist
python3 -m http.server 8080
# Open http://localhost:8080
```

The `dist/` directory is self-contained (HTML + JS + data) and can be copied anywhere. Five views: Dashboard, Similarity, Clusters, Graph, and Catalog.

See [web/README.md](web/README.md) for full feature documentation.

The web data is regenerated automatically when running `extract-spec-md.sh` or `npm run build`.

## Architecture

```
extract-spec-md.sh
    ├─→ Go Extractor (AST + import scanning)  → go_specs_per_it.jsonl + markdown/
    ├─→ Python Extractor (AST + call analysis) → py_specs_per_it.jsonl + markdown/
    ├─→ K8s Resource Quality Check             → per-repo coverage stats
    ├─→ Similarity Analysis (FAISS)            → similarity_analysis.md + CSV + JSON
    └─→ Web UI Data Prep                       → web/public/data/*.json
```

**Components:**

- **go-extractor/** - Ginkgo/Gomega pattern extraction, transitive K8s resource detection via imports and `oc.Run` patterns
- **py-extractor/** - pytest pattern extraction, K8s resource detection via `oc.selector`/helper function analysis
- **match/markdown-similarity.py** - FAISS-based semantic matching with BDD context and K8s resource scoring
- **web/** - React + Vite interactive explorer (dashboard, similarity browser, test catalog)

## CLI Options

```
./extract-spec-md.sh [OPTIONS]

    -g, --go-root PATH      Path to Go test repository (repeatable)
    -p, --py-root PATH      Path to Python test repository (repeatable)
    -o, --output-dir DIR    Output directory (default: spec-md)
    -v, --verbose           Verbose output
    -h, --help              Show help
```

## Troubleshooting

### Missing Dependencies

```bash
# Python dependencies (auto-installed on first run)
cd match && pip install -r requirements.txt

# Go (requires 1.20+)
go version
```

### Low K8s Resource Coverage

If the quality check shows LOW COVERAGE (<50%) for a repo, the resource detection patterns may not cover that repo's coding style. Check how the repo accesses K8s resources and update the detection patterns.

### Empty Results

- Verify test files follow expected patterns (Ginkgo for Go, pytest for Python)
- Check repository paths are correct and accessible
- Use `-v` for verbose output to see extraction details

## Dependencies

- **Go**: Standard library only (no external dependencies)
- **Python**: sentence-transformers, faiss-cpu, pandas, numpy
- See `match/requirements.txt` for complete list
