# Test Similarity Analyzer (Go + Python)

Finds duplicate and similar tests across Go and Python test repositories using semantic analysis with FAISS embeddings.

## 🎯 Key Features

- **Cross-repository duplicate detection** - Finds identical tests across different repositories
- **Language-agnostic analysis** - Go↔Go, Python↔Python, and Python↔Go similarity matching
- **BDD-aware context** - Uses Describe/Context/It hierarchy for better matching
- **Purpose-based filtering** - Reduces false positives by 78% using test purpose compatibility
- **OpenShift-aware** - Detects Route↔Ingress and SCC↔PSA equivalents

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r match/requirements.txt
```

### Run Analysis

```bash
# Single repository
./extract-spec-md.sh -g /path/to/go-tests

# Multiple repositories for cross-repo duplicate detection
./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/cnf-gotests -p /path/to/eco-pytests

# Custom output directory
./extract-spec-md.sh -g /path/to/tests -o my-analysis
```

### View Results

```bash
# Read the human-readable report
cat spec-md/similarity_analysis.md

# Or explore detailed matches
less spec-md/markdown_similarity_results.csv
```

## 📊 What You Get

The pipeline generates:

- **similarity_analysis.md** - Executive summary with duplicates, recommendations, and match type analysis
- **markdown_similarity_results.csv** - Detailed similarity matches with scores and file paths
- **markdown/** - Auto-generated test documentation organized by repository
- **go_specs_per_it.jsonl** / **py_specs_per_it.jsonl** - Extracted test specifications

## 🔍 Understanding Results

### Example Duplicate Found

```
Match: test_21 ↔ test_104 (similarity: 1.000)
  Description: "Verify namespaces are created for ClusterDeployments"
  File 1: eco-pytests/du/deployment/test_deploy_cloudran_site.py
  File 2: eco-pytests/cu/deployment/test_deploy_cloudran_site.py
  → Consolidation opportunity: Same test in DU and CU configurations
```

### Similarity Analysis Report Structure

The `similarity_analysis.md` report includes:

- **Executive Summary** - Total matches, quality indicators, duplicate counts
- **Match Type Analysis** - Distribution of Go↔Go, Python↔Python, Python↔Go matches
- **Potential Duplicates** - High-similarity matches (>0.9) with consolidation recommendations
- **Cross-Language Matches** - Opportunities for test pattern sharing between languages
- **Strategic Recommendations** - Actionable insights for test optimization

### CSV Columns Explained

```csv
query_test_id,query_description,query_file,matched_test_id,matched_description,matched_file,semantic_similarity,context_similarity,...
```

- `semantic_similarity` - Embedding similarity score (0.0-1.0, higher = more similar)
- `context_similarity` - BDD hierarchy similarity (container/describe blocks)
- `query_file` / `matched_file` - Test file paths (shows cross-file duplicates)

### Common Analysis Commands

```bash
# Find perfect duplicates
grep "1.000" spec-md/markdown_similarity_results.csv

# Find cross-repository duplicates
awk -F',' '$3 != $6' spec-md/markdown_similarity_results.csv | head -10

# Count matches by similarity level
awk -F',' 'NR>1 {
  if ($7 >= 0.95) print "Very high"
  else if ($7 >= 0.85) print "High"
  else print "Medium"
}' spec-md/markdown_similarity_results.csv | sort | uniq -c
```

## 📊 Test Spec Files (JSONL)

Each line contains a complete test specification:

```json
{
  "test_id": "eco-gotests/tests/pod_test.go:TestPodHealth",
  "desc": "should verify pod is running",
  "test_type": "integration",
  "purpose": "POD_HEALTH",
  "actions": [{"gvk": "v1/Pod", "verb": "get"}],
  "environment": ["multi_node"],
  "container_labels": ["Describe: Pod Tests", "Context: Health Checks"]
}
```

**Useful queries:**

```bash
# Count tests by purpose
jq -r '.purpose' spec-md/go_specs_per_it.jsonl | sort | uniq -c | sort -nr

# Find tests with specific operations
jq -r 'select(.actions[].gvk == "v1/Pod") | .test_id' spec-md/go_specs_per_it.jsonl
```

## 🎯 Purpose Categories

Tests are automatically categorized by purpose for better filtering:

| Purpose | Description | Example Keywords |
|---------|-------------|------------------|
| POD_HEALTH | Pod status validation | `running`, `status`, `health` |
| POD_MANAGEMENT | Pod lifecycle | `create`, `delete`, `update` |
| NETWORK_CONNECTIVITY | Network reachability | `curl`, `routing`, `connectivity` |
| NETWORK_POLICY | Network policies | `policy`, `ingress`, `egress` |
| RESOURCE_VALIDATION | Resource checks | `count`, `exist`, `validation` |
| OPERATOR_MANAGEMENT | Operator testing | `operator`, `csv`, `subscription` |
| STORAGE_TESTING | Storage/volumes | `pvc`, `pv`, `mount` |
| SECURITY_TESTING | Security contexts | `rbac`, `scc`, `psa` |

## 🛠 Manual Usage (Advanced)

Most users should use the automated pipeline above. For manual extraction:

### Go Extractor

```bash
cd go-extractor
go build -o kubespec-go
./kubespec-go -root /path/to/go/repo > ../go_specs.jsonl
```

### Python Extractor

```bash
cd py-extractor
python extract_kubespec.py --root /path/to/python/tests > ../py_specs.jsonl
```

### Similarity Analysis

```bash
python match/markdown-similarity.py \
  --jsonl all_specs_per_it.jsonl \
  --markdown spec-md/markdown/ \
  --output similarity_results.csv \
  --threshold 0.75
```

## 🔧 Configuration

### Adjust Purpose Compatibility

Edit `match/markdown-similarity.py`:

```python
PURPOSE_COMPATIBILITY = {
    'POD_MANAGEMENT': ['POD_HEALTH', 'RESOURCE_VALIDATION'],
    # Add your own compatibility rules
}
```

### Change Similarity Thresholds

```bash
# Stricter matching (fewer results, higher quality)
./extract-spec-md.sh -g /path/to/tests --threshold 0.85

# More lenient (more results, more false positives)
./extract-spec-md.sh -g /path/to/tests --threshold 0.65
```

## 🏗 Architecture

```
extract-spec-md.sh
    ├─→ Go Extractor (AST parsing)     → go_specs_per_it.jsonl + markdown/
    ├─→ Python Extractor (AST parsing) → py_specs_per_it.jsonl + markdown/
    └─→ Similarity Analysis (FAISS)    → markdown_similarity_results.csv
                                       → similarity_analysis.md
```

**Components:**

- **go-extractor/** - Ginkgo/Gomega pattern extraction with eco-goinfra support
- **py-extractor/** - pytest pattern extraction with openshift library support
- **match/markdown-similarity.py** - FAISS-based semantic matching with BDD context

## 🐛 Troubleshooting

### Missing Dependencies

```bash
# Python dependencies
cd match && pip install -r requirements.txt

# Go version (requires 1.20+)
go version
```

### Empty Results

- Verify test files follow expected patterns (Ginkgo for Go, pytest for Python)
- Check repository paths are correct and accessible
- Ensure test files are not empty

### Low Similarity Scores

- Adjust threshold: `--threshold 0.65` for more matches
- Check if tests actually test similar things (the tool is working correctly)
- Review purpose compatibility settings if filtering too aggressively

## 🔧 Debug Tools

For debugging extraction or matching issues:

```bash
# Check what the extractor sees
./go-extractor/kubespec-go -root /path/to/tests | jq . | less

# Verify similarity analysis
python match/markdown-similarity.py --help

# Enable verbose output
./extract-spec-md.sh -v -g /path/to/tests
```

## 📈 Performance Metrics

**Typical performance:**
- Extraction: <1 second per test file
- Embedding creation: ~4 seconds for 1,000 tests
- Similarity search: ~4 seconds for 1,000 queries
- Total pipeline: ~1-2 minutes for 3-5 repositories

**Filtering effectiveness:**
- Purpose-based filtering reduces false positives by 78%
- Generic assertion filtering improves cross-repo matching by 27%

## 🤝 Contributing

Contributions welcome! Key areas:

1. **Add purpose categories** - Update pattern detection in extractors
2. **Improve detection** - Enhance keyword patterns and operation inference
3. **New frameworks** - Extend AST visitors for additional test frameworks

## 📚 Dependencies

- **Go**: Standard library only (no external dependencies)
- **Python**: sentence-transformers, faiss-cpu, pandas, numpy
- See `match/requirements.txt` for complete list
