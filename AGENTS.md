# CLAUDE.md

This file provides guidance to AI Coding assistants when working with code in this repository.

## Commands

### Automated Pipeline (Recommended)
```bash
# Single repositories
./extract-and-match.sh -g /path/to/eco-gotests -p /path/to/eco-pytests

# Multiple repositories
./extract-and-match.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -p /path/to/eco-pytests

# Custom output directory
./extract-and-match.sh -g /path/to/go-tests -p /path/to/py-tests --output-dir my_results
```

### Manual Steps

#### Build and Run Extractors
```bash
# Go extractor
cd go-extractor
go build -o kubespec-go
./kubespec-go -root /path/to/go/repo > ../go_specs.jsonl

# Python extractor  
cd py-extractor
python extract_kubespec.py --root /path/to/python/tests > ../py_specs.jsonl
```

#### Matching and Analysis
```bash
cd match
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_index_and_match.py --go ../go_specs.jsonl --py ../py_specs.jsonl --out report.csv --cov coverage_matrix.csv

# With LLM re-ranking (requires environment variables)
export LLM_API_KEY=...
export LLM_MODEL=gpt-4o-mini
python build_index_and_match.py --go ../go_specs.jsonl --py ../py_specs.jsonl --out report.csv --llm
```

## Architecture

This is a cross-language test analysis toolkit that extracts KubeSpecs from Go and Python test files, then matches them using semantic embeddings. The system is OpenShift-aware with special handling for Route↔Ingress and SCC↔PSA equivalents.

### Core Components

**go-extractor/main.go**: AST-based Go test parser that extracts structured test specifications including:
- GVK (Group/Version/Kind) detection from composite literals
- Kubernetes API verb extraction (Create, Update, Delete, etc.)
- OpenShift-specific resource detection
- PSA (Pod Security Admission) label extraction from namespace manifests
- Test artifacts and golden files discovery

**py-extractor/extract_kubespec.py**: Python AST visitor that extracts similar specifications from Python tests:
- Method name-based verb detection (create_, patch_, delete_, etc.)
- CLI command parsing for kubectl/oc operations
- PSA and SCC pattern recognition in subprocess calls
- Parametrized test detection

**match/build_index_and_match.py**: Semantic matching engine using sentence transformers:
- Converts test specs to text embeddings using SentenceTransformer
- Uses FAISS for efficient similarity search
- Implements OpenShift equivalence expansion (Route↔Ingress, SCC↔PSA)
- Generates bidirectional matching pairs and coverage matrices

**match/llm_rerank.py**: Optional LLM-based re-ranking for improved semantic matching:
- Uses OpenAI-compatible APIs for test spec comparison
- Provides structured JSON output with overlap assessment
- Blends embedding-based and LLM scores for final ranking

### Key Data Structures

**KubeSpec/spec structure**: Standardized test specification with fields:
- `test_id`: File path and function name
- `level`: "integration" vs "unknown" based on mutating operations
- `actions`: List of GVK+verb combinations and detected operations
- `preconditions`: PSA labels, parametrization, equivalence bridges
- `openshift_specific`: OpenShift-only resources detected
- `artifacts`: Test data files and golden references

### OpenShift Awareness

The system automatically detects and creates equivalence bridges:
- Route (OpenShift) ↔ Ingress (Kubernetes) mapping
- SCC (OpenShift) ↔ PSA (Pod Security Admission) mapping
- Handles both API-level detection and CLI command parsing
