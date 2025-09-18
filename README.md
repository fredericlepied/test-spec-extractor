# Cross-Language Test Matcher (Go + Python)

Toolkit to extract KubeSpecs from Go/Python tests, build embeddings, and match cross-language overlap.
OpenShift-aware (Route↔Ingress, SCC↔PSA) with optional LLM re-ranking.

## How to run

1) Go extractor

```Shellsession
cd go-extractor
go build -o kubespec-go
./kubespec-go -root /path/to/go/repo > ../../go_specs.jsonl
```

2) Python extractor

```Shellsession
$ cd py-extractor
$ python extract_kubespec.py --root /path/to/python/tests > ../../py_specs.jsonl
```

3) Match

```Shellsession
$ cd match
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
$ python build_index_and_match.py --go go_specs.jsonl --py py_specs.jsonl --out report.csv --cov coverage_matrix.csv
```

4) Optional LLM re-rank

```Shell
export LLM_API_KEY=...
export LLM_MODEL=gpt-4o-mini  # or your local instruct model
python match/build_index_and_match.py --go go_specs.jsonl --py py_specs.jsonl --out report.csv --llm
```
