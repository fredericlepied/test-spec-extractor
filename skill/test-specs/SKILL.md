---
name: test-specs
description: >
  Query extracted BDD test specifications from OpenShift Go (Ginkgo) and Python (pytest)
  test repositories. Search tests by keyword, K8s resource, repo, or Polarion ID.
  Find similar/duplicate tests across repositories.
  Use when asked about test coverage, test specifications, test duplicates,
  or K8s resource testing patterns.
---

# Test Spec Query Skill

## How to Query Test Specs

Run `python3 skill/test-specs/scripts/query-specs.py <subcommand>` to query the extracted data. Output is JSON by default; add `--text` for human-readable output.

### Search tests by keyword

```bash
python3 skill/test-specs/scripts/query-specs.py search "upgrade"
python3 skill/test-specs/scripts/query-specs.py search "metallb" --repo eco-gotests
python3 skill/test-specs/scripts/query-specs.py search "PTP" --lang python
python3 skill/test-specs/scripts/query-specs.py search "operator" --resource SRIOV --limit 50
```

### Show full test details

```bash
python3 skill/test-specs/scripts/query-specs.py details --test-id 72245
python3 skill/test-specs/scripts/query-specs.py details --test-id OCP-72245
python3 skill/test-specs/scripts/query-specs.py details --desc "should upgrade successfully"
```

### Find similar or duplicate tests

```bash
python3 skill/test-specs/scripts/query-specs.py similar "upgrade" --min-score 0.90
python3 skill/test-specs/scripts/query-specs.py similar "ptp" --cross-only
```

### Get statistics

```bash
python3 skill/test-specs/scripts/query-specs.py stats
python3 skill/test-specs/scripts/query-specs.py stats --repo eco-gotests
```

### List and filter by K8s resources

```bash
python3 skill/test-specs/scripts/query-specs.py list-resources
python3 skill/test-specs/scripts/query-specs.py list-resources --repo cnf-gotests
python3 skill/test-specs/scripts/query-specs.py by-resource SRIOV
python3 skill/test-specs/scripts/query-specs.py by-resource Route --repo openshift-tests-private
```

### Look up by Polarion ID

```bash
python3 skill/test-specs/scripts/query-specs.py by-id 72245
python3 skill/test-specs/scripts/query-specs.py by-id OCP-53792
```

## Data Files

Read these files directly when the query utility does not cover your need:

- `spec-md/all_specs_per_it.jsonl` — ~4,886 test records (1 JSON per line)
- `spec-md/markdown_similarity_results.json` — ~6,076 similarity match pairs
- `spec-md/markdown/{repo}/.../*.md` — per-source-file structured markdown specs
- `web/public/data/stats.json` — pre-computed dashboard statistics

Repositories: `cnf-gotests`, `eco-gotests`, `openshift-tests`, `openshift-tests-private` (Go/Ginkgo), `eco-pytests`, `slcm-tests` (Python/pytest).

## Ad-hoc jq Queries

Use these `jq` one-liners for queries not covered by the utility:

```bash
# Count tests per repo
jq -r '.repo' spec-md/all_specs_per_it.jsonl | sort | uniq -c | sort -rn

# Find tests exercising a specific K8s resource
jq -r 'select(.k8s_resources[]? == "SRIOV") | .desc' spec-md/all_specs_per_it.jsonl

# List tests with Polarion IDs in a repo
jq -r 'select(.repo == "eco-gotests" and .test_id) | "\(.test_id) \(.desc)"' spec-md/all_specs_per_it.jsonl

# Show top 10 most common K8s resources
jq -r '.k8s_resources[]?' spec-md/all_specs_per_it.jsonl | sort | uniq -c | sort -rn | head -10

# Find high-similarity cross-language matches
jq -r '.[] | select(.is_cross_language and .semantic_similarity > 0.90) | "\(.semantic_similarity | tostring | .[:6]) \(.query_description) <-> \(.matched_description)"' spec-md/markdown_similarity_results.json

# Find tests with no steps extracted
jq -r 'select((.steps | length) == 0) | "\(.repo) \(.desc)"' spec-md/all_specs_per_it.jsonl | head -20
```

## Read markdown specs for a whole test file

```bash
# List markdown files for a repo
find spec-md/markdown/eco-gotests -name "*.md" | head -10

# Read a specific test file's full BDD spec
cat spec-md/markdown/eco-gotests/tests/accel/upgrade/tests/upgrade.go.md
```

## JSONL Record Schema

Each line in `all_specs_per_it.jsonl`:

| Field | Type | Present in |
|---|---|---|
| `desc` | string | Go + Python |
| `file_path` | string | Go + Python |
| `repo` | string | Go + Python |
| `steps` | string[] | Go + Python |
| `validations` | string[] | Go only |
| `test_id` | string | Go only (numeric, without OCP- prefix) |
| `line_number` | int | Go only |
| `k8s_resources` | string[] | Go + some Python |
| `source_url` | string | Go only |
| `labels` | string[] | Go only |
| `prep_steps` | string[] | Go + Python |
| `skip_conditions` | string[] | Go only |
| `cleanup_steps` | string[] | Go only |

## Similarity Record Schema

Each entry in `markdown_similarity_results.json`:

| Field | Type |
|---|---|
| `query_description`, `matched_description` | string |
| `query_file`, `matched_file` | string |
| `semantic_similarity` | float (0.0–1.0) |
| `context_similarity` | float (0.0–1.0) |
| `is_cross_language` | bool |
| `query_steps`, `matched_steps` | string[] |
| `query_validations`, `matched_validations` | string[] |
| `query_k8s_resources`, `matched_k8s_resources` | string[] |

## Tips

- Strip the `OCP-` prefix when searching JSONL directly (use `72245` not `OCP-72245`)
- Use `source_url` to navigate directly to the exact source line in GitHub/GitLab
- Filter cross-language matches (Go↔Python) to find consolidation opportunities
- Treat similarity >0.90 between different repos as likely duplicates
- Use the `labels` field for Ginkgo label filtering (`ginkgo --label-filter`)
- Common K8s resources (Pod, Node, Namespace, etc.) are filtered from similarity embeddings — they appear in >30% of tests
