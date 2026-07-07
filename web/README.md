# Test Spec Explorer

Interactive web UI for browsing ~5,000 OpenShift test specifications across 6 repositories, with similarity analysis, duplicate detection, and K8s resource coverage.

## Quick Start

Open a terminal in this directory and run:

```bash
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser.

> **Note:** Opening `index.html` directly (`file://`) won't work — browsers block loading data files from the local filesystem. The one-liner above is all you need.

## Views

### Dashboard

Overview of all extracted test data. Everything is clickable:

- **Stat cards** — total tests, similar pairs, cross-language pairs, average similarity. Click to jump to the relevant filtered view.
- **Tests by Repository** — bar chart. Click a bar to see that repo's tests in the catalog.
- **Score Distribution** — histogram of similarity scores. Click a bucket to see matches in that range.
- **Test Overlap** — per-repo stacked bars showing internal (amber) and cross-repo (blue) near-duplicates (score >= 0.90). Click a segment to see the matching pairs.
- **K8s Resource Treemap** — top 30 distinctive resources by test count (common resources like Pod, Node filtered). Click to see tests using that resource.
- **Resource Coverage Heatmap** — repos vs resources matrix. Hover for counts, click a cell to see those tests.

### Similarity

Browse all similar test pairs. Left sidebar has filters:

- **Match Type** — All, Go-Go, Py-Py, Cross-language
- **Cross-repo only** — hide pairs within the same repo
- **Min Score** — threshold slider (0.65 to 1.00)
- **Repositories** — checkbox per repo

Click a row to see the side-by-side detail panel (pinned at top) showing both tests' skip conditions, preparation steps, test steps, validations, cleanup, K8s resources, and source links.

### Clusters

Groups of tests that are all near-duplicates of each other (connected components at >= 0.90 similarity). More actionable than individual pairs — shows "these N tests across M repos are all doing the same thing."

Filters: cross-repo only, minimum cluster size, sort by size/score/repos. Click a cluster to expand and see all tests with source and Polarion links.

### Graph

Visual map of test similarity. Each dot is a test, lines connect near-duplicates (>= 0.90). Dot size reflects how many similar tests it has. Color = repository.

- **Scroll** to zoom in/out
- **Drag** background to pan
- **Hover** a dot for test details
- **Click** a dot to find it in the catalog

Filter by repository and minimum cluster size in the left sidebar.

### Catalog

Searchable database of all tests. Full-text search across descriptions, steps, and validations.

Filters: repository, language (Go/Python), K8s resource, Polarion ID (has/doesn't have). Click a row to expand full details: skip conditions, preparation, steps, validations, cleanup, labels, with source code and Polarion links.

## Updating the Data

To regenerate with fresh repository data:

```bash
# From the test-spec-extractor root:
./extract-spec-md.sh -g /path/to/go-repo -p /path/to/py-repo

# Then rebuild the web app:
cd web
npm run build          # builds to dist/
# or
npm run archive        # builds + creates test-spec-explorer.tar.gz
```

To refresh the web data only (without rebuilding the app):

```bash
cd web
npm run prepare-data
```

## Source

Generated from [test-spec-extractor](https://github.com/fredericlepied/test-spec-extractor).
