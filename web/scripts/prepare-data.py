#!/usr/bin/env python3
"""Convert JSONL specs and similarity results into web-ready JSON bundles."""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

POLARION_URL = "https://polarion.engineering.redhat.com/polarion/#/project/OSE/workitem?id=OCP-{}"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_repo(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "external" and i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: find git repo root and use its directory name
    p = Path(file_path)
    for parent in p.parents:
        if (parent / ".git").exists():
            return parent.name
    return ""


def detect_language(file_path: str) -> str:
    return "python" if file_path.endswith(".py") else "go"


def make_relative_path(file_path: str, repo: str) -> str:
    marker = f"/{repo}/"
    idx = file_path.find(marker)
    if idx >= 0:
        return file_path[idx + len(marker) :]
    return file_path


_git_base_url_cache: dict[str, str] = {}
_SSH_RE = re.compile(r"git@([^:]+):(.+?)(?:\.git)?$")
_HTTPS_RE = re.compile(r"https?://([^/]+)/(.+?)(?:\.git)?$")


def _get_git_base_url(repo_root: str) -> str | None:
    """Get the browsable base URL for a git repo (e.g. https://github.com/org/repo/blob/main)."""
    if repo_root in _git_base_url_cache:
        return _git_base_url_cache[repo_root] or None

    try:
        remote = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        _git_base_url_cache[repo_root] = ""
        return None

    try:
        ref = (
            subprocess.check_output(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        branch = ref.split("/")[-1]
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "main"

    ssh_m = _SSH_RE.match(remote)
    https_m = _HTTPS_RE.match(remote)
    if ssh_m:
        host, path = ssh_m.group(1), ssh_m.group(2)
    elif https_m:
        host, path = https_m.group(1), https_m.group(2)
    else:
        _git_base_url_cache[repo_root] = ""
        return None

    sep = "/-/blob" if "gitlab" in host else "/blob"
    base = f"https://{host}/{path}{sep}/{branch}"
    _git_base_url_cache[repo_root] = base
    return base


def make_source_url_from_git(file_path: str, line_number: int | None) -> str | None:
    """Construct a source URL by reading git repo info from disk."""
    p = Path(file_path)
    for parent in p.parents:
        if (parent / ".git").exists():
            base = _get_git_base_url(str(parent))
            if not base:
                return None
            rel = p.relative_to(parent).as_posix()
            url = f"{base}/{rel}"
            if line_number:
                url += f"#L{line_number}"
            return url
    return None


_SOURCE_RE = re.compile(r"-\s*source:\s*\[.*?:(\d+)\]\((https?://\S+)\)")
_TEST_RE = re.compile(r"-\s*\*\*Test\*\*:\s*(.*)")
_FILE_RE = re.compile(r"^##\s+(.*)")


def build_source_url_map(markdown_dir: str) -> dict[str, str]:
    """Parse markdown files to extract source URLs.

    Returns a dict keyed by "repo_relative_path:line_number" -> URL.
    Also keyed by "repo:relative_path:desc" for fallback matching.
    """
    url_map: dict[str, str] = {}
    md_path = Path(markdown_dir)
    if not md_path.exists():
        return url_map

    for md_file in md_path.rglob("*.md"):
        current_file = ""
        current_repo = ""
        current_desc = ""
        for line in md_file.read_text(encoding="utf-8", errors="replace").splitlines():
            file_match = _FILE_RE.match(line)
            if file_match:
                current_file = file_match.group(1).strip()
                current_repo = extract_repo(current_file)
                continue

            test_match = _TEST_RE.match(line.strip())
            if test_match:
                current_desc = test_match.group(1).strip()
                continue

            source_match = _SOURCE_RE.match(line.strip())
            if source_match:
                line_num = source_match.group(1)
                url = source_match.group(2)
                if current_file and current_repo:
                    rel_path = make_relative_path(current_file, current_repo)
                    url_map[f"{current_file}:{line_num}"] = url
                    url_map[f"{current_repo}/{rel_path}:{line_num}"] = url
                    if current_desc:
                        url_map[f"{current_repo}:{current_desc}"] = url

    return url_map


def build_tests_json(specs: list[dict], source_urls: dict[str, str] | None = None) -> list[dict]:
    tests = []
    for i, spec in enumerate(specs):
        lang = detect_language(spec.get("file_path", ""))
        repo = spec.get("repo", "") or extract_repo(spec.get("file_path", ""))
        file_path = spec.get("file_path", "")
        rel_path = make_relative_path(file_path, repo) if repo else file_path
        test_id = spec.get("test_id")
        line_number = spec.get("line_number")

        source_url = spec.get("source_url") or None
        if not source_url and source_urls and file_path and line_number:
            source_url = source_urls.get(f"{file_path}:{line_number}")
            if not source_url and repo:
                source_url = source_urls.get(f"{repo}/{rel_path}:{line_number}")
        if not source_url and file_path:
            source_url = make_source_url_from_git(file_path, line_number)

        tests.append(
            {
                "id": f"{lang[:2]}_{i}",
                "desc": spec.get("desc", ""),
                "repo": repo,
                "language": lang,
                "filePath": rel_path,
                "lineNumber": line_number if line_number else None,
                "testId": test_id or None,
                "polarionUrl": POLARION_URL.format(test_id) if test_id else None,
                "sourceUrl": source_url,
                "steps": spec.get("steps", []),
                "validations": spec.get("validations", []),
                "k8sResources": spec.get("k8s_resources", []),
                "labels": spec.get("labels", []),
                "prepSteps": spec.get("prep_steps", []),
                "skipConditions": spec.get("skip_conditions", []),
                "cleanupSteps": spec.get("cleanup_steps", []),
            }
        )
    return tests


def extract_repo_from_file(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    return parts[0] if parts else ""


def _parse_repr_list(val) -> list[str]:
    """Parse a Python repr'd list string back into a list, or return as-is if already a list."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return []
    import ast

    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def build_similarity_json(raw: list[dict], source_urls: dict[str, str] | None = None) -> list[dict]:
    matches = []
    for i, r in enumerate(raw):
        query_file = r.get("query_file", "")
        matched_file = r.get("matched_file", "")
        is_cross = r.get("is_cross_language", False)

        if isinstance(is_cross, str):
            is_cross = is_cross.lower() == "true"

        query_lang = "python" if query_file.endswith(".py") else "go"
        matched_lang = "python" if matched_file.endswith(".py") else "go"

        if is_cross:
            match_type = "cross"
        elif query_lang == "python":
            match_type = "py-py"
        else:
            match_type = "go-go"

        query_repo = extract_repo_from_file(query_file)
        matched_repo = extract_repo_from_file(matched_file)
        query_desc = r.get("query_description", "")
        matched_desc = r.get("matched_description", "")

        query_source_url = None
        matched_source_url = None
        if source_urls:
            query_source_url = source_urls.get(f"{query_repo}:{query_desc}")
            matched_source_url = source_urls.get(f"{matched_repo}:{matched_desc}")

        matches.append(
            {
                "id": i,
                "queryDesc": query_desc,
                "queryFile": query_file,
                "queryRepo": query_repo,
                "querySourceUrl": query_source_url,
                "matchedDesc": matched_desc,
                "matchedFile": matched_file,
                "matchedRepo": matched_repo,
                "matchedSourceUrl": matched_source_url,
                "semanticSimilarity": round(r.get("semantic_similarity", 0), 4),
                "contextSimilarity": round(r.get("context_similarity", 0), 4),
                "isCrossLanguage": is_cross,
                "matchType": match_type,
                "sharedLabels": _parse_repr_list(r.get("shared_labels", [])),
                "querySteps": _parse_repr_list(r.get("query_steps", [])),
                "queryValidations": _parse_repr_list(r.get("query_validations", [])),
                "queryK8sResources": _parse_repr_list(r.get("query_k8s_resources", [])),
                "matchedSteps": _parse_repr_list(r.get("matched_steps", [])),
                "matchedValidations": _parse_repr_list(r.get("matched_validations", [])),
                "matchedK8sResources": _parse_repr_list(r.get("matched_k8s_resources", [])),
            }
        )
    return matches


def _compute_repo_overlap(
    tests: list[dict], matches: list[dict], repo_counter: Counter
) -> list[dict]:
    """Compute per-repo overlap: % of tests with a near-duplicate (>= 0.90), split by internal vs cross-repo."""
    OVERLAP_THRESHOLD = 0.90
    internal: dict[str, set[str]] = {repo: set() for repo in repo_counter}
    cross: dict[str, set[str]] = {repo: set() for repo in repo_counter}

    for m in matches:
        if m["semanticSimilarity"] < OVERLAP_THRESHOLD:
            continue
        qr, mr = m["queryRepo"], m["matchedRepo"]
        if qr == mr:
            if qr in internal:
                internal[qr].add(m["queryDesc"])
                internal[qr].add(m["matchedDesc"])
        else:
            if qr in cross:
                cross[qr].add(m["queryDesc"])
            if mr in cross:
                cross[mr].add(m["matchedDesc"])

    result = []
    for repo, count in repo_counter.most_common():
        cross_set = cross.get(repo, set())
        internal_only = internal.get(repo, set()) - cross_set
        cross_count = len(cross_set)
        internal_count = len(internal_only)
        cross_pct = round(100 * cross_count / count, 1) if count > 0 else 0
        internal_pct = round(100 * internal_count / count, 1) if count > 0 else 0
        result.append(
            {
                "repo": repo,
                "total": count,
                "crossRepo": cross_count,
                "crossRepoPct": cross_pct,
                "internal": internal_count,
                "internalPct": internal_pct,
            }
        )
    return result


def _build_heatmap(tests: list[dict], common_resources: set[str]) -> dict:
    """Build repo × resource matrix for the heatmap, top 25 distinctive resources."""
    repo_resource: dict[str, Counter] = {}
    for t in tests:
        repo = t["repo"]
        if repo not in repo_resource:
            repo_resource[repo] = Counter()
        for r in t["k8sResources"]:
            if r not in common_resources:
                repo_resource[repo][r] += 1

    total_resource: Counter[str] = Counter()
    for rc in repo_resource.values():
        total_resource.update(rc)
    top_resources = [r for r, _ in total_resource.most_common(25)]

    repos = sorted(repo_resource.keys())
    cells = []
    for repo in repos:
        for res in top_resources:
            count = repo_resource.get(repo, Counter()).get(res, 0)
            cells.append({"repo": repo, "resource": res, "count": count})

    return {"repos": repos, "resources": top_resources, "cells": cells}


def build_clusters_json(tests: list[dict], matches: list[dict]) -> list[dict]:
    """Build duplicate clusters using union-find on matches >= 0.90, with pre-computed graph layout."""
    import networkx as nx

    THRESHOLD = 0.90
    COLORS = [
        "#3b82f6",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#ec4899",
        "#14b8a6",
        "#f97316",
        "#6366f1",
        "#84cc16",
    ]

    test_lookup = {}
    for t in tests:
        test_lookup[f"{t['repo']}:{t['desc']}"] = t

    high_matches = [m for m in matches if m["semanticSimilarity"] >= THRESHOLD]
    if not high_matches:
        return []

    G = nx.Graph()
    for m in high_matches:
        q_key = f"{m['queryRepo']}:{m['queryDesc']}"
        m_key = f"{m['matchedRepo']}:{m['matchedDesc']}"
        G.add_node(q_key)
        G.add_node(m_key)
        G.add_edge(q_key, m_key, score=m["semanticSimilarity"])

    repos_sorted = sorted({k.split(":")[0] for k in G.nodes()})
    repo_color = {r: i % len(COLORS) for i, r in enumerate(repos_sorted)}

    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    clusters = []
    for cluster_id, component in enumerate(
        sorted(nx.connected_components(G), key=len, reverse=True)
    ):
        subgraph = G.subgraph(component)
        node_list = sorted(component)
        node_idx = {n: i for i, n in enumerate(node_list)}

        cluster_tests = []
        for n in node_list:
            t = test_lookup.get(n, {})
            repo = n.split(":")[0]
            nx_pos = pos[n]
            cluster_tests.append(
                {
                    "desc": t.get("desc", n.split(":", 1)[1] if ":" in n else n),
                    "repo": repo,
                    "file": t.get("filePath", ""),
                    "sourceUrl": t.get("sourceUrl"),
                    "language": t.get("language", "go"),
                    "testId": t.get("testId"),
                    "x": round((nx_pos[0] - x_min) / x_range * 800, 1),
                    "y": round((nx_pos[1] - y_min) / y_range * 600, 1),
                    "degree": subgraph.degree(n),
                    "colorIndex": repo_color.get(repo, 0),
                }
            )

        edge_scores = []
        cluster_edges = []
        for u, v, data in subgraph.edges(data=True):
            edge_scores.append(data["score"])
            cluster_edges.append(
                {
                    "sourceIdx": node_idx[u],
                    "targetIdx": node_idx[v],
                    "score": round(data["score"], 4),
                }
            )

        cluster_repos = sorted({t["repo"] for t in cluster_tests})
        clusters.append(
            {
                "id": cluster_id,
                "size": len(cluster_tests),
                "repos": cluster_repos,
                "isCrossRepo": len(cluster_repos) > 1,
                "maxScore": round(max(edge_scores), 4),
                "avgScore": round(sum(edge_scores) / len(edge_scores), 4),
                "tests": cluster_tests,
                "edges": cluster_edges,
            }
        )

    return clusters


def build_stats_json(tests: list[dict], matches: list[dict]) -> dict:
    go_tests = sum(1 for t in tests if t["language"] == "go")
    py_tests = sum(1 for t in tests if t["language"] == "python")

    repo_counter: Counter[str] = Counter()
    k8s_counter: Counter[str] = Counter()
    with_id = 0

    for t in tests:
        repo_counter[t["repo"]] += 1
        for r in t["k8sResources"]:
            k8s_counter[r] += 1
        if t["testId"]:
            with_id += 1

    total_tests = len(tests)
    common_threshold = 0.3
    common_resources = {
        r for r, c in k8s_counter.items() if total_tests > 0 and c / total_tests > common_threshold
    }

    go_go = sum(1 for m in matches if m["matchType"] == "go-go")
    py_py = sum(1 for m in matches if m["matchType"] == "py-py")
    cross = sum(1 for m in matches if m["matchType"] == "cross")

    scores = [m["semanticSimilarity"] for m in matches]
    avg_sim = sum(scores) / len(scores) if scores else 0

    buckets = [
        ("0.95-1.00", 0.95, 1.01),
        ("0.90-0.95", 0.90, 0.95),
        ("0.85-0.90", 0.85, 0.90),
        ("0.80-0.85", 0.80, 0.85),
        ("0.75-0.80", 0.75, 0.80),
        ("0.70-0.75", 0.70, 0.75),
        ("0.65-0.70", 0.65, 0.70),
    ]
    score_dist = []
    for label, lo, hi in buckets:
        count = sum(1 for s in scores if lo <= s < hi)
        score_dist.append({"bucket": label, "count": count})

    return {
        "totalTests": len(tests),
        "goTests": go_tests,
        "pyTests": py_tests,
        "totalMatches": len(matches),
        "crossLanguageMatches": cross,
        "avgSimilarity": round(avg_sim, 4),
        "repos": [
            {"name": name, "count": count, "language": "mixed"}
            for name, count in repo_counter.most_common()
        ],
        "scoreDistribution": score_dist,
        "k8sResources": [
            {"name": name, "count": count}
            for name, count in k8s_counter.most_common(50)
            if name not in common_resources
        ],
        "commonK8sResources": sorted(common_resources),
        "testIdCoverage": {"withId": with_id, "withoutId": len(tests) - with_id},
        "matchTypes": {"goGo": go_go, "pyPy": py_py, "cross": cross},
        "repoOverlap": _compute_repo_overlap(tests, matches, repo_counter),
        "heatmap": _build_heatmap(tests, common_resources),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare web UI data bundles")
    parser.add_argument(
        "--input", default="../spec-md", help="Input directory with JSONL/JSON files"
    )
    parser.add_argument("--output", default="public/data", help="Output directory for JSON bundles")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load specs
    jsonl_path = input_dir / "all_specs_per_it.jsonl"
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        return 1

    # Build source URL map from markdown files
    markdown_dir = input_dir / "markdown"
    source_urls: dict[str, str] = {}
    if markdown_dir.exists():
        print(f"Extracting source URLs from {markdown_dir}...")
        source_urls = build_source_url_map(str(markdown_dir))
        print(f"  {len(source_urls)} source URL mappings found")

    print(f"Loading specs from {jsonl_path}...")
    specs = load_jsonl(str(jsonl_path))
    tests = build_tests_json(specs, source_urls)
    print(f"  {len(tests)} tests processed")

    # Load similarity
    sim_json_path = input_dir / "markdown_similarity_results.json"
    sim_csv_path = input_dir / "markdown_similarity_results.csv"
    matches = []

    if sim_json_path.exists():
        print(f"Loading similarity from {sim_json_path}...")
        with open(sim_json_path, encoding="utf-8") as f:
            raw_sim = json.load(f)
        matches = build_similarity_json(raw_sim, source_urls)
        print(f"  {len(matches)} matches processed")
    elif sim_csv_path.exists():
        print(f"Loading similarity from CSV {sim_csv_path} (fallback)...")
        import csv

        with open(sim_csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_sim = list(reader)
        for r in raw_sim:
            for k in ("semantic_similarity", "context_similarity"):
                if k in r:
                    r[k] = float(r[k])
        matches = build_similarity_json(raw_sim, source_urls)
        print(f"  {len(matches)} matches processed (from CSV, no embedded test details)")
    else:
        print("Warning: No similarity results found")

    # Build stats and clusters
    stats = build_stats_json(tests, matches)

    print("Building duplicate clusters...")
    clusters = build_clusters_json(tests, matches)
    print(f"  {len(clusters)} clusters found")

    # Write outputs
    for name, data in [
        ("tests.json", tests),
        ("similarity.json", matches),
        ("stats.json", stats),
        ("clusters.json", clusters),
    ]:
        out_path = output_dir / name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Wrote {out_path} ({size_mb:.1f} MB)")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
