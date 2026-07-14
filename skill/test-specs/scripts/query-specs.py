#!/usr/bin/env python3
"""Query test specifications extracted by test-spec-extractor."""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "spec-md").is_dir() or (parent / "extract-spec-md.sh").exists():
            return parent
    return p.parent.parent


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_language(file_path):
    return "python" if file_path.endswith(".py") else "go"


def polarion_url(test_id):
    tid = test_id.lstrip("OCP-") if test_id.startswith("OCP-") else test_id
    prefix = "OCP-" if tid.isdigit() else ""
    return (
        f"https://polarion.engineering.redhat.com/polarion/#/project/OSE/workitem?id={prefix}{tid}"
    )


def resolve_data_dir(args):
    if args.data_dir:
        return Path(args.data_dir)
    return find_project_root() / "spec-md"


# --- Filters ---


def apply_filters(records, args):
    filtered = records
    if getattr(args, "repo", None):
        filtered = [r for r in filtered if r.get("repo") == args.repo]
    if getattr(args, "lang", None):
        lang = args.lang.lower()
        filtered = [r for r in filtered if detect_language(r.get("file_path", "")) == lang]
    if getattr(args, "resource", None):
        res = args.resource
        filtered = [r for r in filtered if res in r.get("k8s_resources", [])]
    return filtered


# --- Output formatting ---


def truncate(s, max_len=80):
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def format_test_summary(rec):
    lang = detect_language(rec.get("file_path", ""))
    tid = rec.get("test_id", "")
    repo = rec.get("repo", "")
    desc = rec.get("desc", "")
    return f"[{repo}/{lang}] {desc}" + (f"  (OCP-{tid})" if tid else "")


def format_test_detail(rec):
    lines = []
    lines.append(f"Description: {rec.get('desc', '')}")
    lines.append(f"Repository:  {rec.get('repo', '')}")
    lines.append(f"Language:    {detect_language(rec.get('file_path', ''))}")
    if rec.get("test_id"):
        lines.append(f"Polarion ID: OCP-{rec['test_id']}")
        lines.append(f"Polarion:    {polarion_url(rec['test_id'])}")
    if rec.get("source_url"):
        lines.append(f"Source:      {rec['source_url']}")
    elif rec.get("line_number"):
        lines.append(f"Line:        {rec['line_number']}")
    if rec.get("k8s_resources"):
        lines.append(f"K8s:         {', '.join(rec['k8s_resources'])}")
    if rec.get("labels"):
        lines.append(f"Labels:      {', '.join(rec['labels'])}")
    if rec.get("skip_conditions"):
        lines.append(f"Skip if:     {' | '.join(rec['skip_conditions'])}")
    if rec.get("prep_steps"):
        lines.append("Prep:")
        for s in rec["prep_steps"]:
            lines.append(f"  - {s}")
    if rec.get("steps"):
        lines.append("Steps:")
        for s in rec["steps"]:
            lines.append(f"  - {s}")
    if rec.get("validations"):
        lines.append("Validations:")
        for v in rec["validations"]:
            lines.append(f"  - {v}")
    if rec.get("cleanup_steps"):
        lines.append("Cleanup:")
        for s in rec["cleanup_steps"]:
            lines.append(f"  - {s}")
    return "\n".join(lines)


# --- Subcommands ---


def cmd_search(args):
    data_dir = resolve_data_dir(args)
    records = load_jsonl(data_dir / "all_specs_per_it.jsonl")
    records = apply_filters(records, args)

    pattern = re.compile(re.escape(args.keyword), re.IGNORECASE)
    matches = [r for r in records if pattern.search(r.get("desc", ""))]

    if not args.text:
        json.dump(matches[: args.limit], sys.stdout, indent=2)
        print()
        return

    print(f"Found {len(matches)} tests matching '{args.keyword}'")
    if len(matches) > args.limit:
        print(f"(showing first {args.limit}, use --limit to see more)\n")
    else:
        print()
    for rec in matches[: args.limit]:
        print(format_test_summary(rec))


def cmd_details(args):
    data_dir = resolve_data_dir(args)
    records = load_jsonl(data_dir / "all_specs_per_it.jsonl")

    matches = []
    if args.test_id:
        tid = args.test_id.lstrip("OCP-") if args.test_id.startswith("OCP-") else args.test_id
        matches = [r for r in records if r.get("test_id") == tid]
    elif args.desc:
        pattern = re.compile(re.escape(args.desc), re.IGNORECASE)
        matches = [r for r in records if pattern.search(r.get("desc", ""))]

    if not args.text:
        json.dump(matches, sys.stdout, indent=2)
        print()
        return

    if not matches:
        print("No tests found.")
        return

    for i, rec in enumerate(matches):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        print(format_test_detail(rec))


def cmd_similar(args):
    data_dir = resolve_data_dir(args)
    sim_path = data_dir / "markdown_similarity_results.json"
    if not sim_path.exists():
        print(f"Similarity data not found: {sim_path}", file=sys.stderr)
        sys.exit(1)

    similarity = load_json(sim_path)

    pattern = re.compile(re.escape(args.description), re.IGNORECASE)
    matches = []
    for rec in similarity:
        score = rec.get("semantic_similarity", 0)
        if score < args.min_score:
            continue
        if args.cross_only and not rec.get("is_cross_language"):
            continue
        q_desc = rec.get("query_description", "")
        m_desc = rec.get("matched_description", "")
        if pattern.search(q_desc) or pattern.search(m_desc):
            matches.append(rec)

    matches.sort(key=lambda r: r.get("semantic_similarity", 0), reverse=True)

    if not args.text:
        json.dump(matches[: args.limit], sys.stdout, indent=2)
        print()
        return

    print(f"Found {len(matches)} similarity matches (>= {args.min_score})")
    if len(matches) > args.limit:
        print(f"(showing top {args.limit})\n")
    else:
        print()
    for rec in matches[: args.limit]:
        score = rec.get("semantic_similarity", 0)
        q_desc = rec.get("query_description", "")
        m_desc = rec.get("matched_description", "")
        q_file = rec.get("query_file", "")
        m_file = rec.get("matched_file", "")
        cross = " [cross-lang]" if rec.get("is_cross_language") else ""
        print(f"  {score:.4f}{cross}")
        print(f"    A: {truncate(q_desc, 100)}")
        print(f"       {q_file}")
        print(f"    B: {truncate(m_desc, 100)}")
        print(f"       {m_file}")
        print()


def cmd_stats(args):
    data_dir = resolve_data_dir(args)

    stats_path = find_project_root() / "web" / "public" / "data" / "stats.json"
    if stats_path.exists() and not args.repo:
        stats = load_json(stats_path)
        if not args.text:
            json.dump(stats, sys.stdout, indent=2)
            print()
            return
        print(f"Total tests:          {stats['totalTests']}")
        print(f"  Go tests:           {stats['goTests']}")
        print(f"  Python tests:       {stats['pyTests']}")
        print(f"Similarity matches:   {stats['totalMatches']}")
        print(f"  Cross-language:     {stats['crossLanguageMatches']}")
        print(f"  Avg similarity:     {stats['avgSimilarity']:.4f}")
        print(f"\nPolarion coverage:    {stats['testIdCoverage']['withId']}/{stats['totalTests']}")
        print(f"\nTests per repository:")
        for r in stats["repos"]:
            print(f"  {r['name']:30s} {r['count']:5d}")
        print(f"\nScore distribution:")
        for b in stats["scoreDistribution"]:
            print(f"  {b['bucket']:10s} {b['count']:5d}")
        return

    records = load_jsonl(data_dir / "all_specs_per_it.jsonl")
    if args.repo:
        records = [r for r in records if r.get("repo") == args.repo]

    total = len(records)
    go_count = sum(1 for r in records if detect_language(r.get("file_path", "")) == "go")
    py_count = total - go_count
    with_id = sum(1 for r in records if r.get("test_id"))
    repo_counts = Counter(r.get("repo", "unknown") for r in records)
    resource_counts = Counter()
    for r in records:
        for res in r.get("k8s_resources", []):
            resource_counts[res] += 1

    result = {
        "totalTests": total,
        "goTests": go_count,
        "pyTests": py_count,
        "withTestId": with_id,
        "repos": dict(repo_counts.most_common()),
        "topResources": dict(resource_counts.most_common(20)),
    }

    if not args.text:
        json.dump(result, sys.stdout, indent=2)
        print()
        return

    label = f" ({args.repo})" if args.repo else ""
    print(f"Total tests{label}:     {total}")
    print(f"  Go tests:           {go_count}")
    print(f"  Python tests:       {py_count}")
    print(f"  With Polarion ID:   {with_id}")
    print(f"\nTests per repository:")
    for name, count in repo_counts.most_common():
        print(f"  {name:30s} {count:5d}")
    print(f"\nTop K8s resources:")
    for name, count in resource_counts.most_common(20):
        print(f"  {name:30s} {count:5d}")


def cmd_list_resources(args):
    data_dir = resolve_data_dir(args)
    records = load_jsonl(data_dir / "all_specs_per_it.jsonl")
    if args.repo:
        records = [r for r in records if r.get("repo") == args.repo]

    resource_counts = Counter()
    for r in records:
        for res in r.get("k8s_resources", []):
            resource_counts[res] += 1

    if not args.text:
        json.dump(
            [{"resource": k, "count": v} for k, v in resource_counts.most_common()],
            sys.stdout,
            indent=2,
        )
        print()
        return

    label = f" ({args.repo})" if args.repo else ""
    print(f"K8s resources{label}: {len(resource_counts)} distinct\n")
    for name, count in resource_counts.most_common():
        print(f"  {name:30s} {count:5d}")


def cmd_by_resource(args):
    data_dir = resolve_data_dir(args)
    records = load_jsonl(data_dir / "all_specs_per_it.jsonl")

    res = args.name
    matches = [r for r in records if res in r.get("k8s_resources", [])]
    if args.repo:
        matches = [r for r in matches if r.get("repo") == args.repo]

    if not args.text:
        json.dump(matches[: args.limit], sys.stdout, indent=2)
        print()
        return

    print(f"Found {len(matches)} tests using '{res}'")
    if len(matches) > args.limit:
        print(f"(showing first {args.limit})\n")
    else:
        print()
    for rec in matches[: args.limit]:
        print(format_test_summary(rec))


def cmd_by_id(args):
    data_dir = resolve_data_dir(args)
    records = load_jsonl(data_dir / "all_specs_per_it.jsonl")

    tid = args.polarion_id
    if tid.startswith("OCP-"):
        tid = tid[4:]

    matches = [r for r in records if r.get("test_id") == tid]

    if not args.text:
        json.dump(matches, sys.stdout, indent=2)
        print()
        return

    if not matches:
        print(f"No test found with Polarion ID: OCP-{tid}")
        return

    for i, rec in enumerate(matches):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        print(format_test_detail(rec))


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Query extracted test specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", help="Path to spec-md/ directory (auto-detected by default)")
    parser.add_argument("--text", action="store_true", help="Output as human-readable text instead of JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p = sub.add_parser("search", help="Search tests by keyword")
    p.add_argument("keyword", help="Keyword to search in test descriptions")
    p.add_argument("--repo", help="Filter by repository name")
    p.add_argument("--lang", choices=["go", "python"], help="Filter by language")
    p.add_argument("--resource", help="Filter by K8s resource")
    p.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    # details
    p = sub.add_parser("details", help="Show full test details")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--test-id", help="Polarion test ID (e.g., 72245 or OCP-72245)")
    g.add_argument("--desc", help="Test description (substring match)")

    # similar
    p = sub.add_parser("similar", help="Find similar tests")
    p.add_argument("description", help="Test description to search for")
    p.add_argument("--min-score", type=float, default=0.75, help="Min similarity (default: 0.75)")
    p.add_argument("--cross-only", action="store_true", help="Cross-language matches only")
    p.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    # stats
    p = sub.add_parser("stats", help="Show summary statistics")
    p.add_argument("--repo", help="Filter by repository name")

    # list-resources
    p = sub.add_parser("list-resources", help="List K8s resources with counts")
    p.add_argument("--repo", help="Filter by repository name")

    # by-resource
    p = sub.add_parser("by-resource", help="Find tests by K8s resource")
    p.add_argument("name", help="K8s resource name (e.g., Pod, SR-IOV, Route)")
    p.add_argument("--repo", help="Filter by repository name")
    p.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    # by-id
    p = sub.add_parser("by-id", help="Lookup test by Polarion ID")
    p.add_argument("polarion_id", help="Polarion test ID (e.g., 72245 or OCP-72245)")

    args = parser.parse_args()

    commands = {
        "search": cmd_search,
        "details": cmd_details,
        "similar": cmd_similar,
        "stats": cmd_stats,
        "list-resources": cmd_list_resources,
        "by-resource": cmd_by_resource,
        "by-id": cmd_by_id,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
