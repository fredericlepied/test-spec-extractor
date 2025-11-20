#!/usr/bin/env python3
"""CLI entry point for Python spec extractor, matching Go extractor interface."""

import argparse
import os
import sys
from pathlib import Path

from .parser import parse_file
from .markdown import render_markdown
from .jsonl import write_per_it_jsonl


def split_csv(s: str) -> list[str]:
    """Split comma-separated string into list."""
    parts = s.split(",")
    return [p.strip() for p in parts if p.strip()]


def should_include_file(file_path: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    """Check if file should be included based on patterns."""
    # Simple pattern matching (for now, just check file extension and common exclusions)
    if not file_path.endswith(".py"):
        return False

    # Check exclude patterns
    for pattern in exclude_patterns:
        if pattern in file_path or file_path.endswith(pattern):
            return False

    # Check include patterns
    if include_patterns:
        for pattern in include_patterns:
            if pattern in file_path or file_path.endswith(pattern):
                return True
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract test specs from Python test files")
    parser.add_argument("--root", default=".", help="Root directory to scan for Python tests")
    parser.add_argument("--out", default="spec-out", help="Output directory for generated markdown")
    parser.add_argument(
        "--include",
        default="**/*.py",
        help="Comma-separated include globs (default: **/*.py)",
    )
    parser.add_argument(
        "--exclude",
        default="**/__pycache__/**,**/testdata/**,**/*_testdata/**",
        help="Comma-separated exclude globs",
    )
    parser.add_argument("--jsonl", default="", help="Optional path to write per-test JSONL records")
    args = parser.parse_args()

    # Create output directory if missing
    try:
        os.makedirs(args.out, mode=0o755, exist_ok=True)
    except OSError as e:
        print(f"failed creating output dir: {e}", file=sys.stderr)
        sys.exit(1)

    include_patterns = split_csv(args.include)
    exclude_patterns = split_csv(args.exclude)

    # Find Python test files
    root_path = Path(args.root)
    files = []
    for py_file in root_path.rglob("*.py"):
        file_str = str(py_file)
        # Skip common directories
        if "__pycache__" in file_str or "/testdata/" in file_str or file_str.endswith("_testdata"):
            continue
        # Check exclude patterns
        excluded = False
        for pattern in exclude_patterns:
            if pattern in file_str:
                excluded = True
                break
        if excluded:
            continue
        # All .py files are included by default (unless excluded)
        files.append(file_str)

    # Parse files and generate specs
    parsed = 0
    for file_path in files:
        try:
            spec = parse_file(file_path)
        except Exception as e:
            # Best-effort: skip unparsable files with a warning
            print(f"warn: parse error in {file_path}: {e}", file=sys.stderr)
            continue

        # Only write if the file contains at least one test case
        if spec.has_tests():
            try:
                # Calculate relative path for output
                rel_path = os.path.relpath(file_path, args.root)
                if rel_path.startswith(".."):
                    rel_path = os.path.basename(file_path)

                # Create output path
                out_path = os.path.join(args.out, rel_path.replace(".py", ".md"))
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, mode=0o755, exist_ok=True)

                # Write markdown
                markdown_bytes = render_markdown(spec)
                with open(out_path, "wb") as f:
                    f.write(markdown_bytes)

                # Write JSONL if requested
                if args.jsonl:
                    write_per_it_jsonl(spec, args.jsonl)

            except Exception as e:
                print(f"warn: write failed for {file_path}: {e}", file=sys.stderr)
                continue

        parsed += 1

    print(f"Parsed {parsed}/{len(files)} Python files under {args.root}. Output will be written to {args.out}")


if __name__ == "__main__":
    main()

