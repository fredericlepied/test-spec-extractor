import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("query-specs.py")
QUERIES = (("search", "sample"), ("similar", "sample"), ("by-resource", "Pod"))


class QueryLimitTests(unittest.TestCase):
    def run_query(self, data_dir, query, *options):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--data-dir", str(data_dir), *query, *options],
            capture_output=True,
            text=True,
        )

    def test_invalid_limits_are_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            for query in QUERIES:
                for limit in ("0", "-1", "1.5"):
                    with self.subTest(command=query[0], limit=limit):
                        result = self.run_query(directory, query, "--limit", limit)
                        self.assertEqual(result.returncode, 2)
                        self.assertIn("argument --limit: must be a positive integer", result.stderr)

    def test_explicit_and_default_limits_cap_matching_results(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            records = [{"desc": f"sample {index}", "k8s_resources": ["Pod"]} for index in range(21)]
            (data_dir / "all_specs_per_it.jsonl").write_text(
                "\n".join(map(json.dumps, records)), encoding="utf-8"
            )
            (data_dir / "markdown_similarity_results.json").write_text(
                json.dumps(
                    [
                        {"query_description": record["desc"], "semantic_similarity": 0.9}
                        for record in records
                    ]
                ),
                encoding="utf-8",
            )
            for query in QUERIES:
                for options, expected_count in (((), 20), (("--limit", "1"), 1)):
                    with self.subTest(command=query[0], options=options):
                        result = self.run_query(data_dir, query, *options)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(len(json.loads(result.stdout)), expected_count)
