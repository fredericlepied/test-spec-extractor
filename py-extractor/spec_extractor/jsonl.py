"""JSONL output matching Go extractor PerItRecord format exactly."""

import json
from typing import Optional

from .types import Container, FileSpec, TestCase


def write_per_it_jsonl(spec: FileSpec, file_path: str) -> None:
    """Write per-test JSONL records, matching Go extractor format exactly."""
    if spec is None or not spec.has_tests() or not file_path:
        return

    # Open in append mode (matching Go extractor behavior)
    with open(file_path, "a", encoding="utf-8") as f:
        # Walk and emit
        for container in spec.root.children:
            _emit_container(container, spec.file_path, f)


def _emit_container(container: Container, file_path: str, f):
    """Emit test cases from container recursively."""
    if container is None:
        return

    for test_case in container.cases:
        rec = {
            "desc": test_case.description,
            "file_path": file_path,
        }

        if test_case.labels:
            rec["labels"] = test_case.labels

        # PrepSteps: Container.PrepSteps + TestCase.PrepSteps (in order)
        prep_steps = []
        if container.prep_steps:
            for step in container.prep_steps:
                prep_steps.append(step.text)
        if test_case.prep_steps:
            for step in test_case.prep_steps:
                prep_steps.append(step.text)
        if prep_steps:
            rec["prep_steps"] = prep_steps

        # Steps: TestCase.Steps only
        steps = []
        for step in test_case.steps:
            steps.append(step.text)
        if steps:
            rec["steps"] = steps

        # CleanupSteps: Container.CleanupSteps + TestCase.CleanupSteps (in order)
        cleanup_steps = []
        if container.cleanup_steps:
            for step in container.cleanup_steps:
                cleanup_steps.append(step.text)
        if test_case.cleanup_steps:
            for step in test_case.cleanup_steps:
                cleanup_steps.append(step.text)
        if cleanup_steps:
            rec["cleanup_steps"] = cleanup_steps

        # Write JSON line
        json_line = json.dumps(rec, ensure_ascii=False)
        f.write(json_line + "\n")

    # Recursively process children
    for child in container.children:
        _emit_container(child, file_path, f)

