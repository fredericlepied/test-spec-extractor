"""Markdown rendering matching Go extractor format exactly."""

from typing import List

from .types import Container, FileSpec, TestCase, TestStep


def render_markdown(spec: FileSpec) -> bytes:
    """Render FileSpec to markdown bytes, matching Go extractor format exactly."""
    lines = []
    lines.append(f"## {spec.file_path}\n")

    # Walk containers from root
    for container in spec.root.children:
        _render_container(lines, container, 0, [])

    return "\n".join(lines).encode("utf-8")


def _render_container(lines: List[str], container: Container, depth: int, when_conditions: List[str]):
    """Render a container recursively, matching Go extractor format."""
    # Heading level by depth: 0=>###, 1=>####, 2=>#####
    level = 3 + depth
    if level > 6:
        level = 6
    heading = "#" * level
    lines.append(f"{heading} {container.kind}: {_safe(container.description)}\n")

    # Show inherited When conditions as prerequisites
    if when_conditions:
        lines.append(f"- **when**: {', '.join(when_conditions)}\n")

    if container.labels:
        lines.append(f"- **labels**: {', '.join(container.labels)}\n")

    if container.prep_steps:
        lines.append("- **preparation**:\n")
        for step in container.prep_steps:
            lines.append(f"  - {_safe(step.text)}\n")

    if container.cleanup_steps:
        lines.append("- **cleanup**:\n")
        for step in container.cleanup_steps:
            lines.append(f"  - {_safe(step.text)}\n")

    if container.cases:
        for test_case in container.cases:
            lines.append(f"- **Test**: {_safe(test_case.description)}\n")
            if test_case.labels:
                lines.append(f"  - labels: {', '.join(test_case.labels)}\n")
            if test_case.prep_steps:
                lines.append("  - preparation:\n")
                for step in test_case.prep_steps:
                    lines.append(f"    - {_safe(step.text)}\n")
            if test_case.steps:
                lines.append("  - steps:\n")
                for step in test_case.steps:
                    lines.append(f"    - {_safe(step.text)}\n")
            if test_case.cleanup_steps:
                lines.append("  - cleanup:\n")
                for step in test_case.cleanup_steps:
                    lines.append(f"    - {_safe(step.text)}\n")
        lines.append("")  # Empty line after test cases

    # Pass down When conditions to children, adding current one if this is a When block
    child_when_conditions = when_conditions.copy()
    if container.kind == "When" and container.description:
        child_when_conditions.append(_safe(container.description))

    for child in container.children:
        _render_container(lines, child, depth + 1, child_when_conditions)


def _safe(s: str) -> str:
    """Escape markdown special characters, matching Go extractor's safe function."""
    # Replace newlines with spaces
    s = s.replace("\n", " ")
    # Escape underscores
    s = s.replace("_", "\\_")
    return s

