"""Data structures matching Go extractor exactly."""

from typing import List, Optional


class TestStep:
    """A single test step with text description."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return f"TestStep(text={self.text!r})"


class TestCase:
    """A test case (like It block in Ginkgo)."""

    def __init__(self):
        self.description: str = ""
        self.labels: List[str] = []
        self.prep_steps: List[TestStep] = []  # Individual test prerequisites (including Skip conditions)
        self.steps: List[TestStep] = []
        self.cleanup_steps: List[TestStep] = []  # Cleanup actions and Fail messages

    def __repr__(self) -> str:
        return f"TestCase(description={self.description!r}, labels={self.labels}, steps={len(self.steps)})"


class Container:
    """A container (Describe, Context, When, or Root)."""

    def __init__(self, kind: str = "Root", description: str = ""):
        self.kind: str = kind  # Describe, Context, When, or Root
        self.description: str = description
        self.labels: List[str] = []
        self.children: List["Container"] = []
        self.prep_steps: List[TestStep] = []  # By(...) found in active Before* under this container
        self.cleanup_steps: List[TestStep] = []  # By(...) found in active After* under this container
        self.cases: List[TestCase] = []

    def __repr__(self) -> str:
        return f"Container(kind={self.kind!r}, description={self.description!r}, children={len(self.children)}, cases={len(self.cases)})"


class FileSpec:
    """A file specification with root container."""

    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.root: Container = Container(kind="Root")

    def has_tests(self) -> bool:
        """Returns true if any Container in the tree contains at least one TestCase."""
        return _container_has_tests(self.root)

    def __repr__(self) -> str:
        return f"FileSpec(file_path={self.file_path!r}, has_tests={self.has_tests()})"


def _container_has_tests(c: Container) -> bool:
    """Helper to check if container has tests."""
    if c is None:
        return False
    if len(c.cases) > 0:
        return True
    for child in c.children:
        if _container_has_tests(child):
            return True
    return False

