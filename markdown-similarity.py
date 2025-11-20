#!/usr/bin/env python3
"""
Markdown-Aware Test Similarity Analysis

This script analyzes similarity between BDD test specifications extracted as markdown
and JSONL data. Each test (It statement) becomes a single document in FAISS with
combined hierarchical context for better semantic matching.

Key Features:
- Hierarchical context embedding (Describe > Context > When > It)
- Combined global + test-specific information per document
- BDD structure-aware similarity scoring
- Semantic understanding of preparation/steps/cleanup sections
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class TestContext:
    """Hierarchical context for a single test"""

    file_path: str
    containers: List[str]  # Describe > Context > When hierarchy
    container_labels: List[str]
    preparation_steps: List[str]  # Inherited + test-specific
    cleanup_steps: List[str]  # Inherited + test-specific
    test_description: str
    test_labels: List[str]
    test_steps: List[str]
    test_prep_steps: List[str]  # Test-specific prep (Skip conditions)
    test_cleanup_steps: List[str]  # Test-specific cleanup


@dataclass
class TestDocument:
    """Single test document for FAISS indexing"""

    test_id: str
    context: TestContext
    combined_text: str  # For embedding
    structured_metadata: Dict[str, Any]  # For post-processing


class MarkdownSimilarityAnalyzer:
    """Markdown-aware test similarity analyzer using FAISS"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.test_documents: List[TestDocument] = []
        self.faiss_index = None
        self.test_id_to_doc: Dict[str, TestDocument] = {}

    def load_jsonl_specs(self, jsonl_path: str) -> List[TestContext]:
        """Load test specifications from JSONL file"""
        test_contexts = []

        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    spec = json.loads(line.strip())
                    context = self._jsonl_to_context(spec, line_num)
                    if context:
                        test_contexts.append(context)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

        return test_contexts

    def _jsonl_to_context(self, spec: Dict[str, Any], line_num: int) -> Optional[TestContext]:
        """Convert JSONL spec to TestContext"""
        if not spec.get("desc"):
            return None

        # Extract file path for grouping
        file_path = spec.get("file_path", f"unknown_file_{line_num}")

        # For now, create a flat context - we'll enhance this with markdown data
        return TestContext(
            file_path=file_path,
            containers=[],  # Will be enriched from markdown
            container_labels=[],
            preparation_steps=spec.get("prep_steps", []),
            cleanup_steps=spec.get("cleanup_steps", []),
            test_description=spec["desc"],
            test_labels=spec.get("labels", []),
            test_steps=spec.get("steps", []),
            test_prep_steps=[],  # Will be separated from prep_steps
            test_cleanup_steps=[],
        )

    def load_markdown_specs(self, markdown_dir: str) -> List[TestContext]:
        """Load and parse markdown specification files"""
        test_contexts = []
        markdown_path = Path(markdown_dir)

        md_files = list(markdown_path.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files to process...")

        for i, md_file in enumerate(md_files):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing file {i+1}/{len(md_files)}: {md_file.name}")
            try:
                contexts = self._parse_markdown_file(md_file)
                test_contexts.extend(contexts)
            except Exception as e:
                print(f"Warning: Failed to parse {md_file}: {e}")
                continue

        print(f"Completed processing {len(md_files)} files, extracted {len(test_contexts)} tests")
        return test_contexts

    def _parse_markdown_file(self, md_file: Path) -> List[TestContext]:
        """Parse a single markdown file into TestContext objects"""
        contexts = []

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read {md_file}: {e}")
            return contexts

        # Skip very large files that might cause issues
        if len(content) > 100000:  # 100KB limit
            print(f"Skipping large file {md_file.name} ({len(content)} bytes)")
            return contexts

        # Extract the source file path from the first line
        lines = content.split("\n")
        source_file = str(md_file)
        if lines and lines[0].startswith("## "):
            source_file = lines[0][3:].strip()

        # Parse hierarchical structure
        current_containers = []
        current_labels = []
        current_prep = []
        current_cleanup = []

        # Optimized state machine to parse the markdown efficiently
        # Track iterations only for the outer loop, not inner parsing loops
        i = 0
        max_iterations = len(lines) * 3  # Increased limit, but should rarely be needed
        iterations = 0

        while i < len(lines):
            iterations += 1
            # Safety check: if we've iterated more than expected, something is wrong
            if iterations > max_iterations:
                print(
                    f"Warning: Hit safety limit parsing {md_file.name}, may have incomplete extraction"
                )
                break

            line = lines[i].strip()

            # Track container hierarchy
            if line.startswith("###") and ":" in line:
                level = len(line) - len(line.lstrip("#"))
                container_info = line[level:].strip()
                if ":" in container_info:
                    kind, desc = container_info.split(":", 1)
                    container = f"{kind.strip()}: {desc.strip()}"

                    # Adjust hierarchy based on level
                    container_level = level - 3  # Normalize (### = 0, #### = 1, etc.)
                    if container_level < len(current_containers):
                        current_containers = current_containers[:container_level]
                    current_containers.append(container)
                i += 1
                continue

            # Extract test cases
            elif line.startswith("- **Test**:"):
                test_desc = line[11:].strip()

                # Skip tests with empty descriptions (but still process them)
                test_labels = []
                test_prep = []
                test_steps = []
                test_cleanup = []

                # Parse test details - optimized inner loop
                i += 1
                # Process until we hit the next test or container or end of file
                while i < len(lines):
                    inner_line = lines[i].strip()

                    # Stop conditions: next test, container header, or empty line after content
                    if inner_line.startswith("- **Test**:"):
                        break
                    if inner_line.startswith("###"):
                        break

                    # Parse test metadata efficiently
                    if inner_line.startswith("- labels:"):
                        test_labels = [l.strip() for l in inner_line[9:].split(",")]
                        i += 1
                    elif inner_line.startswith("- preparation:"):
                        i += 1
                        # Collect all preparation steps
                        while i < len(lines):
                            step_line = lines[i].strip()
                            if not step_line.startswith("- ") or step_line.startswith("- **"):
                                break
                            test_prep.append(step_line[2:])
                            i += 1
                        continue  # Don't increment i again, already at next line
                    elif inner_line.startswith("- steps:"):
                        i += 1
                        # Collect all steps
                        while i < len(lines):
                            step_line = lines[i].strip()
                            if not step_line.startswith("- ") or step_line.startswith("- **"):
                                break
                            test_steps.append(step_line[2:])
                            i += 1
                        continue  # Don't increment i again, already at next line
                    elif inner_line.startswith("- cleanup:"):
                        i += 1
                        # Collect all cleanup steps
                        while i < len(lines):
                            step_line = lines[i].strip()
                            if not step_line.startswith("- ") or step_line.startswith("- **"):
                                break
                            test_cleanup.append(step_line[2:])
                            i += 1
                        continue  # Don't increment i again, already at next line
                    elif not inner_line:
                        # Empty line - might be end of test, but continue to be safe
                        i += 1
                    else:
                        # Skip unrecognized lines
                        i += 1

                # Create TestContext (even if description is empty)
                context = TestContext(
                    file_path=source_file,
                    containers=current_containers.copy(),
                    container_labels=current_labels.copy(),
                    preparation_steps=current_prep.copy(),
                    cleanup_steps=current_cleanup.copy(),
                    test_description=test_desc,
                    test_labels=test_labels,
                    test_steps=test_steps,
                    test_prep_steps=test_prep,
                    test_cleanup_steps=test_cleanup,
                )
                contexts.append(context)
                continue  # Don't increment i, already positioned correctly

            # Default: move to next line
            i += 1

        return contexts

    def create_test_documents(self, test_contexts: List[TestContext]) -> List[TestDocument]:
        """Convert TestContext objects to TestDocument objects for FAISS"""
        documents = []

        for i, context in enumerate(test_contexts):
            test_id = f"test_{i}"

            # Create combined text for embedding
            combined_text = self._create_combined_text(context)

            # Create structured metadata
            metadata = {
                "file_path": context.file_path,
                "containers": context.containers,
                "test_description": context.test_description,
                "test_labels": context.test_labels,
                "preparation_count": len(context.preparation_steps + context.test_prep_steps),
                "steps_count": len(context.test_steps),
                "cleanup_count": len(context.cleanup_steps + context.test_cleanup_steps),
            }

            doc = TestDocument(
                test_id=test_id,
                context=context,
                combined_text=combined_text,
                structured_metadata=metadata,
            )
            documents.append(doc)

        return documents

    def _create_combined_text(self, context: TestContext) -> str:
        """Create combined text for embedding with hierarchical context"""
        parts = []

        # Add hierarchical context
        if context.containers:
            parts.append("Context: " + " > ".join(context.containers))

        # Add test description (most important)
        parts.append(f"Test: {context.test_description}")

        # Add preparation context
        all_prep = context.preparation_steps + context.test_prep_steps
        if all_prep:
            prep_text = " | ".join(all_prep)
            parts.append(f"Prerequisites: {prep_text}")

        # Add test steps (core functionality)
        if context.test_steps:
            steps_text = " | ".join(context.test_steps)
            parts.append(f"Steps: {steps_text}")

        # Add cleanup context
        all_cleanup = context.cleanup_steps + context.test_cleanup_steps
        if all_cleanup:
            cleanup_text = " | ".join(all_cleanup)
            parts.append(f"Cleanup: {cleanup_text}")

        # Add labels for additional context
        all_labels = context.container_labels + context.test_labels
        if all_labels:
            labels_text = " ".join(all_labels)
            parts.append(f"Labels: {labels_text}")

        return " || ".join(parts)

    def build_faiss_index(self, test_documents: List[TestDocument]) -> None:
        """Build FAISS index from test documents"""
        self.test_documents = test_documents
        self.test_id_to_doc = {doc.test_id: doc for doc in test_documents}

        if not test_documents:
            print("Warning: No test documents to index")
            return

        print(f"Creating embeddings for {len(test_documents)} test documents...")

        # Create embeddings
        texts = [doc.combined_text for doc in test_documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype("float32"))

        print(f"FAISS index built with {self.faiss_index.ntotal} documents")

    def find_similar_tests(
        self,
        query_documents: List[TestDocument],
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        cross_language_threshold: float = 0.65,
        exclude_same_file: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find similar tests using FAISS similarity search

        Args:
            query_documents: Documents to search for similarities
            top_k: Number of top similar tests to find per query
            similarity_threshold: Threshold for intra-language matches (Python↔Python, Go↔Go)
            cross_language_threshold: Lower threshold for inter-language matches (Python↔Go)
            exclude_same_file: Whether to exclude matches within the same file
        """
        if not self.faiss_index:
            raise ValueError("FAISS index not built. Call build_faiss_index first.")

        results = []
        seen_pairs: Set[Tuple[str, str]] = set()  # Track seen pairs to avoid duplicates

        print(f"Finding similarities for {len(query_documents)} query documents...")
        print(f"  Intra-language threshold: {similarity_threshold}")
        print(f"  Cross-language threshold: {cross_language_threshold}")

        # Helper function to determine if a file is Python or Go
        def is_python_file(file_path: str) -> bool:
            return str(file_path).endswith(".py")

        def is_go_file(file_path: str) -> bool:
            return str(file_path).endswith(".go")

        def is_cross_language(file1: str, file2: str) -> bool:
            """Check if two files are from different languages"""
            return (is_python_file(file1) and is_go_file(file2)) or (
                is_go_file(file1) and is_python_file(file2)
            )

        # Create embeddings for query documents
        query_texts = [doc.combined_text for doc in query_documents]
        query_embeddings = self.model.encode(query_texts, show_progress_bar=True)
        faiss.normalize_L2(query_embeddings)

        # Search for similarities
        scores, indices = self.faiss_index.search(query_embeddings.astype("float32"), top_k + 1)

        cross_lang_count = 0
        intra_lang_count = 0

        for i, (query_doc, doc_scores, doc_indices) in enumerate(
            zip(query_documents, scores, indices)
        ):
            for j, (score, idx) in enumerate(zip(doc_scores, doc_indices)):
                matched_doc = self.test_documents[idx]

                # Skip self-matches
                if query_doc.test_id == matched_doc.test_id:
                    continue

                # Skip same-file matches if requested
                if (
                    exclude_same_file
                    and query_doc.context.file_path == matched_doc.context.file_path
                ):
                    continue

                # Determine if this is a cross-language match
                is_cross = is_cross_language(
                    query_doc.context.file_path, matched_doc.context.file_path
                )

                # Apply appropriate threshold
                threshold = cross_language_threshold if is_cross else similarity_threshold
                if score < threshold:
                    continue

                # Skip duplicate pairs (A->B and B->A) - keep only one direction
                pair_key = tuple(sorted([query_doc.test_id, matched_doc.test_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Track match type
                if is_cross:
                    cross_lang_count += 1
                else:
                    intra_lang_count += 1

                # Calculate additional similarity signals
                context_similarity = self._calculate_context_similarity(
                    query_doc.context, matched_doc.context
                )

                result = {
                    "query_test_id": query_doc.test_id,
                    "query_description": query_doc.context.test_description,
                    "query_file": query_doc.context.file_path,
                    "matched_test_id": matched_doc.test_id,
                    "matched_description": matched_doc.context.test_description,
                    "matched_file": matched_doc.context.file_path,
                    "semantic_similarity": float(score),
                    "context_similarity": context_similarity,
                    "query_containers": query_doc.context.containers,
                    "matched_containers": matched_doc.context.containers,
                    "shared_labels": list(
                        set(query_doc.context.test_labels) & set(matched_doc.context.test_labels)
                    ),
                    "is_cross_language": is_cross,
                }
                results.append(result)

        print(
            f"  Found {intra_lang_count} intra-language matches and {cross_lang_count} cross-language matches"
        )

        # Sort by similarity score
        results.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        return results

    def _calculate_context_similarity(self, ctx1: TestContext, ctx2: TestContext) -> float:
        """Calculate context-based similarity score"""
        score = 0.0

        # Container hierarchy similarity
        if ctx1.containers and ctx2.containers:
            common_containers = set(ctx1.containers) & set(ctx2.containers)
            max_containers = max(len(ctx1.containers), len(ctx2.containers))
            score += 0.3 * (len(common_containers) / max_containers)

        # Label similarity
        if ctx1.test_labels and ctx2.test_labels:
            common_labels = set(ctx1.test_labels) & set(ctx2.test_labels)
            max_labels = max(len(ctx1.test_labels), len(ctx2.test_labels))
            score += 0.2 * (len(common_labels) / max_labels)

        # Step count similarity (structural)
        steps1 = len(ctx1.test_steps)
        steps2 = len(ctx2.test_steps)
        if steps1 > 0 and steps2 > 0:
            step_ratio = min(steps1, steps2) / max(steps1, steps2)
            score += 0.1 * step_ratio

        # File similarity (same file = higher context similarity)
        if ctx1.file_path == ctx2.file_path:
            score += 0.4

        return min(score, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Markdown-aware test similarity analysis")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL spec file")
    parser.add_argument("--markdown", help="Path to markdown specs directory")
    parser.add_argument("--output", required=True, help="Output CSV file for similarity results")
    parser.add_argument(
        "--repo-roots",
        nargs="*",
        default=[],
        help="Repository root directories for path normalization",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top K similar tests to find")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for intra-language matches (Python↔Python, Go↔Go) (default: 0.75)",
    )
    parser.add_argument(
        "--cross-language-threshold",
        type=float,
        default=0.65,
        help="Lower similarity threshold for cross-language matches (Python↔Go) (default: 0.65)",
    )
    parser.add_argument(
        "--exclude-same-file",
        action="store_true",
        default=True,
        help="Exclude matches within the same file (default: True)",
    )
    parser.add_argument(
        "--include-same-file",
        action="store_false",
        dest="exclude_same_file",
        help="Include matches within the same file",
    )

    args = parser.parse_args()

    def normalize_file_path(file_path: str, repo_roots: List[str]) -> str:
        """Normalize file path to start from repository name"""
        abs_path = os.path.abspath(file_path)

        for repo_root in repo_roots:
            abs_repo_root = os.path.abspath(repo_root)
            if abs_path.startswith(abs_repo_root):
                # Get the repository name (basename of the root)
                repo_name = os.path.basename(abs_repo_root)
                # Get the relative path from the repo root
                rel_path = os.path.relpath(abs_path, abs_repo_root)
                return f"{repo_name}/{rel_path}"

        # If no match found, return the original path
        return file_path

    # Initialize analyzer
    analyzer = MarkdownSimilarityAnalyzer(model_name=args.model)

    # Load test specifications
    print(f"Loading JSONL specs from {args.jsonl}")
    jsonl_contexts = analyzer.load_jsonl_specs(args.jsonl)
    print(f"Loaded {len(jsonl_contexts)} tests from JSONL")

    markdown_contexts = []
    if args.markdown:
        print(f"Loading markdown specs from {args.markdown}")
        markdown_contexts = analyzer.load_markdown_specs(args.markdown)
        print(f"Loaded {len(markdown_contexts)} tests from Markdown")

    # Use only one source to avoid duplicates
    # Prefer markdown contexts if available, otherwise use JSONL
    if markdown_contexts:
        print("Using Markdown contexts (preferred over JSONL to avoid duplicates)")
        all_contexts = markdown_contexts
    else:
        print("Using JSONL contexts (no Markdown available)")
        all_contexts = jsonl_contexts

    if not all_contexts:
        print("Error: No test specifications loaded")
        return 1

    # Create test documents
    print("Creating test documents for FAISS indexing...")
    test_documents = analyzer.create_test_documents(all_contexts)

    # Build FAISS index
    analyzer.build_faiss_index(test_documents)

    # Find similarities (using all tests as queries to find similar pairs)
    query_documents = test_documents  # All tests as queries
    similarities = analyzer.find_similar_tests(
        query_documents,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        cross_language_threshold=args.cross_language_threshold,
        exclude_same_file=args.exclude_same_file,
    )

    # Save results
    print(f"Found {len(similarities)} similar test pairs")
    if similarities:
        df = pd.DataFrame(similarities)

        # Normalize file paths if repo roots are provided
        if args.repo_roots:
            print(f"Normalizing file paths using repo roots: {args.repo_roots}")
            df["query_file"] = df["query_file"].apply(
                lambda x: normalize_file_path(x, args.repo_roots)
            )
            df["matched_file"] = df["matched_file"].apply(
                lambda x: normalize_file_path(x, args.repo_roots)
            )

        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

        # Print summary
        high_similarity = df[df["semantic_similarity"] > 0.9]
        print(f"High similarity matches (>0.9): {len(high_similarity)}")
        print(f"Average semantic similarity: {df['semantic_similarity'].mean():.3f}")
    else:
        print("No similar tests found above threshold")

    return 0


if __name__ == "__main__":
    exit(main())
