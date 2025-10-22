#!/usr/bin/env python3

"""
Test Spec Inspector Tool

This tool takes a test source code file and generates all intermediate spec files,
showing the full transformation pipeline from source code to vector database representation.

Usage:
    python inspect_test_specs.py --file /path/to/test_file.py --output inspection_output/
    python inspect_test_specs.py --file /path/to/test_file.go --output inspection_output/
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any


def detect_language(file_path: str) -> str:
    """Returns 'go' or 'python' based on file extension"""
    ext = Path(file_path).suffix.lower()
    if ext == ".go":
        return "go"
    elif ext == ".py":
        return "python"
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Only .go and .py files are supported."
        )


def run_go_extractor(file_path: str, temp_dir: str) -> str:
    """Runs go-extractor/kubespec-go on the file and returns path to generated JSONL file"""
    # Ensure the Go binary exists
    go_binary = Path("go-extractor/kubespec-go")
    if not go_binary.exists():
        print("Building Go extractor...")
        result = subprocess.run(
            ["go", "build", "-o", "kubespec-go"],
            cwd="go-extractor",
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build Go extractor: {result.stderr}")

    # Run the extractor
    output_file = os.path.join(temp_dir, "specs.jsonl")
    result = subprocess.run(
        [str(go_binary.absolute()), "-root", os.path.dirname(file_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Go extractor failed: {result.stderr}")

    # Write output to file
    with open(output_file, "w") as f:
        f.write(result.stdout)

    return output_file


def run_python_extractor(file_path: str, temp_dir: str) -> str:
    """Runs py-extractor/extract_kubespec.py on the file and returns path to generated JSONL file"""
    output_file = os.path.join(temp_dir, "specs.jsonl")

    # Run the Python extractor
    result = subprocess.run(
        [
            sys.executable,
            "py-extractor/extract_kubespec.py",
            "--root",
            os.path.dirname(file_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Python extractor failed: {result.stderr}")

    # Write output to file
    with open(output_file, "w") as f:
        f.write(result.stdout)

    return output_file


def load_specs(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load specs from JSONL file"""
    specs = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))
    return specs


def spec_to_text(spec: Dict[str, Any]) -> str:
    """Generate text representation for a spec (imported from match/build_index_and_match.py)"""
    try:
        # Import the spec_to_text function from the matching module
        sys.path.append("match")
        from build_index_and_match import spec_to_text as original_spec_to_text

        return original_spec_to_text(spec)
    except ImportError as e:
        print(f"Warning: Could not import spec_to_text from match module: {e}")
        print("Using simplified text representation...")
        return generate_simplified_text_representation(spec)


def generate_simplified_text_representation(spec: Dict[str, Any]) -> str:
    """Generate a simplified text representation when the full module is not available"""
    parts = []

    # Basic test information
    test_id = spec.get("test_id", "")
    if test_id:
        parts.append(f"test_id:{test_id}")

    test_type = spec.get("test_type", "")
    if test_type:
        parts.append(f"test_type:{test_type}")

    # Technology
    tech = spec.get("tech", [])
    if tech:
        parts.append(f"tech:{','.join(tech)}")

    # Purpose
    purpose = spec.get("purpose", "")
    if purpose:
        parts.append(f"purpose:{purpose}")

    # Environment
    environment = spec.get("environment", [])
    if environment:
        parts.append(f"env:{','.join(environment)}")

    # Dependencies
    dependencies = spec.get("dependencies", [])
    parts.extend(dependencies)

    # Actions
    actions = spec.get("actions", [])
    for action in actions:
        gvk = action.get("gvk", "")
        verb = action.get("verb", "")
        if gvk and verb:
            parts.append(f"{gvk}:{verb}")
        elif gvk:
            parts.append(gvk)
        elif verb:
            parts.append(f"verb:{verb}")

    # Expectations
    expectations = spec.get("expectations", [])
    for exp in expectations:
        target = exp.get("target", "")
        condition = exp.get("condition", "")
        if target and condition:
            parts.append(f"expect:{target}={condition}")

    # By steps
    by_steps = spec.get("by_steps", [])
    for step in by_steps:
        description = step.get("description", "")
        if description:
            parts.append(f"step:{description}")

        step_actions = step.get("actions", [])
        for action in step_actions:
            gvk = action.get("gvk", "")
            verb = action.get("verb", "")
            if gvk and verb:
                parts.append(f"step_{gvk}:{verb}")
            elif gvk:
                parts.append(f"step_{gvk}")
            elif verb:
                parts.append(f"step_verb:{verb}")

    return "\n".join(parts)


def generate_text_representations(specs: List[Dict[str, Any]], output_dir: str):
    """Generate text representations for all specs and write to individual files"""
    text_dir = os.path.join(output_dir, "text_representations")
    os.makedirs(text_dir, exist_ok=True)

    for spec in specs:
        test_id = spec.get("test_id", "unknown")
        # Sanitize test_id for filename
        safe_filename = "".join(
            c for c in test_id if c.isalnum() or c in ("_", "-", ".")
        ).rstrip()
        if not safe_filename:
            safe_filename = "unknown_test"

        # Generate text representation
        text_content = spec_to_text(spec)

        # Write to file
        text_file = os.path.join(text_dir, f"{safe_filename}.txt")
        with open(text_file, "w") as f:
            f.write(text_content)

        # Store the text file path in the spec for later reference
        spec["_text_file"] = f"text_representations/{safe_filename}.txt"


def create_inspection_report(
    specs: List[Dict[str, Any]], output_dir: str, source_file: str, language: str
):
    """Create index.json and summary.md with metadata and statistics"""

    # Create index.json
    index_data = {
        "source_file": source_file,
        "language": language,
        "total_tests": len(specs),
        "tests": [],
    }

    for spec in specs:
        test_info = {
            "test_id": spec.get("test_id", "unknown"),
            "text_file": spec.get("_text_file", ""),
            "purpose": spec.get("purpose", ""),
            "tech": spec.get("tech", []),
            "actions_count": len(spec.get("actions", [])),
            "by_steps_count": len(spec.get("by_steps", [])),
            "operations": [],
        }

        # Extract operations
        for action in spec.get("actions", []):
            gvk = action.get("gvk", "")
            verb = action.get("verb", "")
            if gvk and verb:
                test_info["operations"].append(f"{gvk}:{verb}")
            elif gvk:
                test_info["operations"].append(gvk)

        index_data["tests"].append(test_info)

    # Write index.json
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    # Create summary.md
    summary_content = f"""# Test Spec Inspection Report

**Source File**: {source_file}
**Language**: {language.title()}
**Total Tests Found**: {len(specs)}

## Tests Extracted

"""

    for i, spec in enumerate(specs, 1):
        test_id = spec.get("test_id", "unknown")
        purpose = spec.get("purpose", "UNKNOWN")
        tech = spec.get("tech", [])
        actions = spec.get("actions", [])
        by_steps = spec.get("by_steps", [])
        text_file = spec.get("_text_file", "")

        # Extract operations summary
        operations = []
        for action in actions:
            gvk = action.get("gvk", "")
            verb = action.get("verb", "")
            if gvk and verb:
                operations.append(f"{gvk}:{verb}")
            elif gvk:
                operations.append(gvk)

        summary_content += f"""### {i}. {test_id}
- **Purpose**: {purpose}
- **Technology**: {', '.join(tech) if tech else 'None'}
- **Operations**: {', '.join(operations[:5])}{'...' if len(operations) > 5 else ''}
- **Actions Count**: {len(actions)}
- **By Steps Count**: {len(by_steps)}
- **Text File**: {text_file}

"""

    # Add statistics
    all_purposes = [spec.get("purpose", "UNKNOWN") for spec in specs]
    all_tech = []
    for spec in specs:
        all_tech.extend(spec.get("tech", []))

    purpose_counts = {}
    for purpose in all_purposes:
        purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1

    tech_counts = {}
    for tech in all_tech:
        tech_counts[tech] = tech_counts.get(tech, 0) + 1

    summary_content += f"""## Statistics

### Purpose Distribution
"""
    for purpose, count in sorted(purpose_counts.items()):
        summary_content += f"- **{purpose}**: {count} tests\n"

    if tech_counts:
        summary_content += f"""
### Technology Distribution
"""
        for tech, count in sorted(tech_counts.items()):
            summary_content += f"- **{tech}**: {count} tests\n"

    # Write summary.md
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write(summary_content)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect test spec extraction pipeline from source code to vector database representation"
    )
    parser.add_argument(
        "--file", required=True, help="Path to test source file (.py or .go)"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for inspection results"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.file):
        print(f"Error: File does not exist: {args.file}")
        sys.exit(1)

    # Detect language
    try:
        language = detect_language(args.file)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"Inspecting {language} test file: {args.file}")
    print(f"Output directory: {args.output}")

    # Use temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Run appropriate extractor
            if language == "go":
                print("Running Go extractor...")
                specs_file = run_go_extractor(args.file, temp_dir)
            else:  # python
                print("Running Python extractor...")
                specs_file = run_python_extractor(args.file, temp_dir)

            # Load specs
            print("Loading extracted specs...")
            specs = load_specs(specs_file)

            if not specs:
                print("Warning: No specs extracted from the file")
                return

            print(f"Found {len(specs)} test specs")

            # Copy specs.jsonl to output directory
            import shutil

            shutil.copy2(specs_file, os.path.join(args.output, "specs.jsonl"))

            # Generate text representations
            print("Generating text representations...")
            generate_text_representations(specs, args.output)

            # Create inspection report
            print("Creating inspection report...")
            create_inspection_report(specs, args.output, args.file, language)

            print(f"\nInspection complete! Results saved to: {args.output}")
            print(f"Files generated:")
            print(f"  - specs.jsonl (raw extracted specs)")
            print(f"  - text_representations/ (text sent to vector DB)")
            print(f"  - index.json (metadata)")
            print(f"  - summary.md (human-readable summary)")

        except Exception as e:
            print(f"Error during inspection: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
