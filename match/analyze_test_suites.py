#!/usr/bin/env python3
"""
Test Suite Analyzer

Generates high-level descriptions of test suites based on extracted KubeSpecs.
Analyzes coverage patterns, identifies gaps, and provides comprehensive insights.
"""

import json
import argparse
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import sys


def load_specs(spec_file):
    """Load test specifications from JSONL file."""
    specs = []
    with open(spec_file, "r") as f:
        for line in f:
            if line.strip():
                specs.append(json.loads(line))
    return specs


def analyze_test_suite(specs, suite_name):
    """Generate comprehensive analysis of a test suite."""
    analysis = {
        "suite_name": suite_name,
        "total_tests": len(specs),
        "test_types": Counter(),
        "purposes": Counter(),
        "environments": Counter(),
        "dependencies": Counter(),
        "operations": Counter(),
        "resources": Counter(),
        "verbs": Counter(),
        "expectation_targets": Counter(),
        "openshift_resources": Counter(),
        "artifacts": Counter(),
        "concurrency_patterns": Counter(),
        "coverage_metrics": {},
        "test_distribution": {},
        "key_insights": [],
    }

    if not specs:
        return analysis

    # Analyze each test
    for spec in specs:
        # Test types
        if spec.get("test_type"):
            analysis["test_types"][spec["test_type"]] += 1

        # Purposes
        if spec.get("purpose"):
            analysis["purposes"][spec["purpose"]] += 1

        # Environments
        for env in spec.get("environment") or []:
            analysis["environments"][env] += 1

        # Dependencies
        for dep in spec.get("dependencies") or []:
            analysis["dependencies"][dep] += 1

        # Operations and resources
        for action in spec.get("actions") or []:
            if "gvk" in action and "verb" in action:
                operation = f"{action['gvk']}:{action['verb']}"
                analysis["operations"][operation] += 1
                analysis["resources"][action["gvk"]] += 1
                analysis["verbs"][action["verb"]] += 1

        # Expectation targets
        for exp in spec.get("expectations") or []:
            if "target" in exp:
                analysis["expectation_targets"][exp["target"]] += 1

        # OpenShift resources
        for resource in spec.get("openshift_specific") or []:
            analysis["openshift_resources"][resource] += 1

        # Artifacts
        for artifact in spec.get("artifacts") or []:
            analysis["artifacts"][artifact] += 1

        # Concurrency patterns
        for pattern in spec.get("concurrency") or []:
            analysis["concurrency_patterns"][pattern] += 1

    # Calculate coverage metrics
    total_operations = sum(analysis["operations"].values())
    unique_operations = len(analysis["operations"])
    unique_resources = len(analysis["resources"])

    analysis["coverage_metrics"] = {
        "total_operations": total_operations,
        "unique_operations": unique_operations,
        "unique_resources": unique_resources,
        "avg_operations_per_test": total_operations / len(specs) if specs else 0,
        "test_type_diversity": len(analysis["test_types"]),
        "purpose_diversity": len(analysis["purposes"]),
        "environment_diversity": len(analysis["environments"]),
    }

    # Test distribution analysis
    analysis["test_distribution"] = {
        "by_type": dict(analysis["test_types"].most_common()),
        "by_purpose": dict(analysis["purposes"].most_common()),
        "by_environment": dict(analysis["environments"].most_common()),
    }

    # Generate key insights
    analysis["key_insights"] = generate_insights(analysis)

    return analysis


def generate_insights(analysis):
    """Generate key insights about the test suite."""
    insights = []

    # Test type insights
    if analysis["test_types"]:
        most_common_type = analysis["test_types"].most_common(1)[0]
        insights.append(
            f"Primary test type: {most_common_type[0]} ({most_common_type[1]} tests, {most_common_type[1]/analysis['total_tests']*100:.1f}%)"
        )

    # Purpose insights
    if analysis["purposes"]:
        most_common_purpose = analysis["purposes"].most_common(1)[0]
        insights.append(
            f"Main focus: {most_common_purpose[0]} ({most_common_purpose[1]} tests, {most_common_purpose[1]/analysis['total_tests']*100:.1f}%)"
        )

    # Resource insights
    if analysis["resources"]:
        top_resources = analysis["resources"].most_common(3)
        resource_list = ", ".join([f"{res} ({count})" for res, count in top_resources])
        insights.append(f"Most tested resources: {resource_list}")

    # Operation insights
    if analysis["verbs"]:
        top_verbs = analysis["verbs"].most_common(3)
        verb_list = ", ".join([f"{verb} ({count})" for verb, count in top_verbs])
        insights.append(f"Most common operations: {verb_list}")

    # Environment insights
    if analysis["environments"]:
        env_list = ", ".join(
            [f"{env} ({count})" for env, count in analysis["environments"].most_common(3)]
        )
        insights.append(f"Target environments: {env_list}")

    # Coverage insights
    if analysis["coverage_metrics"]["unique_operations"] > 0:
        insights.append(
            f"Operation coverage: {analysis['coverage_metrics']['unique_operations']} unique operations across {analysis['coverage_metrics']['unique_resources']} resource types"
        )

    # OpenShift insights
    if analysis["openshift_resources"]:
        openshift_count = sum(analysis["openshift_resources"].values())
        insights.append(
            f"OpenShift-specific: {openshift_count} operations using {len(analysis['openshift_resources'])} OpenShift resources"
        )

    return insights


def compare_test_suites(analysis1, analysis2):
    """Compare two test suites and identify gaps."""
    comparison = {
        "suite1": analysis1["suite_name"],
        "suite2": analysis2["suite_name"],
        "test_count_diff": analysis1["total_tests"] - analysis2["total_tests"],
        "operation_gaps": {},
        "resource_gaps": {},
        "purpose_gaps": {},
        "environment_gaps": {},
        "coverage_comparison": {},
        "unique_to_suite1": {},
        "unique_to_suite2": {},
        "recommendations": [],
    }

    # Operation gaps
    ops1 = set(analysis1["operations"].keys())
    ops2 = set(analysis2["operations"].keys())
    comparison["operation_gaps"] = {
        "unique_to_suite1": list(ops1 - ops2),
        "unique_to_suite2": list(ops2 - ops1),
        "common": list(ops1 & ops2),
    }

    # Resource gaps
    res1 = set(analysis1["resources"].keys())
    res2 = set(analysis2["resources"].keys())
    comparison["resource_gaps"] = {
        "unique_to_suite1": list(res1 - res2),
        "unique_to_suite2": list(res2 - res1),
        "common": list(res1 & res2),
    }

    # Purpose gaps
    purp1 = set(analysis1["purposes"].keys())
    purp2 = set(analysis2["purposes"].keys())
    comparison["purpose_gaps"] = {
        "unique_to_suite1": list(purp1 - purp2),
        "unique_to_suite2": list(purp2 - purp1),
        "common": list(purp1 & purp2),
    }

    # Environment gaps
    env1 = set(analysis1["environments"].keys())
    env2 = set(analysis2["environments"].keys())
    comparison["environment_gaps"] = {
        "unique_to_suite1": list(env1 - env2),
        "unique_to_suite2": list(env2 - env1),
        "common": list(env1 & env2),
    }

    # Coverage comparison
    comparison["coverage_comparison"] = {
        "suite1_operations": analysis1["coverage_metrics"]["unique_operations"],
        "suite2_operations": analysis2["coverage_metrics"]["unique_operations"],
        "suite1_resources": analysis1["coverage_metrics"]["unique_resources"],
        "suite2_resources": analysis2["coverage_metrics"]["unique_resources"],
    }

    # Generate recommendations
    comparison["recommendations"] = generate_recommendations(comparison)

    return comparison


def generate_recommendations(comparison):
    """Generate recommendations based on suite comparison."""
    recommendations = []

    # Operation recommendations
    if comparison["operation_gaps"]["unique_to_suite1"]:
        recommendations.append(
            f"Consider adding {len(comparison['operation_gaps']['unique_to_suite1'])} operations from {comparison['suite1']} to {comparison['suite2']}"
        )

    if comparison["operation_gaps"]["unique_to_suite2"]:
        recommendations.append(
            f"Consider adding {len(comparison['operation_gaps']['unique_to_suite2'])} operations from {comparison['suite2']} to {comparison['suite1']}"
        )

    # Resource recommendations
    if comparison["resource_gaps"]["unique_to_suite1"]:
        recommendations.append(
            f"Consider adding tests for {len(comparison['resource_gaps']['unique_to_suite1'])} resource types from {comparison['suite1']} to {comparison['suite2']}"
        )

    if comparison["resource_gaps"]["unique_to_suite2"]:
        recommendations.append(
            f"Consider adding tests for {len(comparison['resource_gaps']['unique_to_suite2'])} resource types from {comparison['suite2']} to {comparison['suite1']}"
        )

    # Purpose recommendations
    if comparison["purpose_gaps"]["unique_to_suite1"]:
        recommendations.append(
            f"Consider adding {len(comparison['purpose_gaps']['unique_to_suite1'])} purpose categories from {comparison['suite1']} to {comparison['suite2']}"
        )

    if comparison["purpose_gaps"]["unique_to_suite2"]:
        recommendations.append(
            f"Consider adding {len(comparison['purpose_gaps']['unique_to_suite2'])} purpose categories from {comparison['suite2']} to {comparison['suite1']}"
        )

    # Coverage recommendations
    if (
        comparison["coverage_comparison"]["suite1_operations"]
        > comparison["coverage_comparison"]["suite2_operations"]
    ):
        recommendations.append(
            f"{comparison['suite1']} has {comparison['coverage_comparison']['suite1_operations'] - comparison['coverage_comparison']['suite2_operations']} more unique operations than {comparison['suite2']}"
        )
    elif (
        comparison["coverage_comparison"]["suite2_operations"]
        > comparison["coverage_comparison"]["suite1_operations"]
    ):
        recommendations.append(
            f"{comparison['suite2']} has {comparison['coverage_comparison']['suite2_operations'] - comparison['coverage_comparison']['suite1_operations']} more unique operations than {comparison['suite1']}"
        )

    return recommendations


def print_suite_analysis(analysis):
    """Print formatted analysis of a test suite."""
    print(f"\n{'='*60}")
    print(f"TEST SUITE ANALYSIS: {analysis['suite_name'].upper()}")
    print(f"{'='*60}")

    print(f"\n📊 OVERVIEW:")
    print(f"  Total Tests: {analysis['total_tests']}")
    print(f"  Unique Operations: {analysis['coverage_metrics']['unique_operations']}")
    print(f"  Unique Resources: {analysis['coverage_metrics']['unique_resources']}")
    print(f"  Avg Operations/Test: {analysis['coverage_metrics']['avg_operations_per_test']:.1f}")

    print(f"\n🎯 TEST TYPES:")
    for test_type, count in analysis["test_distribution"]["by_type"].items():
        percentage = count / analysis["total_tests"] * 100
        print(f"  {test_type}: {count} ({percentage:.1f}%)")

    print(f"\n🎯 PURPOSES:")
    for purpose, count in analysis["test_distribution"]["by_purpose"].items():
        percentage = count / analysis["total_tests"] * 100
        print(f"  {purpose}: {count} ({percentage:.1f}%)")

    print(f"\n🌍 ENVIRONMENTS:")
    for env, count in analysis["test_distribution"]["by_environment"].items():
        print(f"  {env}: {count}")

    print(f"\n🔧 TOP RESOURCES:")
    for resource, count in analysis["resources"].most_common(5):
        print(f"  {resource}: {count}")

    print(f"\n⚡ TOP OPERATIONS:")
    for operation, count in analysis["operations"].most_common(5):
        print(f"  {operation}: {count}")

    print(f"\n💡 KEY INSIGHTS:")
    for insight in analysis["key_insights"]:
        print(f"  • {insight}")


def print_suite_comparison(comparison):
    """Print formatted comparison between two test suites."""
    print(f"\n{'='*60}")
    print(f"SUITE COMPARISON: {comparison['suite1']} vs {comparison['suite2']}")
    print(f"{'='*60}")

    print(f"\n📊 OVERVIEW:")
    print(f"  Test Count Difference: {comparison['test_count_diff']:+d}")
    print(
        f"  {comparison['suite1']} Operations: {comparison['coverage_comparison']['suite1_operations']}"
    )
    print(
        f"  {comparison['suite2']} Operations: {comparison['coverage_comparison']['suite2_operations']}"
    )
    print(
        f"  {comparison['suite1']} Resources: {comparison['coverage_comparison']['suite1_resources']}"
    )
    print(
        f"  {comparison['suite2']} Resources: {comparison['coverage_comparison']['suite2_resources']}"
    )

    print(f"\n🔍 OPERATION GAPS:")
    if comparison["operation_gaps"]["unique_to_suite1"]:
        print(
            f"  Unique to {comparison['suite1']} ({len(comparison['operation_gaps']['unique_to_suite1'])}):"
        )
        for op in comparison["operation_gaps"]["unique_to_suite1"][:5]:
            print(f"    • {op}")
        if len(comparison["operation_gaps"]["unique_to_suite1"]) > 5:
            print(f"    ... and {len(comparison['operation_gaps']['unique_to_suite1']) - 5} more")

    if comparison["operation_gaps"]["unique_to_suite2"]:
        print(
            f"  Unique to {comparison['suite2']} ({len(comparison['operation_gaps']['unique_to_suite2'])}):"
        )
        for op in comparison["operation_gaps"]["unique_to_suite2"][:5]:
            print(f"    • {op}")
        if len(comparison["operation_gaps"]["unique_to_suite2"]) > 5:
            print(f"    ... and {len(comparison['operation_gaps']['unique_to_suite2']) - 5} more")

    print(f"\n🎯 PURPOSE GAPS:")
    if comparison["purpose_gaps"]["unique_to_suite1"]:
        print(
            f"  Unique to {comparison['suite1']}: {', '.join(comparison['purpose_gaps']['unique_to_suite1'])}"
        )
    if comparison["purpose_gaps"]["unique_to_suite2"]:
        print(
            f"  Unique to {comparison['suite2']}: {', '.join(comparison['purpose_gaps']['unique_to_suite2'])}"
        )

    print(f"\n🌍 ENVIRONMENT GAPS:")
    if comparison["environment_gaps"]["unique_to_suite1"]:
        print(
            f"  Unique to {comparison['suite1']}: {', '.join(comparison['environment_gaps']['unique_to_suite1'])}"
        )
    if comparison["environment_gaps"]["unique_to_suite2"]:
        print(
            f"  Unique to {comparison['suite2']}: {', '.join(comparison['environment_gaps']['unique_to_suite2'])}"
        )

    print(f"\n💡 RECOMMENDATIONS:")
    for rec in comparison["recommendations"]:
        print(f"  • {rec}")


def analyze_individual_suites(specs, suite_type):
    """Analyze individual test suites from combined specs."""
    suites = {}

    # Group specs by repository (extracted from test_id)
    for spec in specs:
        test_id = spec.get("test_id", "")
        if ":" in test_id:
            # Extract repository name from test_id (e.g., "eco-gotests/path/file.go:TestFunc")
            repo_name = test_id.split("/")[0]
            if repo_name not in suites:
                suites[repo_name] = []
            suites[repo_name].append(spec)

    return suites


def main():
    parser = argparse.ArgumentParser(description="Analyze test suites and identify gaps")
    parser.add_argument("--go", help="Go specs JSONL file")
    parser.add_argument("--py", help="Python specs JSONL file")
    parser.add_argument("--output", help="Output directory for analysis results")
    parser.add_argument("--compare", action="store_true", help="Compare all test suites")

    args = parser.parse_args()

    if not args.go and not args.py:
        print("Error: At least one spec file (--go or --py) is required")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(".")
    output_dir.mkdir(exist_ok=True)

    all_analyses = {}

    # Analyze Go suites individually
    if args.go:
        print("Analyzing Go test suites...")
        go_specs = load_specs(args.go)
        go_suites = analyze_individual_suites(go_specs, "Go")

        for suite_name, specs in go_suites.items():
            print(f"\n--- Analyzing {suite_name} ---")
            analysis = analyze_test_suite(specs, suite_name)
            all_analyses[suite_name] = analysis
            print_suite_analysis(analysis)

            # Save individual suite analysis
            with open(output_dir / f"{suite_name}_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)

    # Analyze Python suites individually
    if args.py:
        print("\nAnalyzing Python test suites...")
        py_specs = load_specs(args.py)
        py_suites = analyze_individual_suites(py_specs, "Python")

        for suite_name, specs in py_suites.items():
            print(f"\n--- Analyzing {suite_name} ---")
            analysis = analyze_test_suite(specs, suite_name)
            all_analyses[suite_name] = analysis
            print_suite_analysis(analysis)

            # Save individual suite analysis
            with open(output_dir / f"{suite_name}_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)

    # Compare all suites if requested
    if args.compare and len(all_analyses) > 1:
        print(f"\n{'='*60}")
        print("COMPREHENSIVE SUITE COMPARISON")
        print(f"{'='*60}")

        suite_names = list(all_analyses.keys())
        comparisons = []

        # Compare each pair of suites
        for i in range(len(suite_names)):
            for j in range(i + 1, len(suite_names)):
                suite1_name = suite_names[i]
                suite2_name = suite_names[j]

                print(f"\n--- Comparing {suite1_name} vs {suite2_name} ---")
                comparison = compare_test_suites(
                    all_analyses[suite1_name], all_analyses[suite2_name]
                )
                comparisons.append(comparison)
                print_suite_comparison(comparison)

        # Save all comparisons
        with open(output_dir / "all_suite_comparisons.json", "w") as f:
            json.dump(comparisons, f, indent=2, default=str)

    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    print(f"Analyzed {len(all_analyses)} individual test suites")


if __name__ == "__main__":
    main()
