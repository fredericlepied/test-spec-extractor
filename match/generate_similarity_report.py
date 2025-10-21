#!/usr/bin/env python3
"""
Similarity Matches Report Generator

Generates comprehensive reports analyzing test similarity matches,
identifying potential duplicates, complementary tests, and coverage gaps.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import sys
from datetime import datetime
from collections import defaultdict, Counter
import re


def load_similarity_data(test_report_file, test_coverage_file):
    """Load similarity matching data from CSV files."""
    test_report = pd.read_csv(test_report_file)
    test_coverage = pd.read_csv(test_coverage_file)
    return test_report, test_coverage


def load_test_specs(go_specs_file, py_specs_file):
    """Load test specifications from JSONL files."""
    go_specs = []
    py_specs = []

    with open(go_specs_file, "r") as f:
        for line in f:
            go_specs.append(json.loads(line.strip()))

    with open(py_specs_file, "r") as f:
        for line in f:
            py_specs.append(json.loads(line.strip()))

    return go_specs, py_specs


def load_all_specs(go_specs_file, py_specs_file):
    """Load and combine all test specifications."""
    go_specs = []
    py_specs = []

    with open(go_specs_file, "r") as f:
        for line in f:
            spec = json.loads(line.strip())
            spec["_language"] = "go"
            go_specs.append(spec)

    with open(py_specs_file, "r") as f:
        for line in f:
            spec = json.loads(line.strip())
            spec["_language"] = "py"
            py_specs.append(spec)

    all_specs = go_specs + py_specs
    return all_specs, go_specs, py_specs


def format_number(num):
    """Format numbers with commas for readability."""
    return f"{num:,}"


def format_percentage(value, total):
    """Format percentage with 1 decimal place."""
    if total == 0:
        return "0.0%"
    return f"{(value / total * 100):.1f}%"


def analyze_similarity_patterns(test_report):
    """Analyze patterns in similarity matches."""
    patterns = {
        "total_matches": len(test_report),
        "high_similarity": len(test_report[test_report["blended_score"] >= 0.8]),
        "medium_similarity": len(
            test_report[
                (test_report["blended_score"] >= 0.6) & (test_report["blended_score"] < 0.8)
            ]
        ),
        "low_similarity": len(test_report[test_report["blended_score"] < 0.6]),
        "perfect_matches": len(test_report[test_report["blended_score"] == 1.0]),
        "avg_score": test_report["blended_score"].mean(),
        "score_std": test_report["blended_score"].std(),
    }

    # Score distribution
    score_ranges = [
        (0.9, 1.0, "Very High"),
        (0.8, 0.9, "High"),
        (0.7, 0.8, "Medium-High"),
        (0.6, 0.7, "Medium"),
        (0.5, 0.6, "Low-Medium"),
        (0.0, 0.5, "Low"),
    ]

    distribution = {}
    for min_score, max_score, label in score_ranges:
        count = len(
            test_report[
                (test_report["blended_score"] >= min_score)
                & (test_report["blended_score"] < max_score)
            ]
        )
        distribution[label] = count

    patterns["score_distribution"] = distribution
    return patterns


def analyze_shared_signals(test_report):
    """Analyze shared signals patterns."""
    signal_analysis = {
        "exact_matches": 0,
        "resource_matches": 0,
        "category_matches": 0,
        "verb_group_matches": 0,
        "purpose_matches": 0,
        "tech_matches": 0,
        "common_patterns": [],
    }

    # Count different types of shared signals
    for _, row in test_report.iterrows():
        shared_signals = str(row.get("shared_signals", ""))

        if "exact:" in shared_signals:
            signal_analysis["exact_matches"] += 1
        if "resource:" in shared_signals:
            signal_analysis["resource_matches"] += 1
        if "category:" in shared_signals:
            signal_analysis["category_matches"] += 1
        if "verb_group:" in shared_signals:
            signal_analysis["verb_group_matches"] += 1
        if "purpose:" in shared_signals:
            signal_analysis["purpose_matches"] += 1
        if "tech:" in shared_signals:
            signal_analysis["tech_matches"] += 1

    # Find common shared signal patterns
    signal_patterns = Counter()
    for _, row in test_report.iterrows():
        shared_signals = str(row.get("shared_signals", ""))
        if shared_signals and shared_signals != "nan":
            # Extract individual signal types
            signals = shared_signals.split(";")
            for signal in signals:
                if ":" in signal:
                    signal_type = signal.split(":")[0]
                    signal_patterns[signal_type] += 1

    signal_analysis["common_patterns"] = dict(signal_patterns.most_common(10))
    return signal_analysis


def identify_potential_duplicates(test_report, go_specs, py_specs, threshold=0.95):
    """Identify potential duplicate tests."""
    duplicates = []

    for _, row in test_report.iterrows():
        if row["blended_score"] >= threshold:
            go_idx = int(row["idx_a"])
            py_idx = int(row["idx_b"])

            if go_idx < len(go_specs) and py_idx < len(py_specs):
                go_test = go_specs[go_idx]
                py_test = py_specs[py_idx]

                duplicates.append(
                    {
                        "go_test": go_test["test_id"],
                        "py_test": py_test["test_id"],
                        "score": row["blended_score"],
                        "shared_signals": row["shared_signals"],
                        "go_purpose": go_test.get("purpose", "Unknown"),
                        "py_purpose": py_test.get("purpose", "Unknown"),
                        "go_actions": len(go_test.get("actions", [])),
                        "py_actions": len(py_test.get("actions", [])),
                    }
                )

    return duplicates


def identify_complementary_tests(test_report, go_specs, py_specs, score_range=(0.6, 0.8)):
    """Identify complementary tests that could benefit from cross-pollination."""
    complementary = []

    for _, row in test_report.iterrows():
        if score_range[0] <= row["blended_score"] < score_range[1]:
            go_idx = int(row["idx_a"])
            py_idx = int(row["idx_b"])

            if go_idx < len(go_specs) and py_idx < len(py_specs):
                go_test = go_specs[go_idx]
                py_test = py_specs[py_idx]

                # Check if they have different purposes but similar operations
                go_purpose = go_test.get("purpose", "Unknown")
                py_purpose = py_test.get("purpose", "Unknown")

                if go_purpose != py_purpose:
                    complementary.append(
                        {
                            "go_test": go_test["test_id"],
                            "py_test": py_test["test_id"],
                            "score": row["blended_score"],
                            "go_purpose": go_purpose,
                            "py_purpose": py_purpose,
                            "shared_signals": row["shared_signals"],
                            "suggestion": f"Consider combining {go_purpose} and {py_purpose} approaches",
                        }
                    )

    return complementary


def generate_match_type_analysis(test_report):
    """Generate analysis of match types (Go->Go, Go->Py, Py->Go, Py->Py)."""
    analysis = []
    analysis.append("## 🔄 Match Type Analysis")
    analysis.append("")

    if "match_type" in test_report.columns:
        match_types = test_report["match_type"].value_counts()
        total_matches = len(test_report)

        analysis.append("**Match Type Distribution:**")
        for match_type, count in match_types.items():
            percentage = format_percentage(count, total_matches)
            analysis.append(f"- **{match_type}**: {format_number(count)} matches ({percentage})")
        analysis.append("")

        # Analyze intra-language vs cross-language matches
        intra_language = 0
        cross_language = 0

        for match_type, count in match_types.items():
            if "->" in match_type:
                lang_a, lang_b = match_type.split("->")
                if lang_a == lang_b:
                    intra_language += count
                else:
                    cross_language += count

        analysis.append("**Language Analysis:**")
        analysis.append(
            f"- **Intra-language matches**: {format_number(intra_language)} ({format_percentage(intra_language, total_matches)})"
        )
        analysis.append(
            f"- **Cross-language matches**: {format_number(cross_language)} ({format_percentage(cross_language, total_matches)})"
        )
        analysis.append("")

        # Insights
        if intra_language > cross_language:
            analysis.append(
                "💡 **Insight**: More intra-language matches found - potential for test consolidation within same language"
            )
        elif cross_language > intra_language:
            analysis.append(
                "💡 **Insight**: More cross-language matches found - good cross-pollination opportunities"
            )
        else:
            analysis.append("💡 **Insight**: Balanced intra and cross-language matches")
        analysis.append("")
    else:
        analysis.append("ℹ️ **Match type information not available in the data**")
        analysis.append("")

    return "\n".join(analysis)


def generate_executive_summary(patterns, signal_analysis, duplicates, complementary):
    """Generate executive summary for similarity analysis."""
    summary = []
    summary.append("## 📊 Similarity Analysis Executive Summary")
    summary.append("")

    # Overall statistics
    summary.append("**Match Overview:**")
    summary.append(f"- **Total Matches:** {format_number(patterns['total_matches'])}")
    summary.append(
        f"- **High Similarity (≥0.8):** {format_number(patterns['high_similarity'])} ({format_percentage(patterns['high_similarity'], patterns['total_matches'])})"
    )
    summary.append(
        f"- **Medium Similarity (0.6-0.8):** {format_number(patterns['medium_similarity'])} ({format_percentage(patterns['medium_similarity'], patterns['total_matches'])})"
    )
    summary.append(
        f"- **Low Similarity (<0.6):** {format_number(patterns['low_similarity'])} ({format_percentage(patterns['low_similarity'], patterns['total_matches'])})"
    )
    summary.append(f"- **Average Score:** {patterns['avg_score']:.3f}")
    summary.append("")

    # Quality indicators
    duplicate_ratio = (
        len(duplicates) / patterns["total_matches"] if patterns["total_matches"] > 0 else 0
    )
    complementary_ratio = (
        len(complementary) / patterns["total_matches"] if patterns["total_matches"] > 0 else 0
    )

    summary.append("**Quality Indicators:**")
    summary.append(
        f"- **Potential Duplicates:** {len(duplicates)} ({format_percentage(len(duplicates), patterns['total_matches'])})"
    )
    summary.append(
        f"- **Complementary Tests:** {len(complementary)} ({format_percentage(len(complementary), patterns['total_matches'])})"
    )
    summary.append(
        f"- **Exact Operation Matches:** {signal_analysis['exact_matches']} ({format_percentage(signal_analysis['exact_matches'], patterns['total_matches'])})"
    )
    summary.append("")

    # Recommendations
    if duplicate_ratio > 0.1:
        summary.append("⚠️ **High Duplicate Ratio**: Consider consolidating similar tests")
    elif duplicate_ratio < 0.05:
        summary.append("✅ **Low Duplicate Ratio**: Good test diversity")

    if complementary_ratio > 0.2:
        summary.append(
            "💡 **High Complementary Potential**: Many tests could benefit from cross-pollination"
        )

    if patterns["avg_score"] > 0.7:
        summary.append("🔍 **High Similarity**: Tests show strong functional overlap")
    elif patterns["avg_score"] < 0.4:
        summary.append("📈 **Low Similarity**: Tests are quite different, good coverage diversity")

    summary.append("")
    return "\n".join(summary)


def generate_score_distribution(patterns, test_report):
    """Generate score distribution analysis."""
    dist = []
    dist.append("## 📈 Similarity Score Distribution")
    dist.append("")

    dist.append("**Score Range Analysis:**")
    for label, count in patterns["score_distribution"].items():
        percentage = format_percentage(count, patterns["total_matches"])
        bar_length = (
            int((count / patterns["total_matches"]) * 50) if patterns["total_matches"] > 0 else 0
        )
        bar = "█" * bar_length + "░" * (50 - bar_length)
        dist.append(f"- **{label}**: {format_number(count)} ({percentage}) {bar}")

    dist.append("")

    # Statistical summary
    dist.append("**Statistical Summary:**")
    dist.append(f"- **Mean Score:** {patterns['avg_score']:.3f}")
    dist.append(f"- **Standard Deviation:** {patterns['score_std']:.3f}")
    dist.append(
        f"- **Score Range:** {test_report['blended_score'].min():.3f} - {test_report['blended_score'].max():.3f}"
    )
    dist.append("")

    return "\n".join(dist)


def generate_shared_signals_analysis(signal_analysis):
    """Generate shared signals analysis."""
    signals = []
    signals.append("## 🔗 Shared Signals Analysis")
    signals.append("")

    signals.append("**Signal Type Distribution:**")
    signals.append(f"- **Exact Matches:** {format_number(signal_analysis['exact_matches'])}")
    signals.append(f"- **Resource Matches:** {format_number(signal_analysis['resource_matches'])}")
    signals.append(f"- **Category Matches:** {format_number(signal_analysis['category_matches'])}")
    signals.append(
        f"- **Verb Group Matches:** {format_number(signal_analysis['verb_group_matches'])}"
    )
    signals.append(f"- **Purpose Matches:** {format_number(signal_analysis['purpose_matches'])}")
    signals.append(f"- **Technology Matches:** {format_number(signal_analysis['tech_matches'])}")
    signals.append("")

    signals.append("**Most Common Signal Types:**")
    for signal_type, count in signal_analysis["common_patterns"].items():
        signals.append(f"- **{signal_type}**: {format_number(count)} occurrences")
    signals.append("")

    return "\n".join(signals)


def generate_duplicate_analysis(duplicates):
    """Generate potential duplicates analysis."""
    dup = []
    dup.append("## 🔍 Potential Duplicate Analysis")
    dup.append("")

    if not duplicates:
        dup.append("✅ **No potential duplicates found** (threshold: ≥0.95)")
        dup.append("")
        return "\n".join(dup)

    dup.append(f"**Found {len(duplicates)} potential duplicates:**")
    dup.append("")

    # Group by score ranges
    high_dup = [d for d in duplicates if d["score"] >= 0.98]
    medium_dup = [d for d in duplicates if 0.95 <= d["score"] < 0.98]

    if high_dup:
        dup.append("**🔴 High Confidence Duplicates (≥0.98):**")
        for i, dup_test in enumerate(high_dup[:10], 1):  # Show top 10
            dup.append(f"{i}. **Go**: `{dup_test['go_test']}`")
            dup.append(f"   **Python**: `{dup_test['py_test']}`")
            dup.append(f"   **Score**: {dup_test['score']:.3f}")
            dup.append(f"   **Shared**: {dup_test['shared_signals']}")
            dup.append("")

    if medium_dup:
        dup.append("**🟡 Medium Confidence Duplicates (0.95-0.98):**")
        for i, dup_test in enumerate(medium_dup[:5], 1):  # Show top 5
            dup.append(f"{i}. **Go**: `{dup_test['go_test']}`")
            dup.append(f"   **Python**: `{dup_test['py_test']}`")
            dup.append(f"   **Score**: {dup_test['score']:.3f}")
            dup.append("")

    dup.append("**Recommendations:**")
    dup.append("- Review high confidence duplicates for consolidation opportunities")
    dup.append("- Consider creating shared test utilities for common patterns")
    dup.append("- Document differences between similar tests to justify their existence")
    dup.append("")

    return "\n".join(dup)


def generate_complementary_analysis(complementary):
    """Generate complementary tests analysis."""
    comp = []
    comp.append("## 🤝 Complementary Tests Analysis")
    comp.append("")

    if not complementary:
        comp.append("ℹ️ **No complementary tests found** (score range: 0.6-0.8)")
        comp.append("")
        return "\n".join(comp)

    comp.append(f"**Found {len(complementary)} complementary test pairs:**")
    comp.append("")

    # Group by purpose combinations
    purpose_combinations = defaultdict(list)
    for comp_test in complementary:
        key = f"{comp_test['go_purpose']} ↔ {comp_test['py_purpose']}"
        purpose_combinations[key].append(comp_test)

    comp.append("**Purpose Combinations:**")
    for purpose_combo, tests in purpose_combinations.items():
        comp.append(f"- **{purpose_combo}**: {len(tests)} pairs")

    comp.append("")

    # Show examples
    comp.append("**Example Complementary Pairs:**")
    for i, comp_test in enumerate(complementary[:5], 1):  # Show top 5
        comp.append(f"{i}. **Go**: `{comp_test['go_test']}` ({comp_test['go_purpose']})")
        comp.append(f"   **Python**: `{comp_test['py_test']}` ({comp_test['py_purpose']})")
        comp.append(f"   **Score**: {comp_test['score']:.3f}")
        comp.append(f"   **Suggestion**: {comp_test['suggestion']}")
        comp.append("")

    comp.append("**Opportunities:**")
    comp.append("- Cross-pollinate test approaches between Go and Python suites")
    comp.append("- Create hybrid tests that combine different testing strategies")
    comp.append("- Share test utilities and helper functions")
    comp.append("- Establish common testing patterns and conventions")
    comp.append("")

    return "\n".join(comp)


def generate_top_matches(test_report, go_specs, py_specs, top_n=20):
    """Generate top matches analysis."""
    top = []
    top.append("## 🏆 Top Similarity Matches")
    top.append("")

    # Get top matches
    top_matches = test_report.nlargest(top_n, "blended_score")

    top.append(f"**Top {top_n} Most Similar Test Pairs:**")
    top.append("")

    for i, (_, row) in enumerate(top_matches.iterrows(), 1):
        go_idx = int(row["idx_a"])
        py_idx = int(row["idx_b"])

        go_test_name = "Unknown"
        py_test_name = "Unknown"

        if go_idx < len(go_specs):
            go_test_name = go_specs[go_idx]["test_id"]
        if py_idx < len(py_specs):
            py_test_name = py_specs[py_idx]["test_id"]

        top.append(f"**{i}. Score: {row['blended_score']:.3f}**")
        top.append(f"- **Go**: `{go_test_name}`")
        top.append(f"- **Python**: `{py_test_name}`")
        top.append(f"- **Shared Signals**: {row['shared_signals']}")
        top.append("")

    return "\n".join(top)


def generate_insights_and_recommendations(patterns, signal_analysis, duplicates, complementary):
    """Generate insights and recommendations."""
    insights = []
    insights.append("## 💡 Insights & Strategic Recommendations")
    insights.append("")

    # Calculate key metrics
    total_matches = patterns["total_matches"]
    duplicate_ratio = len(duplicates) / total_matches if total_matches > 0 else 0
    complementary_ratio = len(complementary) / total_matches if total_matches > 0 else 0
    exact_match_ratio = signal_analysis["exact_matches"] / total_matches if total_matches > 0 else 0

    insights.append("**Key Insights:**")

    if duplicate_ratio > 0.15:
        insights.append(
            f"- 🔴 **High Duplication Risk**: {format_percentage(duplicate_ratio, 1)} of matches are potential duplicates"
        )
    elif duplicate_ratio > 0.05:
        insights.append(
            f"- 🟡 **Moderate Duplication**: {format_percentage(duplicate_ratio, 1)} of matches may be duplicates"
        )
    else:
        insights.append(
            f"- ✅ **Low Duplication**: Only {format_percentage(duplicate_ratio, 1)} potential duplicates found"
        )

    if complementary_ratio > 0.3:
        insights.append(
            f"- 💡 **High Cross-Pollination Potential**: {format_percentage(complementary_ratio, 1)} of matches are complementary"
        )
    elif complementary_ratio > 0.1:
        insights.append(
            f"- 🤝 **Moderate Complementarity**: {format_percentage(complementary_ratio, 1)} of matches are complementary"
        )
    else:
        insights.append(
            f"- 📊 **Low Complementarity**: Only {format_percentage(complementary_ratio, 1)} complementary matches found"
        )

    if exact_match_ratio > 0.2:
        insights.append(
            f"- 🎯 **Strong Functional Overlap**: {format_percentage(exact_match_ratio, 1)} of matches share exact operations"
        )
    else:
        insights.append(
            f"- 🔍 **Limited Functional Overlap**: Only {format_percentage(exact_match_ratio, 1)} of matches share exact operations"
        )

    insights.append("")

    # Generate recommendations
    recommendations = []

    if duplicate_ratio > 0.1:
        recommendations.append(
            {
                "priority": "High",
                "category": "Test Consolidation",
                "issue": f"High duplicate ratio ({format_percentage(duplicate_ratio, 1)})",
                "action": "Review and consolidate duplicate tests, create shared utilities",
                "impact": "Reduced maintenance overhead and clearer test purpose",
            }
        )

    if complementary_ratio > 0.2:
        recommendations.append(
            {
                "priority": "Medium",
                "category": "Test Enhancement",
                "issue": f"High complementary potential ({format_percentage(complementary_ratio, 1)})",
                "action": "Cross-pollinate test approaches and create hybrid tests",
                "impact": "Better test coverage and shared best practices",
            }
        )

    if patterns["avg_score"] < 0.4:
        recommendations.append(
            {
                "priority": "Low",
                "category": "Test Diversity",
                "issue": f"Low average similarity ({patterns['avg_score']:.3f})",
                "action": "Consider if tests are too different or if there are missing test categories",
                "impact": "Better understanding of test coverage gaps",
            }
        )

    if exact_match_ratio < 0.1:
        recommendations.append(
            {
                "priority": "Medium",
                "category": "Test Alignment",
                "issue": f"Low exact operation matches ({format_percentage(exact_match_ratio, 1)})",
                "action": "Align test approaches to cover similar operations",
                "impact": "More consistent testing across different implementations",
            }
        )

    # Format recommendations
    if recommendations:
        insights.append("**Strategic Recommendations:**")
        insights.append("")

        high_priority = [r for r in recommendations if r["priority"] == "High"]
        medium_priority = [r for r in recommendations if r["priority"] == "Medium"]
        low_priority = [r for r in recommendations if r["priority"] == "Low"]

        if high_priority:
            insights.append("**🔴 High Priority:**")
            for i, rec in enumerate(high_priority, 1):
                insights.append(f"{i}. **{rec['category']}**: {rec['issue']}")
                insights.append(f"   - **Action**: {rec['action']}")
                insights.append(f"   - **Impact**: {rec['impact']}")
                insights.append("")

        if medium_priority:
            insights.append("**🟡 Medium Priority:**")
            for i, rec in enumerate(medium_priority, 1):
                insights.append(f"{i}. **{rec['category']}**: {rec['issue']}")
                insights.append(f"   - **Action**: {rec['action']}")
                insights.append(f"   - **Impact**: {rec['impact']}")
                insights.append("")

        if low_priority:
            insights.append("**🟢 Low Priority:**")
            for i, rec in enumerate(low_priority, 1):
                insights.append(f"{i}. **{rec['category']}**: {rec['issue']}")
                insights.append(f"   - **Action**: {rec['action']}")
                insights.append(f"   - **Impact**: {rec['impact']}")
                insights.append("")

    return "\n".join(insights)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive similarity matches report")
    parser.add_argument("--test-report", required=True, help="Test report CSV file")
    parser.add_argument("--test-coverage", required=True, help="Test coverage CSV file")
    parser.add_argument("--go-specs", required=True, help="Go specs JSONL file")
    parser.add_argument("--py-specs", required=True, help="Python specs JSONL file")
    parser.add_argument("-o", "--output", help="Output file (default: similarity_report.md)")

    args = parser.parse_args()

    # Load data
    print("Loading similarity data...")
    test_report, test_coverage = load_similarity_data(args.test_report, args.test_coverage)
    all_specs, go_specs, py_specs = load_all_specs(args.go_specs, args.py_specs)

    print("Analyzing similarity patterns...")
    patterns = analyze_similarity_patterns(test_report)
    signal_analysis = analyze_shared_signals(test_report)

    print("Identifying potential duplicates...")
    duplicates = identify_potential_duplicates(test_report, all_specs, all_specs)

    print("Identifying complementary tests...")
    complementary = identify_complementary_tests(test_report, all_specs, all_specs)

    print("Generating comprehensive report...")

    # Generate report sections
    report_sections = [
        f"# Comprehensive Similarity Matches Analysis Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Analysis Type:** Language-agnostic similarity analysis\n",
        f"---\n",
        generate_executive_summary(patterns, signal_analysis, duplicates, complementary),
        generate_match_type_analysis(test_report),
        generate_score_distribution(patterns, test_report),
        generate_shared_signals_analysis(signal_analysis),
        generate_duplicate_analysis(duplicates),
        generate_complementary_analysis(complementary),
        generate_top_matches(test_report, all_specs, all_specs),
        generate_insights_and_recommendations(patterns, signal_analysis, duplicates, complementary),
    ]

    # Write report
    output_file = args.output or "similarity_report.md"
    with open(output_file, "w") as f:
        f.write("\n".join(report_sections))

    print(f"Comprehensive similarity report written to: {output_file}")


if __name__ == "__main__":
    main()
