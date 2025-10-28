#!/usr/bin/env python3
"""
Test Suite Report Generator

Generates high-level, human-readable reports for test suite owners
to validate analysis accuracy and understand test coverage.
"""

import json
import argparse
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime


def load_analysis(analysis_file):
    """Load test suite analysis from JSON file."""
    with open(analysis_file, "r") as f:
        return json.load(f)


def format_number(num):
    """Format numbers with commas for readability."""
    return f"{num:,}"


def format_percentage(value, total):
    """Format percentage with 1 decimal place."""
    if total == 0:
        return "0.0%"
    return f"{(value / total * 100):.1f}%"


def generate_executive_summary(analysis):
    """Generate executive summary section with enhanced insights."""
    summary = []
    summary.append("## 📋 Executive Summary")
    summary.append("")

    # Basic stats
    total_tests = analysis["total_tests"]
    unique_ops = analysis["coverage_metrics"]["unique_operations"]
    unique_resources = analysis["coverage_metrics"]["unique_resources"]
    avg_ops_per_test = analysis["coverage_metrics"]["avg_operations_per_test"]

    # Calculate additional metrics
    test_types = analysis.get("test_distribution", {}).get("by_type", {})
    purposes = analysis.get("test_distribution", {}).get("by_purpose", {})
    environments = analysis.get("test_distribution", {}).get("by_environment", {})

    # Test complexity assessment
    complexity_level = (
        "High" if avg_ops_per_test > 3 else "Medium" if avg_ops_per_test > 1.5 else "Low"
    )

    # Coverage diversity
    purpose_diversity = len(purposes)
    environment_diversity = len(environments)

    # Most common test type
    most_common_type = max(test_types.items(), key=lambda x: x[1]) if test_types else ("Unknown", 0)
    most_common_purpose = max(purposes.items(), key=lambda x: x[1]) if purposes else ("Unknown", 0)

    # Test distribution health
    integration_ratio = test_types.get("integration", 0) / total_tests if total_tests > 0 else 0
    test_health = (
        "Excellent"
        if integration_ratio > 0.7
        else "Good" if integration_ratio > 0.4 else "Needs Improvement"
    )

    summary.append(f"**Test Suite Overview:**")
    summary.append(f"- **Total Tests:** {format_number(total_tests)}")
    summary.append(f"- **Unique Operations:** {format_number(unique_ops)}")
    summary.append(f"- **Resource Types:** {format_number(unique_resources)}")
    summary.append(
        f"- **Avg Operations/Test:** {avg_ops_per_test:.1f} ({complexity_level} complexity)"
    )
    summary.append("")

    summary.append(f"**Test Health Score:** {test_health}")
    summary.append(f"- Integration Test Ratio: {integration_ratio:.1%}")
    summary.append(f"- Purpose Diversity: {purpose_diversity} categories")
    summary.append(f"- Environment Coverage: {environment_diversity} environments")
    summary.append("")

    summary.append(f"**Primary Focus:**")
    summary.append(
        f"- Most Common Type: {most_common_type[0]} ({most_common_type[1]} tests, {format_percentage(most_common_type[1], total_tests)})"
    )
    summary.append(
        f"- Main Purpose: {most_common_purpose[0]} ({most_common_purpose[1]} tests, {format_percentage(most_common_purpose[1], total_tests)})"
    )
    summary.append("")

    # Test type distribution
    test_types = analysis.get("test_distribution", {}).get("by_type", {})
    if test_types:
        summary.append(f"**Test Type Distribution:**")
        for test_type, count in test_types.items():
            percentage = format_percentage(count, total_tests)
            summary.append(f"- {test_type.title()}: {format_number(count)} ({percentage})")
        summary.append("")

    # Purpose distribution
    purposes = analysis.get("test_distribution", {}).get("by_purpose", {})
    if purposes:
        summary.append(f"**Primary Focus Areas:**")
        for purpose, count in purposes.items():
            percentage = format_percentage(count, total_tests)
            summary.append(
                f"- {purpose.replace('_', ' ').title()}: {format_number(count)} ({percentage})"
            )
        summary.append("")

    # Environment coverage
    environments = analysis.get("test_distribution", {}).get("by_environment", {})
    if environments:
        summary.append(f"**Target Environments:**")
        for env, count in environments.items():
            summary.append(f"- {env.replace('_', ' ').title()}: {format_number(count)} tests")
        summary.append("")

    return "\n".join(summary)


def generate_coverage_analysis(analysis):
    """Generate coverage analysis section."""
    coverage = []
    coverage.append("## 🔍 Coverage Analysis")
    coverage.append("")

    # Top resources
    resources = analysis["resources"]
    if resources:
        coverage.append("**Most Tested Resources:**")
        # Sort by count and take top 10
        sorted_resources = sorted(resources.items(), key=lambda x: x[1], reverse=True)[:10]
        for resource, count in sorted_resources:
            coverage.append(f"- `{resource}`: {format_number(count)} operations")
        coverage.append("")

    # Top operations
    operations = analysis["operations"]
    if operations:
        coverage.append("**Most Common Operations:**")
        # Sort by count and take top 10
        sorted_operations = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:10]
        for operation, count in sorted_operations:
            coverage.append(f"- `{operation}`: {format_number(count)} tests")
        coverage.append("")

    # Operation categories
    verbs = analysis["verbs"]
    if verbs:
        coverage.append("**Operation Categories:**")
        # Sort by count
        sorted_verbs = sorted(verbs.items(), key=lambda x: x[1], reverse=True)
        for verb, count in sorted_verbs:
            coverage.append(f"- {verb.title()}: {format_number(count)} operations")
        coverage.append("")

    # OpenShift-specific resources
    openshift_resources = analysis["openshift_resources"]
    if openshift_resources:
        coverage.append("**OpenShift-Specific Resources:**")
        # Sort by count
        sorted_openshift = sorted(openshift_resources.items(), key=lambda x: x[1], reverse=True)
        for resource, count in sorted_openshift:
            coverage.append(f"- `{resource}`: {format_number(count)} operations")
        coverage.append("")

    # Networking technology analysis
    purposes = analysis.get("test_distribution", {}).get("by_purpose", {})
    tech = {}
    for purpose, count in purposes.items():
        if purpose in [
            "SRIOV_TESTING",
            "DUAL_STACK_TESTING",
            "IPV4_ONLY_TESTING",
            "IPV6_ONLY_TESTING",
            "PTP_TESTING",
        ]:
            tech[purpose] = count

    if tech:
        coverage.append("**Networking Technologies:**")
        for tech_name, count in tech.items():
            tech_display = tech_name.replace("_TESTING", "").replace("_", " ").title()
            # Special formatting for IP stack categories
            if tech_name == "IPV4_ONLY_TESTING":
                tech_display = "IPv4 Only"
            elif tech_name == "IPV6_ONLY_TESTING":
                tech_display = "IPv6 Only"
            elif tech_name == "DUAL_STACK_TESTING":
                tech_display = "Dual Stack (IPv4+IPv6)"
            coverage.append(f"- {tech_display}: {format_number(count)} tests")
        coverage.append("")

    return "\n".join(coverage)


def generate_quality_metrics(analysis):
    """Generate enhanced quality metrics section with insights."""
    metrics = []
    metrics.append("## 📊 Quality Metrics & Insights")
    metrics.append("")

    coverage_metrics = analysis["coverage_metrics"]
    total_tests = analysis["total_tests"]

    # Calculate quality scores
    test_diversity = coverage_metrics["test_type_diversity"]
    purpose_diversity = coverage_metrics["purpose_diversity"]
    environment_diversity = coverage_metrics["environment_diversity"]
    avg_ops_per_test = coverage_metrics["avg_operations_per_test"]

    diversity_score = min(
        100, (test_diversity * 20 + purpose_diversity * 10 + environment_diversity * 15)
    )
    complexity_score = min(100, max(0, (avg_ops_per_test - 0.5) * 30))
    coverage_score = min(
        100,
        (coverage_metrics["unique_resources"] * 2 + coverage_metrics["unique_operations"] * 0.5),
    )

    overall_score = (diversity_score + complexity_score + coverage_score) / 3
    quality_grade = (
        "A"
        if overall_score >= 90
        else (
            "B"
            if overall_score >= 80
            else "C" if overall_score >= 70 else "D" if overall_score >= 60 else "F"
        )
    )

    metrics.append(f"**Overall Quality Score: {quality_grade} ({overall_score:.1f}/100)**")
    metrics.append("")

    metrics.append(f"**Test Diversity Score: {diversity_score:.1f}/100**")
    metrics.append(f"- Test Type Diversity: {test_diversity} different types")
    metrics.append(f"- Purpose Diversity: {purpose_diversity} different purposes")
    metrics.append(f"- Environment Diversity: {environment_diversity} different environments")
    metrics.append("")

    metrics.append(f"**Coverage Score: {coverage_score:.1f}/100**")
    metrics.append(f"- Resource Coverage: {coverage_metrics['unique_resources']} resource types")
    metrics.append(
        f"- Operation Coverage: {coverage_metrics['unique_operations']} unique operations"
    )
    metrics.append(f"- Operations per Test: {avg_ops_per_test:.1f} average")
    metrics.append("")

    # Test complexity assessment with recommendations
    if avg_ops_per_test > 3:
        complexity = "High (3-5 operations per test)"
        complexity_note = "✅ Good: Tests are comprehensive and thorough"
    elif avg_ops_per_test > 1.5:
        complexity = "Medium (1.5-3 operations per test)"
        complexity_note = "⚠️ Consider: Some tests might benefit from more comprehensive coverage"
    else:
        complexity = "Low (1-1.5 operations per test)"
        complexity_note = "❌ Attention: Tests may be too simple and miss edge cases"

    metrics.append(f"**Test Complexity: {complexity}**")
    metrics.append(f"- {complexity_note}")
    metrics.append("")

    # Resource utilization analysis
    operations = analysis.get("operations", {})
    if operations:
        total_operations = sum(operations.values())
        most_used_ops = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:5]

        metrics.append(f"**Resource Utilization:**")
        metrics.append(f"- Total Operations: {format_number(total_operations)}")
        metrics.append(f"- Most Used Operations:")
        for op, count in most_used_ops:
            percentage = (count / total_operations) * 100
            metrics.append(f"  - `{op}`: {count} times ({percentage:.1f}%)")
        metrics.append("")

    # Test effectiveness indicators
    purposes = analysis.get("test_distribution", {}).get("by_purpose", {})
    if purposes:
        purpose_balance = (
            len([p for p in purposes.values() if p > 0]) / len(purposes) if purposes else 0
        )
        metrics.append(f"**Test Effectiveness:**")
        metrics.append(f"- Purpose Balance: {purpose_balance:.1%} of purpose categories have tests")
        if purpose_balance < 0.5:
            metrics.append(f"  - ⚠️ Consider: Many purpose categories are under-tested")
        else:
            metrics.append(f"  - ✅ Good: Well-distributed test purposes")
        metrics.append("")

    return "\n".join(metrics)


def generate_insights_and_recommendations(analysis):
    """Generate comprehensive insights and actionable recommendations."""
    insights = []
    insights.append("## 💡 Strategic Insights & Actionable Recommendations")
    insights.append("")

    # Key insights from analysis
    key_insights = analysis.get("key_insights", [])
    if key_insights:
        insights.append("**Analysis Insights:**")
        for insight in key_insights:
            insights.append(f"- {insight}")
        insights.append("")

    # Comprehensive recommendations based on analysis
    recommendations = []
    total_tests = analysis["total_tests"]
    coverage_metrics = analysis["coverage_metrics"]

    # Test type recommendations with specific actions
    test_types = analysis.get("test_distribution", {}).get("by_type", {})
    if test_types:
        most_common_type = max(test_types.items(), key=lambda x: x[1])
        type_ratio = most_common_type[1] / total_tests

        if most_common_type[0] == "unit" and type_ratio > 0.8:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Test Type Balance",
                    "issue": "Heavy unit test focus (80%+) may miss integration issues",
                    "action": "Add 20-30 integration tests covering critical user workflows",
                    "impact": "Better end-to-end coverage and real-world scenario testing",
                }
            )
        elif most_common_type[0] == "integration" and type_ratio > 0.8:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Test Type Balance",
                    "issue": "Heavy integration test focus may slow development",
                    "action": "Add unit tests for individual components and edge cases",
                    "impact": "Faster feedback loop and better component isolation",
                }
            )

    # Purpose diversity recommendations
    purposes = analysis.get("test_distribution", {}).get("by_purpose", {})
    if purposes:
        purpose_count = len(purposes)
        purpose_balance = len([p for p in purposes.values() if p > 0]) / len(purposes)

        if purpose_count < 3:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Test Coverage",
                    "issue": f"Limited test purpose diversity ({purpose_count} categories)",
                    "action": "Add tests for security, performance, and error handling scenarios",
                    "impact": "More comprehensive test coverage and risk mitigation",
                }
            )
        elif purpose_balance < 0.5:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Test Coverage",
                    "issue": f"Many purpose categories are under-tested ({purpose_balance:.1%} coverage)",
                    "action": "Review and add tests for under-represented purpose categories",
                    "impact": "Better balanced test coverage across all scenarios",
                }
            )

    # Environment coverage recommendations
    environments = analysis.get("test_distribution", {}).get("by_environment", {})
    if environments:
        env_count = len(environments)
        if env_count == 1:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Environment Coverage",
                    "issue": f"Testing only in {list(environments.keys())[0]} environment",
                    "action": "Add tests for cloud, bare-metal, and edge environments",
                    "impact": "Better compatibility across deployment scenarios",
                }
            )
        elif env_count > 5:
            recommendations.append(
                {
                    "priority": "Low",
                    "category": "Environment Coverage",
                    "issue": f"Testing across {env_count} environments may be excessive",
                    "action": "Focus on 2-3 primary target environments",
                    "impact": "Reduced maintenance overhead and clearer test strategy",
                }
            )

    # Operation diversity recommendations
    operations = analysis.get("operations", {})
    if operations:
        total_ops = sum(operations.values())
        unique_ops = len(operations)
        diversity_ratio = unique_ops / total_ops

        if diversity_ratio < 0.1:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Operation Diversity",
                    "issue": f"Low operation diversity ({diversity_ratio:.1%} unique operations)",
                    "action": "Add tests using different Kubernetes API operations (patch, watch, etc.)",
                    "impact": "More comprehensive API coverage and edge case testing",
                }
            )
        elif diversity_ratio > 0.5:
            recommendations.append(
                {
                    "priority": "Low",
                    "category": "Operation Diversity",
                    "issue": f"Very high operation diversity ({diversity_ratio:.1%} unique operations)",
                    "action": "Consider consolidating similar operations for maintainability",
                    "impact": "Easier test maintenance and clearer patterns",
                }
            )

    # Resource coverage recommendations
    resources = analysis.get("resources", {})
    if resources:
        resource_count = len(resources)
        if resource_count < 5:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Resource Coverage",
                    "issue": f"Limited resource coverage ({resource_count} resource types)",
                    "action": "Add tests for core Kubernetes resources (Services, ConfigMaps, Secrets, etc.)",
                    "impact": "Better coverage of essential Kubernetes functionality",
                }
            )
        elif resource_count > 50:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Resource Coverage",
                    "issue": f"Very broad resource coverage ({resource_count} resource types)",
                    "action": "Focus on 20-30 core resource types for better maintainability",
                    "impact": "Reduced complexity and clearer test focus",
                }
            )

    # Test complexity recommendations
    avg_ops_per_test = coverage_metrics["avg_operations_per_test"]
    if avg_ops_per_test < 1.5:
        recommendations.append(
            {
                "priority": "High",
                "category": "Test Quality",
                "issue": f"Low test complexity ({avg_ops_per_test:.1f} operations per test)",
                "action": "Enhance tests with more comprehensive scenarios and edge cases",
                "impact": "Better test coverage and reduced production issues",
            }
        )
    elif avg_ops_per_test > 5:
        recommendations.append(
            {
                "priority": "Medium",
                "category": "Test Quality",
                "issue": f"Very high test complexity ({avg_ops_per_test:.1f} operations per test)",
                "action": "Break down complex tests into smaller, focused test cases",
                "impact": "Easier debugging and better test isolation",
            }
        )

    # Format recommendations with priority and impact
    if recommendations:
        insights.append("**Strategic Recommendations:**")
        insights.append("")

        # Group by priority
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


def generate_validation_questions(analysis):
    """Generate validation questions for test suite owners."""
    questions = []
    questions.append("## ❓ Validation Questions")
    questions.append("")
    questions.append("Please review the following questions to validate the analysis accuracy:")
    questions.append("")

    # Test type validation
    test_types = analysis.get("test_distribution", {}).get("by_type", {})
    if test_types:
        questions.append("**Test Type Classification:**")
        for test_type, count in test_types.items():
            percentage = format_percentage(count, analysis["total_tests"])
            questions.append(
                f"- Are {format_number(count)} tests ({percentage}) correctly classified as `{test_type}` tests?"
            )
        questions.append("")

    # Purpose validation
    purposes = analysis.get("test_distribution", {}).get("by_purpose", {})
    if purposes:
        questions.append("**Purpose Classification:**")
        for purpose, count in purposes.items():
            percentage = format_percentage(count, analysis["total_tests"])
            questions.append(
                f"- Are {format_number(count)} tests ({percentage}) correctly classified as `{purpose.replace('_', ' ').lower()}` tests?"
            )
        questions.append("")

    # Environment validation
    environments = analysis.get("test_distribution", {}).get("by_environment", {})
    if environments:
        questions.append("**Environment Classification:**")
        for env, count in environments.items():
            questions.append(
                f"- Are {format_number(count)} tests correctly targeting `{env.replace('_', ' ').lower()}` environments?"
            )
        questions.append("")

    # Resource validation
    resources = analysis.get("resources", {})
    if resources:
        questions.append("**Resource Coverage:**")
        # Sort by count and take top 5
        top_resources = sorted(resources.items(), key=lambda x: x[1], reverse=True)[:5]
        for resource, count in top_resources:
            questions.append(
                f"- Are tests correctly using `{resource}` resources ({format_number(count)} operations)?"
            )
        questions.append("")

    # Operation validation
    operations = analysis.get("operations", {})
    if operations:
        questions.append("**Operation Coverage:**")
        # Sort by count and take top 5
        top_operations = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:5]
        for operation, count in top_operations:
            questions.append(
                f"- Are tests correctly performing `{operation}` operations ({format_number(count)} times)?"
            )
        questions.append("")

    questions.append("**General Questions:**")
    questions.append("- Are there any missing test types, purposes, or environments?")
    questions.append("- Are there any incorrectly classified tests?")
    questions.append("- Are there any missing or incorrect resource types?")
    questions.append("- Are there any missing or incorrect operations?")
    questions.append("- Are there any other patterns or characteristics not captured?")
    questions.append("")

    return "\n".join(questions)


def generate_suite_report(analysis_file, output_file=None):
    """Generate comprehensive test suite report."""
    analysis = load_analysis(analysis_file)
    suite_name = analysis["suite_name"]

    # Generate report sections
    report = []
    report.append(f"# Test Suite Analysis Report: {suite_name}")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Add all sections
    report.append(generate_executive_summary(analysis))
    report.append("")
    report.append("---")
    report.append("")

    report.append(generate_coverage_analysis(analysis))
    report.append("")
    report.append("---")
    report.append("")

    report.append(generate_quality_metrics(analysis))
    report.append("")
    report.append("---")
    report.append("")

    report.append(generate_insights_and_recommendations(analysis))
    report.append("")
    report.append("---")
    report.append("")

    report.append(generate_validation_questions(analysis))

    # Join all sections
    full_report = "\n".join(report)

    # Write to file or stdout
    if output_file:
        with open(output_file, "w") as f:
            f.write(full_report)
        print(f"Report written to: {output_file}")
    else:
        print(full_report)

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Generate high-level test suite report")
    parser.add_argument("analysis_file", help="Path to analysis JSON file")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument(
        "--format",
        choices=["markdown", "html"],
        default="markdown",
        help="Output format",
    )

    args = parser.parse_args()

    if not Path(args.analysis_file).exists():
        print(f"Error: Analysis file '{args.analysis_file}' not found")
        sys.exit(1)

    try:
        generate_suite_report(args.analysis_file, args.output)
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
