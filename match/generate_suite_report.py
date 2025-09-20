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
    with open(analysis_file, 'r') as f:
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
    """Generate executive summary section."""
    summary = []
    summary.append("## 📋 Executive Summary")
    summary.append("")
    
    # Basic stats
    total_tests = analysis['total_tests']
    unique_ops = analysis['coverage_metrics']['unique_operations']
    unique_resources = analysis['coverage_metrics']['unique_resources']
    avg_ops_per_test = analysis['coverage_metrics']['avg_operations_per_test']
    
    summary.append(f"**Test Suite Overview:**")
    summary.append(f"- **Total Tests:** {format_number(total_tests)}")
    summary.append(f"- **Unique Operations:** {format_number(unique_ops)}")
    summary.append(f"- **Resource Types:** {format_number(unique_resources)}")
    summary.append(f"- **Avg Operations/Test:** {avg_ops_per_test:.1f}")
    summary.append("")
    
    # Test type distribution
    test_types = analysis.get('test_distribution', {}).get('by_type', {})
    if test_types:
        summary.append(f"**Test Type Distribution:**")
        for test_type, count in test_types.items():
            percentage = format_percentage(count, total_tests)
            summary.append(f"- {test_type.title()}: {format_number(count)} ({percentage})")
        summary.append("")
    
    # Purpose distribution
    purposes = analysis.get('test_distribution', {}).get('by_purpose', {})
    if purposes:
        summary.append(f"**Primary Focus Areas:**")
        for purpose, count in purposes.items():
            percentage = format_percentage(count, total_tests)
            summary.append(f"- {purpose.replace('_', ' ').title()}: {format_number(count)} ({percentage})")
        summary.append("")
    
    # Environment coverage
    environments = analysis.get('test_distribution', {}).get('by_environment', {})
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
    resources = analysis['resources']
    if resources:
        coverage.append("**Most Tested Resources:**")
        # Sort by count and take top 10
        sorted_resources = sorted(resources.items(), key=lambda x: x[1], reverse=True)[:10]
        for resource, count in sorted_resources:
            coverage.append(f"- `{resource}`: {format_number(count)} operations")
        coverage.append("")
    
    # Top operations
    operations = analysis['operations']
    if operations:
        coverage.append("**Most Common Operations:**")
        # Sort by count and take top 10
        sorted_operations = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:10]
        for operation, count in sorted_operations:
            coverage.append(f"- `{operation}`: {format_number(count)} tests")
        coverage.append("")
    
    # Operation categories
    verbs = analysis['verbs']
    if verbs:
        coverage.append("**Operation Categories:**")
        # Sort by count
        sorted_verbs = sorted(verbs.items(), key=lambda x: x[1], reverse=True)
        for verb, count in sorted_verbs:
            coverage.append(f"- {verb.title()}: {format_number(count)} operations")
        coverage.append("")
    
    # OpenShift-specific resources
    openshift_resources = analysis['openshift_resources']
    if openshift_resources:
        coverage.append("**OpenShift-Specific Resources:**")
        # Sort by count
        sorted_openshift = sorted(openshift_resources.items(), key=lambda x: x[1], reverse=True)
        for resource, count in sorted_openshift:
            coverage.append(f"- `{resource}`: {format_number(count)} operations")
        coverage.append("")
    
    return "\n".join(coverage)


def generate_quality_metrics(analysis):
    """Generate quality metrics section."""
    metrics = []
    metrics.append("## 📊 Quality Metrics")
    metrics.append("")
    
    coverage_metrics = analysis['coverage_metrics']
    
    metrics.append("**Test Diversity:**")
    metrics.append(f"- Test Type Diversity: {coverage_metrics['test_type_diversity']} different types")
    metrics.append(f"- Purpose Diversity: {coverage_metrics['purpose_diversity']} different purposes")
    metrics.append(f"- Environment Diversity: {coverage_metrics['environment_diversity']} different environments")
    metrics.append("")
    
    metrics.append("**Coverage Depth:**")
    metrics.append(f"- Operations per Test: {coverage_metrics['avg_operations_per_test']:.1f} average")
    metrics.append(f"- Resource Coverage: {coverage_metrics['unique_resources']} resource types")
    metrics.append(f"- Operation Coverage: {coverage_metrics['unique_operations']} unique operations")
    metrics.append("")
    
    # Test complexity analysis (simplified)
    total_tests = analysis['total_tests']
    if total_tests > 0:
        # Estimate complexity based on operations per test
        avg_ops = coverage_metrics['avg_operations_per_test']
        if avg_ops <= 1:
            complexity_note = "Most tests are simple (≤1 operation)"
        elif avg_ops <= 3:
            complexity_note = "Tests are moderately complex (1-3 operations)"
        elif avg_ops <= 5:
            complexity_note = "Tests are complex (3-5 operations)"
        else:
            complexity_note = "Tests are very complex (>5 operations)"
        
        metrics.append("**Test Complexity:**")
        metrics.append(f"- {complexity_note}")
        metrics.append(f"- Average operations per test: {avg_ops:.1f}")
        metrics.append("")
    
    return "\n".join(metrics)


def generate_insights_and_recommendations(analysis):
    """Generate insights and recommendations section."""
    insights = []
    insights.append("## 💡 Key Insights & Recommendations")
    insights.append("")
    
    # Key insights from analysis
    key_insights = analysis.get('key_insights', [])
    if key_insights:
        insights.append("**Analysis Insights:**")
        for insight in key_insights:
            insights.append(f"- {insight}")
        insights.append("")
    
    # Recommendations based on analysis
    recommendations = []
    
    # Test type recommendations
    test_types = analysis.get('test_distribution', {}).get('by_type', {})
    if test_types:
        most_common_type = max(test_types.items(), key=lambda x: x[1])
        if most_common_type[0] == 'unit' and most_common_type[1] / analysis['total_tests'] > 0.8:
            recommendations.append("Consider adding more integration tests to improve end-to-end coverage")
        elif most_common_type[0] == 'integration' and most_common_type[1] / analysis['total_tests'] > 0.8:
            recommendations.append("Consider adding more unit tests for better component-level coverage")
    
    # Purpose recommendations
    purposes = analysis.get('test_distribution', {}).get('by_purpose', {})
    if purposes:
        purpose_count = len(purposes)
        if purpose_count < 3:
            recommendations.append("Consider diversifying test purposes to cover more scenarios")
        elif purpose_count > 8:
            recommendations.append("Consider consolidating similar test purposes for better organization")
    
    # Environment recommendations
    environments = analysis.get('test_distribution', {}).get('by_environment', {})
    if environments:
        env_count = len(environments)
        if env_count == 1:
            recommendations.append("Consider testing across multiple environments for better coverage")
        elif env_count > 5:
            recommendations.append("Consider focusing on primary target environments")
    
    # Operation recommendations
    operations = analysis['operations']
    if operations:
        total_ops = sum(operations.values())
        unique_ops = len(operations)
        if unique_ops / total_ops < 0.1:
            recommendations.append("Consider diversifying operations to cover more Kubernetes APIs")
        elif unique_ops / total_ops > 0.5:
            recommendations.append("Consider consolidating similar operations for better maintainability")
    
    # Resource recommendations
    resources = analysis['resources']
    if resources:
        resource_count = len(resources)
        if resource_count < 5:
            recommendations.append("Consider testing more Kubernetes resource types")
        elif resource_count > 50:
            recommendations.append("Consider focusing on core resource types for better maintainability")
    
    if recommendations:
        insights.append("**Recommendations:**")
        for i, rec in enumerate(recommendations, 1):
            insights.append(f"{i}. {rec}")
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
    test_types = analysis.get('test_distribution', {}).get('by_type', {})
    if test_types:
        questions.append("**Test Type Classification:**")
        for test_type, count in test_types.items():
            percentage = format_percentage(count, analysis['total_tests'])
            questions.append(f"- Are {format_number(count)} tests ({percentage}) correctly classified as `{test_type}` tests?")
        questions.append("")
    
    # Purpose validation
    purposes = analysis.get('test_distribution', {}).get('by_purpose', {})
    if purposes:
        questions.append("**Purpose Classification:**")
        for purpose, count in purposes.items():
            percentage = format_percentage(count, analysis['total_tests'])
            questions.append(f"- Are {format_number(count)} tests ({percentage}) correctly classified as `{purpose.replace('_', ' ').lower()}` tests?")
        questions.append("")
    
    # Environment validation
    environments = analysis.get('test_distribution', {}).get('by_environment', {})
    if environments:
        questions.append("**Environment Classification:**")
        for env, count in environments.items():
            questions.append(f"- Are {format_number(count)} tests correctly targeting `{env.replace('_', ' ').lower()}` environments?")
        questions.append("")
    
    # Resource validation
    resources = analysis.get('resources', {})
    if resources:
        questions.append("**Resource Coverage:**")
        # Sort by count and take top 5
        top_resources = sorted(resources.items(), key=lambda x: x[1], reverse=True)[:5]
        for resource, count in top_resources:
            questions.append(f"- Are tests correctly using `{resource}` resources ({format_number(count)} operations)?")
        questions.append("")
    
    # Operation validation
    operations = analysis.get('operations', {})
    if operations:
        questions.append("**Operation Coverage:**")
        # Sort by count and take top 5
        top_operations = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:5]
        for operation, count in top_operations:
            questions.append(f"- Are tests correctly performing `{operation}` operations ({format_number(count)} times)?")
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
    suite_name = analysis['suite_name']
    
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
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"Report written to: {output_file}")
    else:
        print(full_report)
    
    return full_report


def main():
    parser = argparse.ArgumentParser(description='Generate high-level test suite report')
    parser.add_argument('analysis_file', help='Path to analysis JSON file')
    parser.add_argument('-o', '--output', help='Output file path (default: stdout)')
    parser.add_argument('--format', choices=['markdown', 'html'], default='markdown', help='Output format')
    
    args = parser.parse_args()
    
    if not Path(args.analysis_file).exists():
        print(f"Error: Analysis file '{args.analysis_file}' not found")
        sys.exit(1)
    
    try:
        generate_suite_report(args.analysis_file, args.output)
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
