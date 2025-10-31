#!/bin/bash

# extract-and-match.sh - Test Spec Extractor and Matching Pipeline
# This script runs the complete pipeline: extract specs from Go and Python tests, then perform similarity matching

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GO_ROOTS=()
PY_ROOTS=()
OUTPUT_DIR="results"
CLEANUP=true
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Test Spec Extractor and Matching Pipeline

OPTIONS:
    -g, --go-root PATH      Path to Go test repository (can be used multiple times)
    -p, --py-root PATH      Path to Python test repository (can be used multiple times)
    -o, --output-dir DIR    Output directory for results (default: results)
    -c, --no-cleanup        Don't clean up intermediate files
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Single repositories
    ./extract-and-match.sh -g /path/to/eco-gotests -p /path/to/eco-pytests
    
    # Multiple Go repositories
    ./extract-and-match.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -p /path/to/eco-pytests
    
    # Multiple Python repositories  
    ./extract-and-match.sh -g /path/to/eco-gotests -p /path/to/eco-pytests -p /path/to/other-pytests
    
    # Multiple repositories of both types
    ./extract-and-match.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -p /path/to/eco-pytests -p /path/to/other-pytests
    
    # Custom output directory
    ./extract-and-match.sh --go-root /path/to/go-tests --py-root /path/to/py-tests --output-dir my_results

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--go-root)
            GO_ROOTS+=("$2")
            shift 2
            ;;
        -p|--py-root)
            PY_ROOTS+=("$2")
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--no-cleanup)
            CLEANUP=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ ${#GO_ROOTS[@]} -eq 0 || ${#PY_ROOTS[@]} -eq 0 ]]; then
    print_error "At least one --go-root and one --py-root are required"
    show_usage
    exit 1
fi

# Validate input directories
for go_root in "${GO_ROOTS[@]}"; do
    if [[ ! -d "$go_root" ]]; then
        print_error "Go root directory does not exist: $go_root"
        exit 1
    fi
done

for py_root in "${PY_ROOTS[@]}"; do
    if [[ ! -d "$py_root" ]]; then
        print_error "Python root directory does not exist: $py_root"
        exit 1
    fi
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

print_status "Starting Test Spec Extractor and Matching Pipeline"
print_status "Go repositories: ${#GO_ROOTS[@]}"
for go_root in "${GO_ROOTS[@]}"; do
    print_status "  - $go_root"
done
print_status "Python repositories: ${#PY_ROOTS[@]}"
for py_root in "${PY_ROOTS[@]}"; do
    print_status "  - $py_root"
done
print_status "Output directory: $OUTPUT_DIR"
echo

# Step 1: Build Go extractor (if needed)
print_status "Step 1: Checking Go extractor..."
cd go-extractor

# Check if binary exists and is newer than source files
NEED_BUILD=false
if [[ ! -f "kubespec-go" ]]; then
    print_status "Go binary not found, will build..."
    NEED_BUILD=true
elif [[ "main.go" -nt "kubespec-go" ]] || [[ "go.mod" -nt "kubespec-go" ]]; then
    print_status "Go source files newer than binary, will rebuild..."
    NEED_BUILD=true
else
    print_status "Go binary is up to date"
fi

if [[ "$NEED_BUILD" == "true" ]]; then
    print_status "Building Go extractor..."
    if ! go build -o kubespec-go; then
        print_error "Failed to build Go extractor"
        exit 1
    fi
    print_success "Go extractor built successfully"
else
    print_success "Go extractor ready (using existing binary)"
fi
cd ..

# Step 2: Extract Go specs from all repositories
print_status "Step 2: Extracting Go test specs from ${#GO_ROOTS[@]} repositories..."
GO_SPECS_COUNT=0
for i in "${!GO_ROOTS[@]}"; do
    go_root="${GO_ROOTS[$i]}"
    repo_name=$(basename "$go_root")
    print_status "  Extracting from $repo_name..."
    
    if ! ./go-extractor/kubespec-go -root "$go_root" > "$OUTPUT_DIR/go_specs_${i}_${repo_name}.jsonl"; then
        print_error "Failed to extract Go specs from $go_root"
        exit 1
    fi
    
    repo_count=$(wc -l < "$OUTPUT_DIR/go_specs_${i}_${repo_name}.jsonl")
    GO_SPECS_COUNT=$((GO_SPECS_COUNT + repo_count))
    print_success "  Extracted $repo_count specs from $repo_name"
done

# Combine all Go specs into one file
print_status "  Combining Go specs..."
cat "$OUTPUT_DIR"/go_specs_*.jsonl > "$OUTPUT_DIR/go_specs.jsonl"
print_success "Extracted $GO_SPECS_COUNT total Go test specs"

# Optional: Extract per-file Ginkgo markdown specs
print_status "Step 2b: Generating per-file Ginkgo markdown specs..."
cd go-extractor

# Build spec-extractor if needed (rebuild when any .go or go.mod/sum is newer than the binary)
NEED_BUILD_SPEC=false
if [[ ! -f "spec-extractor/spec-extractor" ]]; then
    NEED_BUILD_SPEC=true
else
    if find spec-extractor -type f \( -name '*.go' -o -name 'go.mod' -o -name 'go.sum' \) -newer spec-extractor/spec-extractor | grep -q .; then
        NEED_BUILD_SPEC=true
    fi
fi
if [[ "$NEED_BUILD_SPEC" == "true" ]]; then
    print_status "Building spec-extractor (changes detected)..."
    if ! (cd spec-extractor && go build -o spec-extractor); then
        print_warning "Failed to build spec-extractor, skipping markdown generation"
        cd ..
        goto_after_md=true
    else
        goto_after_md=false
    fi
else
    print_status "spec-extractor binary is up to date"
    goto_after_md=false
fi

if [[ "$goto_after_md" != "true" ]]; then
    mkdir -p "../$OUTPUT_DIR/spec-md"
    # Per-It JSONL output
    GO_PER_IT_JSONL="../$OUTPUT_DIR/go_specs_per_it.jsonl"
    rm -f "$GO_PER_IT_JSONL"
    for i in "${!GO_ROOTS[@]}"; do
        go_root="${GO_ROOTS[$i]}"
        repo_name=$(basename "$go_root")
        outdir="../$OUTPUT_DIR/spec-md/$repo_name"
        print_status "  Rendering specs for $repo_name..."
        if ! ./spec-extractor/spec-extractor --root "$go_root" --out "$outdir" --jsonl "$GO_PER_IT_JSONL" >/dev/null 2>&1; then
            print_warning "  Failed to render markdown for $repo_name"
        fi
    done
fi
cd ..

# Optional: Run similarity on per-It Go tests (self-match) to find duplicates
if [[ -f "$OUTPUT_DIR/go_specs_per_it.jsonl" ]]; then
    print_status "Step 2c: Finding similar Go tests (per-It JSONL)..."
    if python match/build_index_and_match.py \
        --go "$OUTPUT_DIR/go_specs_per_it.jsonl" \
        --py "$OUTPUT_DIR/go_specs_per_it.jsonl" \
        --out "$OUTPUT_DIR/go_per_it_sim.csv" \
        --cov "$OUTPUT_DIR/go_per_it_cov.csv"; then
        print_success "Generated $OUTPUT_DIR/go_per_it_sim.csv (per-It similarities)"
    else
        print_warning "Failed to compute per-It similarities"
    fi
fi

# Step 3: Extract Python specs from all repositories
print_status "Step 3: Extracting Python test specs from ${#PY_ROOTS[@]} repositories..."
PY_SPECS_COUNT=0
for i in "${!PY_ROOTS[@]}"; do
    py_root="${PY_ROOTS[$i]}"
    repo_name=$(basename "$py_root")
    print_status "  Extracting from $repo_name..."
    
    if ! python py-extractor/extract_kubespec.py --root "$py_root" > "$OUTPUT_DIR/py_specs_${i}_${repo_name}.jsonl"; then
        print_error "Failed to extract Python specs from $py_root"
        exit 1
    fi
    
    repo_count=$(wc -l < "$OUTPUT_DIR/py_specs_${i}_${repo_name}.jsonl")
    PY_SPECS_COUNT=$((PY_SPECS_COUNT + repo_count))
    print_success "  Extracted $repo_count specs from $repo_name"
done

# Combine all Python specs into one file
print_status "  Combining Python specs..."
cat "$OUTPUT_DIR"/py_specs_*.jsonl > "$OUTPUT_DIR/py_specs.jsonl"
print_success "Extracted $PY_SPECS_COUNT total Python test specs"

# Step 4: Setup Python environment for matching
print_status "Step 4: Setting up Python environment for matching..."
cd match

# Check if virtual environment already exists
if [[ -d ".venv" ]]; then
    print_status "Using existing virtual environment..."
    if [[ ! -f ".venv/bin/activate" ]]; then
        print_error "Virtual environment exists but activation script not found"
        print_status "Recreating virtual environment..."
        rm -rf .venv
        if ! python -m venv .venv; then
            print_error "Failed to create Python virtual environment"
            exit 1
        fi
    fi
else
    print_status "Creating new Python virtual environment..."
    if ! python -m venv .venv; then
        print_error "Failed to create Python virtual environment"
        exit 1
    fi
fi

print_status "Activating virtual environment..."
if ! source .venv/bin/activate; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Check if dependencies are already installed
if ! python -c "import sentence_transformers, faiss, pandas, numpy" > /dev/null 2>&1; then
    print_status "Installing Python dependencies..."
    if ! pip install --upgrade pip > /dev/null 2>&1; then
        print_error "Failed to upgrade pip"
        exit 1
    fi
    if ! pip install -r requirements.txt > /dev/null 2>&1; then
        print_error "Failed to install Python dependencies"
        exit 1
    fi
    print_success "Dependencies installed"
else
    print_status "Dependencies already installed"
fi
print_success "Python environment ready"

# Step 5: Run similarity matching
print_status "Step 5: Running similarity matching..."
if ! python build_index_and_match.py \
    --go "../$OUTPUT_DIR/go_specs.jsonl" \
    --py "../$OUTPUT_DIR/py_specs.jsonl" \
    --out "../$OUTPUT_DIR/test_report.csv" \
    --cov "../$OUTPUT_DIR/test_coverage.csv"; then
    print_error "Failed to run similarity matching"
    exit 1
fi

# Step 6: Analyze results
print_status "Step 6: Analyzing results..."
cd ..

TOTAL_MATCHES=$(wc -l < "$OUTPUT_DIR/test_report.csv")
PERFECT_MATCHES=$(awk -F',' '$3 == 1.0 {count++} END {print count+0}' "$OUTPUT_DIR/test_report.csv")
MEANINGFUL_MATCHES=$(awk -F',' '$3 < 0.99 {count++} END {print count+0}' "$OUTPUT_DIR/test_report.csv")

print_success "Matching completed successfully!"
echo
print_status "Results Summary:"
echo "  - Total matches: $TOTAL_MATCHES"
echo "  - Perfect matches (score = 1.0): $PERFECT_MATCHES"
echo "  - Meaningful matches (score < 0.99): $MEANINGFUL_MATCHES"
echo "  - Go test specs: $GO_SPECS_COUNT"
echo "  - Python test specs: $PY_SPECS_COUNT"

# Step 7: Show sample results
print_status "Step 7: Sample results (top 5 matches):"
if [[ -f "$OUTPUT_DIR/test_report.csv" ]]; then
    echo
    echo "Top 5 similarity matches:"
    head -6 "$OUTPUT_DIR/test_report.csv" | column -t -s',' -N "Go_Test,Python_Test,Score,Shared_Signals"
    echo
    
    # Show matches with shared signals
    print_status "Matches with shared operations (first 5):"
    matches_with_signals=$(grep -v ',$' "$OUTPUT_DIR/test_report.csv" | head -6)
    if [[ -n "$matches_with_signals" ]]; then
        echo "$matches_with_signals" | column -t -s',' -N "Go_Test,Python_Test,Score,Shared_Signals"
    else
        echo "No matches found with shared operations"
    fi
    echo
fi

# Step 8: Test Suite Analysis
print_status "Step 8: Analyzing individual test suites..."
if [[ -f "$OUTPUT_DIR/go_specs.jsonl" && -f "$OUTPUT_DIR/py_specs.jsonl" ]]; then
    print_status "Running per-suite analysis and comparisons..."
    if python match/analyze_test_suites.py --go "$OUTPUT_DIR/go_specs.jsonl" --py "$OUTPUT_DIR/py_specs.jsonl" --output "$OUTPUT_DIR" --compare; then
        print_success "Test suite analysis completed"
        print_status "Analysis files generated:"
        # List individual suite analysis files
        for file in "$OUTPUT_DIR"/*_analysis.json; do
            if [[ -f "$file" ]]; then
                suite_name=$(basename "$file" _analysis.json)
                echo "  - $file ($suite_name analysis)"
            fi
        done
        if [[ -f "$OUTPUT_DIR/all_suite_comparisons.json" ]]; then
            echo "  - $OUTPUT_DIR/all_suite_comparisons.json (All suite comparisons)"
        fi
        
        # Generate high-level reports
        print_status "Generating high-level reports for test suite owners..."
        for file in "$OUTPUT_DIR"/*_analysis.json; do
            if [[ -f "$file" ]]; then
                suite_name=$(basename "$file" _analysis.json)
                report_file="$OUTPUT_DIR/${suite_name}_report.md"
                if python match/generate_suite_report.py "$file" -o "$report_file"; then
                    echo "  - $report_file ($suite_name report)"
                else
                    print_warning "Failed to generate report for $suite_name"
                fi
            fi
        done
        
        # Generate similarity matches report
        print_status "Generating similarity matches report..."
        if [[ -f "$OUTPUT_DIR/test_report.csv" && -f "$OUTPUT_DIR/test_coverage.csv" && -f "$OUTPUT_DIR/go_specs.jsonl" && -f "$OUTPUT_DIR/py_specs.jsonl" ]]; then
            similarity_report="$OUTPUT_DIR/similarity_report.md"
            if python match/generate_similarity_report.py \
                --test-report "$OUTPUT_DIR/test_report.csv" \
                --test-coverage "$OUTPUT_DIR/test_coverage.csv" \
                --go-specs "$OUTPUT_DIR/go_specs.jsonl" \
                --py-specs "$OUTPUT_DIR/py_specs.jsonl" \
                -o "$similarity_report"; then
                echo "  - $similarity_report (similarity analysis)"
            else
                print_warning "Failed to generate similarity report"
            fi
        else
            print_warning "Missing files for similarity report generation"
        fi
    else
        print_warning "Test suite analysis failed, but continuing..."
    fi
else
    print_warning "Skipping test suite analysis (spec files not found)"
fi

# Step 9: Cleanup (optional)
if [[ "$CLEANUP" == "true" ]]; then
    print_status "Step 9: Cleaning up intermediate files..."
    # Only remove Go binary if it was built during this run
    if [[ "$NEED_BUILD" == "true" ]]; then
        rm -f go-extractor/kubespec-go
        print_status "Removed Go binary (was built during this run)"
    else
        print_status "Preserving Go binary (was already up to date)"
    fi
    rm -f "$OUTPUT_DIR"/go_specs_*.jsonl
    rm -f "$OUTPUT_DIR"/py_specs_*.jsonl
    # Remove old combined suite files (we only generate individual suite reports)
    rm -f "$OUTPUT_DIR"/go_suite_analysis.json "$OUTPUT_DIR"/py_suite_analysis.json
    rm -f "$OUTPUT_DIR"/go_suite_report.md "$OUTPUT_DIR"/py_suite_report.md
    print_success "Cleanup completed (virtual environment preserved)"
else
    print_warning "Skipping cleanup (intermediate files preserved)"
fi

# Final output
echo
print_success "Pipeline completed successfully!"
print_status "Output files:"
echo "  - $OUTPUT_DIR/go_specs.jsonl (Combined Go test specs)"
echo "  - $OUTPUT_DIR/py_specs.jsonl (Combined Python test specs)"
echo "  - $OUTPUT_DIR/test_report.csv (Similarity matches)"
echo "  - $OUTPUT_DIR/test_coverage.csv (Coverage matrix)"
echo "  - $OUTPUT_DIR/*_analysis.json (Individual test suite analyses)"
echo "  - $OUTPUT_DIR/*_report.md (High-level reports for test suite owners)"
echo "  - $OUTPUT_DIR/similarity_report.md (Similarity matches analysis)"
echo "  - $OUTPUT_DIR/all_suite_comparisons.json (All suite comparisons)"

if [[ "$CLEANUP" == "false" ]]; then
    print_status "Individual repository files (preserved):"
    for i in "${!GO_ROOTS[@]}"; do
        repo_name=$(basename "${GO_ROOTS[$i]}")
        echo "  - $OUTPUT_DIR/go_specs_${i}_${repo_name}.jsonl"
    done
    for i in "${!PY_ROOTS[@]}"; do
        repo_name=$(basename "${PY_ROOTS[$i]}")
        echo "  - $OUTPUT_DIR/py_specs_${i}_${repo_name}.jsonl"
    done
fi

if [[ "$VERBOSE" == "true" ]]; then
    print_status "Verbose mode: Detailed logs available above"
fi

echo
print_success "Test spec extraction and matching pipeline completed!"
