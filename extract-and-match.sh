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

# Step 8: Cleanup (optional)
if [[ "$CLEANUP" == "true" ]]; then
    print_status "Step 8: Cleaning up intermediate files..."
    # Only remove Go binary if it was built during this run
    if [[ "$NEED_BUILD" == "true" ]]; then
        rm -f go-extractor/kubespec-go
        print_status "Removed Go binary (was built during this run)"
    else
        print_status "Preserving Go binary (was already up to date)"
    fi
    rm -f "$OUTPUT_DIR"/go_specs_*.jsonl
    rm -f "$OUTPUT_DIR"/py_specs_*.jsonl
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
