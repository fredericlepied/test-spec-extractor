#!/bin/bash

# extract-spec-md.sh - Ginkgo Spec Markdown Generator
# This script generates per-file Ginkgo markdown specs from Go test repositories

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GO_ROOTS=()
OUTPUT_DIR="spec-md"
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

Ginkgo Spec Markdown Generator

OPTIONS:
    -g, --go-root PATH      Path to Go test repository (can be used multiple times)
    -o, --output-dir DIR    Output directory for results (default: spec-md)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Single repository
    ./extract-spec-md.sh -g /path/to/eco-gotests
    
    # Multiple Go repositories
    ./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/openshift-tests
    
    # Custom output directory
    ./extract-spec-md.sh --go-root /path/to/go-tests --output-dir my_spec_md

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--go-root)
            GO_ROOTS+=("$2")
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
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
if [[ ${#GO_ROOTS[@]} -eq 0 ]]; then
    print_error "At least one --go-root is required"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

print_status "Starting Ginkgo Spec Markdown Generator"
print_status "Go repositories: ${#GO_ROOTS[@]}"
for go_root in "${GO_ROOTS[@]}"; do
    print_status "  - $go_root"
done
print_status "Output directory: $OUTPUT_DIR"
echo

# Step 1: Build spec-extractor if needed
print_status "Step 1: Checking spec-extractor..."
cd go-extractor

# Check if binary exists and is newer than source files
NEED_BUILD_SPEC=false
if [[ ! -f "spec-extractor/spec-extractor" ]]; then
    print_status "spec-extractor binary not found, will build..."
    NEED_BUILD_SPEC=true
else
    if find spec-extractor -type f \( -name '*.go' -o -name 'go.mod' -o -name 'go.sum' \) -newer spec-extractor/spec-extractor | grep -q .; then
        print_status "spec-extractor source files newer than binary, will rebuild..."
        NEED_BUILD_SPEC=true
    else
        print_status "spec-extractor binary is up to date"
    fi
fi

if [[ "$NEED_BUILD_SPEC" == "true" ]]; then
    print_status "Building spec-extractor..."
    if ! (cd spec-extractor && go build -o spec-extractor); then
        print_error "Failed to build spec-extractor"
        exit 1
    fi
    print_success "spec-extractor built successfully"
else
    print_success "spec-extractor ready (using existing binary)"
fi

# Step 2: Generate per-file Ginkgo markdown specs
print_status "Step 2: Generating per-file Ginkgo markdown specs..."
mkdir -p "../$OUTPUT_DIR"

# Per-It JSONL output
GO_PER_IT_JSONL="../$OUTPUT_DIR/go_specs_per_it.jsonl"
rm -f "$GO_PER_IT_JSONL"

TOTAL_FILES=0
for i in "${!GO_ROOTS[@]}"; do
    go_root="${GO_ROOTS[$i]}"
    repo_name=$(basename "$go_root")
    outdir="../$OUTPUT_DIR/$repo_name"
    print_status "  Processing $repo_name..."
    
    if [[ "$VERBOSE" == "true" ]]; then
        if ! ./spec-extractor/spec-extractor --root "$go_root" --out "$outdir" --jsonl "$GO_PER_IT_JSONL"; then
            print_warning "  Failed to render markdown for $repo_name"
            continue
        fi
    else
        if ! ./spec-extractor/spec-extractor --root "$go_root" --out "$outdir" --jsonl "$GO_PER_IT_JSONL" >/dev/null 2>&1; then
            print_warning "  Failed to render markdown for $repo_name"
            continue
        fi
    fi
    
    # Count generated files
    if [[ -d "$outdir" ]]; then
        repo_files=$(find "$outdir" -name "*.md" | wc -l)
        TOTAL_FILES=$((TOTAL_FILES + repo_files))
        print_success "  Generated $repo_files markdown files for $repo_name"
    fi
done

cd ..

# Step 3: Analyze per-It specs
print_status "Step 3: Analyzing per-It specs..."
if [[ -f "$OUTPUT_DIR/go_specs_per_it.jsonl" ]]; then
    PER_IT_COUNT=$(wc -l < "$OUTPUT_DIR/go_specs_per_it.jsonl")
    print_success "Generated $PER_IT_COUNT per-It test specs"
    
    # Optional: Find similar Go tests (self-match) to find duplicates
    print_status "Step 3b: Finding similar Go tests (per-It analysis)..."
    cd match
    
    # Check if Python environment exists
    if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
        print_status "Using existing virtual environment..."
        if source .venv/bin/activate; then
            # Check if dependencies are available
            if python -c "import sentence_transformers, faiss, pandas, numpy" > /dev/null 2>&1; then
                if python build_index_and_match.py \
                    --go "../$OUTPUT_DIR/go_specs_per_it.jsonl" \
                    --py "../$OUTPUT_DIR/go_specs_per_it.jsonl" \
                    --out "../$OUTPUT_DIR/go_per_it_sim.csv" \
                    --cov "../$OUTPUT_DIR/go_per_it_cov.csv"; then
                    print_success "Generated $OUTPUT_DIR/go_per_it_sim.csv (per-It similarities)"
                else
                    print_warning "Failed to compute per-It similarities"
                fi
            else
                print_warning "Python dependencies not available, skipping similarity analysis"
            fi
        else
            print_warning "Failed to activate virtual environment, skipping similarity analysis"
        fi
    else
        print_warning "Virtual environment not found, skipping similarity analysis"
        print_status "Run ./extract-and-match.sh first to set up the environment"
    fi
    cd ..
else
    print_warning "No per-It specs generated"
fi

# Final output
echo
print_success "Spec markdown generation completed!"
print_status "Output files:"
echo "  - $OUTPUT_DIR/ (Markdown specs organized by repository)"
if [[ -f "$OUTPUT_DIR/go_specs_per_it.jsonl" ]]; then
    echo "  - $OUTPUT_DIR/go_specs_per_it.jsonl (Per-It test specs in JSONL format)"
fi
if [[ -f "$OUTPUT_DIR/go_per_it_sim.csv" ]]; then
    echo "  - $OUTPUT_DIR/go_per_it_sim.csv (Per-It similarity analysis)"
fi

echo
print_status "Generated markdown files by repository:"
for i in "${!GO_ROOTS[@]}"; do
    go_root="${GO_ROOTS[$i]}"
    repo_name=$(basename "$go_root")
    outdir="$OUTPUT_DIR/$repo_name"
    if [[ -d "$outdir" ]]; then
        repo_files=$(find "$outdir" -name "*.md" | wc -l)
        echo "  - $repo_name: $repo_files files"
    fi
done

echo
print_success "Total: $TOTAL_FILES markdown files generated"

if [[ "$VERBOSE" == "true" ]]; then
    print_status "Verbose mode: Detailed logs available above"
fi

echo
print_success "Ginkgo spec markdown generation completed!"