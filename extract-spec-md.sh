#!/bin/bash

# extract-spec-md.sh - Ginkgo Spec Markdown Generator with Intelligent Similarity Analysis
# This script generates per-file Ginkgo markdown specs from Go test repositories and performs
# markdown-aware similarity analysis using hierarchical BDD context and semantic embeddings

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

Ginkgo Spec Markdown Generator with Intelligent Similarity Analysis

This script performs a complete workflow:
1. Extracts BDD specifications from Ginkgo test files 
2. Generates structured markdown documentation
3. Performs intelligent similarity analysis using hierarchical context

OPTIONS:
    -g, --go-root PATH      Path to Go test repository (can be used multiple times)
    -o, --output-dir DIR    Output directory for results (default: spec-md)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

FEATURES:
    ✓ Enhanced BDD pattern extraction (Describe, Context, When, It, Specify, DescribeTable, Entry)
    ✓ Semantic organization (preparation, steps, cleanup sections)
    ✓ Skip conditions as negative prerequisites  
    ✓ Markdown-aware similarity analysis with hierarchical context
    ✓ FAISS-powered semantic embeddings for intelligent test matching
    ✓ Duplicate detection and test consolidation recommendations

OUTPUT FILES:
    - {output-dir}/ - Structured markdown specs organized by repository
    - {output-dir}/go_specs_per_it.jsonl - Per-test specifications in JSONL format
    - {output-dir}/markdown_similarity_results.csv - Detailed similarity analysis
    - {output-dir}/similarity_analysis.md - Human-readable similarity report

EXAMPLES:
    # Single repository with full analysis
    ./extract-spec-md.sh -g /path/to/eco-gotests
    
    # Multiple Go repositories for comprehensive analysis
    ./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -g /path/to/cnf-gotests
    
    # Custom output directory
    ./extract-spec-md.sh --go-root /path/to/go-tests --output-dir my_analysis

SIMILARITY ANALYSIS:
    The integrated similarity analysis uses advanced techniques:
    - Each test becomes a single document in FAISS with combined context
    - Hierarchical BDD structure (Describe > Context > When > It) 
    - Semantic embeddings of test descriptions, steps, and prerequisites
    - Context-aware similarity scoring with container hierarchy
    - Identification of potential duplicates and related tests

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

# Step 3: Markdown-Aware Similarity Analysis
print_status "Step 3: Running markdown-aware similarity analysis..."
if [[ -f "$OUTPUT_DIR/go_specs_per_it.jsonl" ]]; then
    PER_IT_COUNT=$(wc -l < "$OUTPUT_DIR/go_specs_per_it.jsonl")
    print_success "Generated $PER_IT_COUNT per-It test specs"
    
    # Run the new markdown-aware similarity analysis
    print_status "Step 3b: Performing intelligent similarity analysis with BDD context..."
    cd match
    
    # Setup Python environment
    if [[ ! -d ".venv" ]]; then
        print_status "Creating Python virtual environment..."
        if ! python -m venv .venv; then
            print_warning "Failed to create virtual environment, skipping similarity analysis"
            cd ..
        else
            print_status "Virtual environment created"
        fi
    fi
    
    if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
        print_status "Activating virtual environment..."
        if source .venv/bin/activate; then
            # Check if dependencies are available
            if ! python -c "import sentence_transformers, faiss, pandas, numpy" > /dev/null 2>&1; then
                print_status "Installing Python dependencies for similarity analysis..."
                if ! pip install --upgrade pip > /dev/null 2>&1; then
                    print_warning "Failed to upgrade pip, skipping similarity analysis"
                    cd ..
                elif ! pip install -r requirements.txt > /dev/null 2>&1; then
                    print_warning "Failed to install dependencies, skipping similarity analysis"
                    cd ..
                else
                    print_success "Dependencies installed"
                fi
            fi
            
            # Run the new markdown-aware similarity analysis
            if python -c "import sentence_transformers, faiss, pandas, numpy" > /dev/null 2>&1; then
                cd ..
                print_status "Running markdown-aware similarity analysis..."
                
                # Run with both JSONL and markdown data for enhanced context
                if python markdown-similarity.py \
                    --jsonl "$OUTPUT_DIR/go_specs_per_it.jsonl" \
                    --markdown "$OUTPUT_DIR/" \
                    --output "$OUTPUT_DIR/markdown_similarity_results.csv" \
                    --repo-roots "${GO_ROOTS[@]}" \
                    --threshold 0.75 \
                    --top-k 10 \
                    --exclude-same-file; then
                    
                    # Generate summary report
                    print_status "Generating markdown similarity summary..."
                    cd match
                    source .venv/bin/activate
                    cd ..
                    
                    # Create summary generation script
                    cat > "$OUTPUT_DIR/generate_summary.py" << 'EOF'
import pandas as pd
import sys

def generate_summary(csv_file, output_file):
    try:
        df = pd.read_csv(csv_file)
    except:
        with open(output_file, 'w') as f:
            f.write("# Similarity Analysis Results\n\nNo similarity data found.\n")
        return
    
    if len(df) == 0:
        with open(output_file, 'w') as f:
            f.write("# Similarity Analysis Results\n\nNo similarities found above threshold.\n")
        return
    
    # Calculate statistics
    avg_similarity = df['semantic_similarity'].mean()
    high_similarity = df[df['semantic_similarity'] > 0.9]
    medium_similarity = df[(df['semantic_similarity'] > 0.8) & (df['semantic_similarity'] <= 0.9)]
    
    with open(output_file, 'w') as f:
        f.write("# Cross-File BDD Test Similarity Analysis Results\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total similarity matches**: {len(df)}\n")
        f.write(f"- **Average semantic similarity**: {avg_similarity:.3f}\n")
        f.write(f"- **High similarity matches (>0.9)**: {len(high_similarity)} (potential duplicates)\n")
        f.write(f"- **Medium similarity matches (0.8-0.9)**: {len(medium_similarity)} (related tests)\n")
        f.write(f"- **Files with similar tests**: {df['query_file'].nunique()}\n")
        f.write(f"- **Note**: Same-file matches are excluded to focus on meaningful cross-file similarities\n\n")
        
        if len(high_similarity) > 0:
            f.write("## Potential Test Duplicates (>0.9 similarity)\n\n")
            f.write("| Test 1 | Test 2 | Similarity | File 1 | File 2 |\n")
            f.write("|--------|--------|------------|--------|--------|\n")
            
            for _, row in high_similarity.head(15).iterrows():
                # Use the full normalized paths from CSV for consistency
                f1 = row['query_file']
                f2 = row['matched_file']
                
                # Handle potentially missing descriptions and create meaningful differentiators
                query_desc = str(row['query_description']) if pd.notna(row['query_description']) else "No description"
                matched_desc = str(row['matched_description']) if pd.notna(row['matched_description']) else "No description"
                
                # If descriptions are identical, differentiate by test ID or context
                if query_desc == matched_desc:
                    query_id = str(row['query_test_id']) if pd.notna(row['query_test_id']) else "unknown"
                    matched_id = str(row['matched_test_id']) if pd.notna(row['matched_test_id']) else "unknown"
                    query_display = f"{query_desc} ({query_id})"
                    matched_display = f"{matched_desc} ({matched_id})"
                else:
                    query_display = query_desc
                    matched_display = matched_desc
                
                # Truncate for display but ensure they're different
                query_trunc = query_display[:40] + "..." if len(query_display) > 40 else query_display
                matched_trunc = matched_display[:40] + "..." if len(matched_display) > 40 else matched_display
                
                f.write(f"| {query_trunc} | {matched_trunc} | {row['semantic_similarity']:.3f} | {f1} | {f2} |\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_summary.py input.csv output.md")
        sys.exit(1)
    generate_summary(sys.argv[1], sys.argv[2])
EOF
                    
                    if python "$OUTPUT_DIR/generate_summary.py" "$OUTPUT_DIR/markdown_similarity_results.csv" "$OUTPUT_DIR/similarity_analysis.md"; then
                        print_success "Generated $OUTPUT_DIR/similarity_analysis.md (BDD similarity report)"
                        rm "$OUTPUT_DIR/generate_summary.py"  # Cleanup
                    else
                        print_warning "Failed to generate similarity summary"
                        rm "$OUTPUT_DIR/generate_summary.py"  # Cleanup
                    fi
                else
                    print_warning "Failed to run markdown-aware similarity analysis"
                fi
            else
                print_warning "Python dependencies not available, skipping similarity analysis"
                cd ..
            fi
        else
            print_warning "Failed to activate virtual environment, skipping similarity analysis"
            cd ..
        fi
    else
        print_warning "Virtual environment setup failed, skipping similarity analysis"
    fi
else
    print_warning "No per-It specs generated, skipping similarity analysis"
fi

# Final output
echo
print_success "Spec markdown generation completed!"
print_status "Output files:"
echo "  - $OUTPUT_DIR/ (Markdown specs organized by repository)"
if [[ -f "$OUTPUT_DIR/go_specs_per_it.jsonl" ]]; then
    echo "  - $OUTPUT_DIR/go_specs_per_it.jsonl (Per-It test specs in JSONL format)"
fi
if [[ -f "$OUTPUT_DIR/markdown_similarity_results.csv" ]]; then
    echo "  - $OUTPUT_DIR/markdown_similarity_results.csv (Detailed similarity analysis)"
fi
if [[ -f "$OUTPUT_DIR/similarity_analysis.md" ]]; then
    echo "  - $OUTPUT_DIR/similarity_analysis.md (BDD similarity report)"
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