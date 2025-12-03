#!/bin/bash

# extract-spec-md.sh - Ginkgo Spec Markdown Generator with Intelligent Similarity Analysis
# This script generates per-file Ginkgo markdown specs from Go test repositories and performs
# markdown-aware similarity analysis using hierarchical BDD context and semantic embeddings

set -e  # Exit on any error

# Store project root directory at script start
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GO_ROOTS=()
PY_ROOTS=()
OUTPUT_DIR="spec-md"
VERBOSE=false
EXPAND_FUNCTIONS=true
EXPORT_EXPANDED=false
EXPANDED_OUTPUT_DIR=""

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
    -p, --py-root PATH      Path to Python test repository (can be used multiple times)
    -o, --output-dir DIR    Output directory for results (default: spec-md)
    -v, --verbose           Verbose output
    -e, --expand-functions  Expand function calls up to k8s/ocp calls (default: true)
    -x, --export-expanded   Export expanded code to individual files
    --expanded-output-dir DIR  Output directory for expanded code files (default: {output-dir}/expanded_code/)
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
    - {output-dir}/go_specs_per_it.jsonl - Per-test specifications in JSONL format (Go tests)
    - {output-dir}/py_specs_per_it.jsonl - Per-test specifications in JSONL format (Python tests)
    - {output-dir}/markdown_similarity_results.csv - Detailed similarity analysis
    - {output-dir}/similarity_analysis.md - Human-readable similarity report

EXAMPLES:
    # Single repository with full analysis
    ./extract-spec-md.sh -g /path/to/eco-gotests
    
    # Multiple Go repositories for comprehensive analysis
    ./extract-spec-md.sh -g /path/to/eco-gotests -g /path/to/openshift-tests -g /path/to/cnf-gotests
    
    # Python repository
    ./extract-spec-md.sh -p /path/to/eco-pytests
    
    # Both Go and Python repositories for cross-language similarity
    ./extract-spec-md.sh -g /path/to/eco-gotests -p /path/to/eco-pytests
    
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
        -p|--py-root)
            PY_ROOTS+=("$2")
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
        -e|--expand-functions)
            EXPAND_FUNCTIONS=true
            shift
            ;;
        -x|--export-expanded)
            EXPORT_EXPANDED=true
            shift
            ;;
        --expanded-output-dir)
            EXPANDED_OUTPUT_DIR="$2"
            shift 2
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
if [[ ${#GO_ROOTS[@]} -eq 0 && ${#PY_ROOTS[@]} -eq 0 ]]; then
    print_error "At least one --go-root or --py-root is required"
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

print_status "Starting Spec Markdown Generator"
if [[ ${#GO_ROOTS[@]} -gt 0 ]]; then
    print_status "Go repositories: ${#GO_ROOTS[@]}"
    for go_root in "${GO_ROOTS[@]}"; do
        print_status "  - $go_root"
    done
fi
if [[ ${#PY_ROOTS[@]} -gt 0 ]]; then
    print_status "Python repositories: ${#PY_ROOTS[@]}"
    for py_root in "${PY_ROOTS[@]}"; do
        print_status "  - $py_root"
    done
fi
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

# Step 2b: Generate per-file Python markdown specs (if Python repos provided)
if [[ ${#PY_ROOTS[@]} -gt 0 ]]; then
    print_status "Step 2b: Generating per-file Python markdown specs..."
    
    # Per-It JSONL output for Python
    PY_PER_IT_JSONL="$OUTPUT_DIR/py_specs_per_it.jsonl"
    rm -f "$PY_PER_IT_JSONL"
    
    PY_TOTAL_FILES=0
    for i in "${!PY_ROOTS[@]}"; do
        py_root="${PY_ROOTS[$i]}"
        # Extract repository name: if path ends with /src/eco_pytests, use parent dir name
        # Otherwise use basename
        if [[ "$py_root" == */src/eco_pytests ]]; then
            repo_name=$(basename "$(dirname "$(dirname "$py_root")")")
        else
            repo_name=$(basename "$py_root")
        fi
        # Normalize: replace underscores with hyphens for consistency
        repo_name=$(echo "$repo_name" | sed 's/_/-/g')
        outdir="$OUTPUT_DIR/$repo_name"
        print_status "  Processing $repo_name..."
        
        # Use absolute paths to avoid issues when changing directories
        abs_outdir=$(cd "$(dirname "$outdir")" && pwd)/$(basename "$outdir")
        abs_jsonl=$(cd "$(dirname "$PY_PER_IT_JSONL")" && pwd)/$(basename "$PY_PER_IT_JSONL")
        
        # Build command with expansion flags for Python
        PY_EXPAND_FLAGS=""
        if [[ "$EXPAND_FUNCTIONS" == "true" ]]; then
            PY_EXPAND_FLAGS="--expand-functions"
        fi
        if [[ "$EXPORT_EXPANDED" == "true" ]]; then
            PY_EXPAND_FLAGS="$PY_EXPAND_FLAGS --export-expanded"
            if [[ -n "$EXPANDED_OUTPUT_DIR" ]]; then
                PY_EXPAND_FLAGS="$PY_EXPAND_FLAGS --expanded-output-dir $EXPANDED_OUTPUT_DIR"
            else
                PY_EXPAND_FLAGS="$PY_EXPAND_FLAGS --expanded-output-dir $abs_outdir/expanded_code"
            fi
        fi
        
        if [[ "$VERBOSE" == "true" ]]; then
            if ! (cd py-extractor && python3 -m spec_extractor.main --root "$py_root" --out "$abs_outdir" --jsonl "$abs_jsonl" $PY_EXPAND_FLAGS); then
                print_warning "  Failed to render markdown for $repo_name"
                cd ..
                continue
            fi
        else
            if ! (cd py-extractor && python3 -m spec_extractor.main --root "$py_root" --out "$abs_outdir" --jsonl "$abs_jsonl" $PY_EXPAND_FLAGS >/dev/null 2>&1); then
                print_warning "  Failed to render markdown for $repo_name"
                cd ..
                continue
            fi
        fi
        
        cd ..
        
        # Count generated files (use absolute path since we're back in project root)
        if [[ -d "$abs_outdir" ]]; then
            repo_files=$(find "$abs_outdir" -name "*.md" | wc -l)
            PY_TOTAL_FILES=$((PY_TOTAL_FILES + repo_files))
            print_success "  Generated $repo_files markdown files for $repo_name"
        fi
    done
    
    TOTAL_FILES=$((TOTAL_FILES + PY_TOTAL_FILES))
fi

# Step 3: Markdown-Aware Similarity Analysis
print_status "Step 3: Running markdown-aware similarity analysis..."

# Use absolute paths for JSONL files to avoid path resolution issues
# Convert OUTPUT_DIR to absolute path if it's relative
if [[ "$OUTPUT_DIR" != /* ]]; then
    # Relative path - make it absolute from project root
    ABS_OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
else
    # Already absolute
    ABS_OUTPUT_DIR="$OUTPUT_DIR"
fi
GO_JSONL="$ABS_OUTPUT_DIR/go_specs_per_it.jsonl"
PY_JSONL="$ABS_OUTPUT_DIR/py_specs_per_it.jsonl"

# Combine JSONL files if both exist
COMBINED_JSONL=""
if [[ -f "$GO_JSONL" && -f "$PY_JSONL" ]]; then
    COMBINED_JSONL="$ABS_OUTPUT_DIR/all_specs_per_it.jsonl"
    cat "$GO_JSONL" "$PY_JSONL" > "$COMBINED_JSONL"
    print_status "Combined Go and Python JSONL files for similarity analysis"
elif [[ -f "$GO_JSONL" ]]; then
    COMBINED_JSONL="$GO_JSONL"
elif [[ -f "$PY_JSONL" ]]; then
    COMBINED_JSONL="$PY_JSONL"
fi

if [[ -n "$COMBINED_JSONL" && -f "$COMBINED_JSONL" && -s "$COMBINED_JSONL" ]]; then
    PER_IT_COUNT=$(wc -l < "$COMBINED_JSONL")
    print_success "Generated $PER_IT_COUNT per-It test specs"
    
    # Run the new markdown-aware similarity analysis
    print_status "Step 3b: Performing intelligent similarity analysis with BDD context..."
    # Use PROJECT_ROOT to find match directory
    MATCH_DIR="$PROJECT_ROOT/match"
    if [[ -d "$MATCH_DIR" ]]; then
        cd "$MATCH_DIR"
    else
        print_warning "match directory not found at $MATCH_DIR, skipping similarity analysis"
        # Don't exit, just skip similarity analysis
        MATCH_DIR=""
    fi
    
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
    
    if [[ -n "$MATCH_DIR" && -d "$MATCH_DIR/.venv" && -f "$MATCH_DIR/.venv/bin/activate" ]]; then
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
                ALL_REPO_ROOTS=("${GO_ROOTS[@]}" "${PY_ROOTS[@]}")
                if python markdown-similarity.py \
                    --jsonl "$COMBINED_JSONL" \
                    --markdown "$OUTPUT_DIR/" \
                    --output "$OUTPUT_DIR/markdown_similarity_results.csv" \
                    --repo-roots "${ALL_REPO_ROOTS[@]}" \
                    --threshold 0.75 \
                    --top-k 10 \
                    --exclude-same-file; then
                    
                    # Generate summary report
                    print_status "Generating markdown similarity summary..."
                    cd "$MATCH_DIR"
                    source .venv/bin/activate
                    cd "$PROJECT_ROOT"
                    
                    # Create summary generation script with match type analysis (per AGENTS.md)
                    cat > "$OUTPUT_DIR/generate_summary.py" << 'EOF'
import pandas as pd
import sys

def is_python_file(file_path):
    return str(file_path).endswith('.py')

def is_go_file(file_path):
    return str(file_path).endswith('.go')

def get_match_type(row):
    query_py = is_python_file(row['query_file'])
    query_go = is_go_file(row['query_file'])
    matched_py = is_python_file(row['matched_file'])
    matched_go = is_go_file(row['matched_file'])
    
    if query_py and matched_py:
        return 'Python↔Python'
    elif query_go and matched_go:
        return 'Go↔Go'
    elif (query_py and matched_go) or (query_go and matched_py):
        return 'Python↔Go'
    else:
        return 'Unknown'

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
    
    # Add match type column if not present (per AGENTS.md: Language-Agnostic Analysis)
    if 'is_cross_language' in df.columns:
        df['match_type'] = df.apply(
            lambda row: 'Python↔Go' if row['is_cross_language'] else 
                       ('Python↔Python' if is_python_file(row['query_file']) else 'Go↔Go'),
            axis=1
        )
    else:
        df['match_type'] = df.apply(get_match_type, axis=1)
    
    # Calculate statistics
    avg_similarity = df['semantic_similarity'].mean()
    high_similarity = df[df['semantic_similarity'] > 0.9]
    medium_similarity = df[(df['semantic_similarity'] > 0.8) & (df['semantic_similarity'] <= 0.9)]
    
    # Match type distribution (per AGENTS.md: Match Type Distribution)
    match_type_counts = df['match_type'].value_counts()
    total_matches = len(df)
    intra_language = df[df['match_type'].isin(['Python↔Python', 'Go↔Go'])]
    cross_language = df[df['match_type'] == 'Python↔Go']
    
    with open(output_file, 'w') as f:
        f.write("# Cross-File BDD Test Similarity Analysis Results\n\n")
        f.write(f"**Analysis Type:** Language-agnostic similarity analysis\n\n")
        f.write(f"## Executive Summary\n\n")
        f.write(f"- **Total similarity matches**: {total_matches}\n")
        f.write(f"- **Average semantic similarity**: {avg_similarity:.3f}\n")
        f.write(f"- **High similarity matches (>0.9)**: {len(high_similarity)} (potential duplicates)\n")
        f.write(f"- **Medium similarity matches (0.8-0.9)**: {len(medium_similarity)} (related tests)\n")
        f.write(f"- **Files with similar tests**: {df['query_file'].nunique()}\n")
        f.write(f"- **Note**: Same-file matches are excluded to focus on meaningful cross-file similarities\n\n")
        
        # Match Type Analysis (per AGENTS.md requirement)
        f.write("## 🔄 Match Type Analysis\n\n")
        f.write("### Match Type Distribution\n\n")
        for match_type, count in match_type_counts.items():
            percentage = (count / total_matches * 100) if total_matches > 0 else 0
            f.write(f"- **{match_type}**: {count} matches ({percentage:.1f}%)\n")
        
        f.write(f"\n### Intra-Language vs Cross-Language\n\n")
        intra_pct = (len(intra_language) / total_matches * 100) if total_matches > 0 else 0
        cross_pct = (len(cross_language) / total_matches * 100) if total_matches > 0 else 0
        f.write(f"- **Intra-language matches** (Python↔Python, Go↔Go): {len(intra_language)} ({intra_pct:.1f}%)\n")
        f.write(f"- **Cross-language matches** (Python↔Go): {len(cross_language)} ({cross_pct:.1f}%)\n")
        
        if len(intra_language) > 0:
            f.write(f"\n  - Average intra-language similarity: {intra_language['semantic_similarity'].mean():.3f}\n")
        if len(cross_language) > 0:
            f.write(f"\n  - Average cross-language similarity: {cross_language['semantic_similarity'].mean():.3f}\n")
        
        # Score Distribution (per AGENTS.md: Similarity Report Generation)
        f.write(f"\n### Score Distribution\n\n")
        f.write(f"- **Perfect duplicates (1.0)**: {len(df[df['semantic_similarity'] == 1.0])}\n")
        f.write(f"- **Very high similarity (0.95-0.99)**: {len(df[(df['semantic_similarity'] >= 0.95) & (df['semantic_similarity'] < 1.0)])}\n")
        f.write(f"- **High similarity (0.90-0.94)**: {len(df[(df['semantic_similarity'] >= 0.90) & (df['semantic_similarity'] < 0.95)])}\n")
        f.write(f"- **Medium similarity (0.80-0.89)**: {len(df[(df['semantic_similarity'] >= 0.80) & (df['semantic_similarity'] < 0.90)])}\n")
        f.write(f"- **Lower similarity (0.65-0.79)**: {len(df[(df['semantic_similarity'] >= 0.65) & (df['semantic_similarity'] < 0.80)])}\n")
        
        # Strategic Recommendations (per AGENTS.md)
        f.write(f"\n### Strategic Recommendations\n\n")
        if len(high_similarity) > 0:
            f.write(f"- **{len(high_similarity)} potential duplicates** (>0.9 similarity) - consider consolidation\n")
        if len(cross_language) > 0:
            f.write(f"- **{len(cross_language)} cross-language matches** - opportunities for test pattern sharing between Python and Go implementations\n")
        if len(intra_language) > len(cross_language) * 10:
            f.write(f"- More intra-language matches found - potential for test consolidation within same language\n")
        elif len(cross_language) > len(intra_language) * 0.1:
            f.write(f"- Good cross-pollination opportunities between languages\n")
        
        if len(high_similarity) > 0:
            f.write(f"\n## Potential Test Duplicates (>0.9 similarity)\n\n")
            f.write("| Test 1 | Test 2 | Similarity | Match Type | File 1 | File 2 |\n")
            f.write("|--------|--------|------------|------------|--------|--------|\n")
            
            for _, row in high_similarity.head(15).iterrows():
                f1 = row['query_file']
                f2 = row['matched_file']
                match_type = row.get('match_type', get_match_type(row))
                
                query_desc = str(row['query_description']) if pd.notna(row['query_description']) else "No description"
                matched_desc = str(row['matched_description']) if pd.notna(row['matched_description']) else "No description"
                
                if query_desc == matched_desc:
                    query_id = str(row['query_test_id']) if pd.notna(row['query_test_id']) else "unknown"
                    matched_id = str(row['matched_test_id']) if pd.notna(row['matched_test_id']) else "unknown"
                    query_display = f"{query_desc} ({query_id})"
                    matched_display = f"{matched_desc} ({matched_id})"
                else:
                    query_display = query_desc
                    matched_display = matched_desc
                
                query_trunc = query_display[:40] + "..." if len(query_display) > 40 else query_display
                matched_trunc = matched_display[:40] + "..." if len(matched_display) > 40 else matched_display
                
                f.write(f"| {query_trunc} | {matched_trunc} | {row['semantic_similarity']:.3f} | {match_type} | {f1} | {f2} |\n")
        
        # Top Cross-Language Matches (if any)
        if len(cross_language) > 0:
            f.write(f"\n## Top Cross-Language Matches (Python↔Go)\n\n")
            f.write("| Python Test | Go Test | Similarity |\n")
            f.write("|-------------|---------|------------|\n")
            top_cross = cross_language.nlargest(10, 'semantic_similarity')
            for _, row in top_cross.iterrows():
                py_desc = str(row['query_description'] if is_python_file(row['query_file']) else row['matched_description'])[:50]
                go_desc = str(row['matched_description'] if is_python_file(row['query_file']) else row['query_description'])[:50]
                f.write(f"| {py_desc}... | {go_desc}... | {row['semantic_similarity']:.3f} |\n")

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
    echo "  - $OUTPUT_DIR/go_specs_per_it.jsonl (Per-It test specs in JSONL format - Go)"
fi
if [[ -f "$OUTPUT_DIR/py_specs_per_it.jsonl" ]]; then
    echo "  - $OUTPUT_DIR/py_specs_per_it.jsonl (Per-It test specs in JSONL format - Python)"
fi
if [[ -f "$OUTPUT_DIR/all_specs_per_it.jsonl" ]]; then
    echo "  - $OUTPUT_DIR/all_specs_per_it.jsonl (Combined Go and Python specs)"
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
        echo "  - $repo_name (Go): $repo_files files"
    fi
done
for i in "${!PY_ROOTS[@]}"; do
    py_root="${PY_ROOTS[$i]}"
    # Extract repository name: if path ends with /src/eco_pytests, use parent dir name
    # Otherwise use basename
    if [[ "$py_root" == */src/eco_pytests ]]; then
        repo_name=$(basename "$(dirname "$(dirname "$py_root")")")
    else
        repo_name=$(basename "$py_root")
    fi
    # Normalize: replace underscores with hyphens for consistency
    repo_name=$(echo "$repo_name" | sed 's/_/-/g')
    outdir="$OUTPUT_DIR/$repo_name"
    if [[ -d "$outdir" ]]; then
        repo_files=$(find "$outdir" -name "*.md" | wc -l)
        echo "  - $repo_name (Python): $repo_files files"
    fi
done

echo
print_success "Total: $TOTAL_FILES markdown files generated"

if [[ "$VERBOSE" == "true" ]]; then
    print_status "Verbose mode: Detailed logs available above"
fi

echo
print_success "Spec markdown generation completed!"