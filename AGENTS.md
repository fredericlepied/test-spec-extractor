# AI Coding Assistent Rules for Test Spec Extractor

## Continuous Improvement

**Always** review this document at the end of the developement of a change to improve the knowledge and practices.

## Design Principles

- TDD methodology
- Use the DRY principle
- don't create new Markdown files without asking the user
- keep the minimal documentation needed not an exhaustive one

## Documentation Maintenance

### README.md Guidelines

**CRITICAL: Keep README.md SHORT and FOCUSED**

- **Target length**: 250-300 lines maximum
- **Current length**: 274 lines (monitor to prevent bloat)
- **Principle**: Users want to get started quickly, not read a manual

**What to INCLUDE:**
- ✅ Brief description (1 paragraph)
- ✅ Key features (5-7 bullet points max)
- ✅ Quick start (installation + basic usage)
- ✅ What you get (output files)
- ✅ 2-3 practical examples max
- ✅ Basic troubleshooting (common issues only)

**What to EXCLUDE:**
- ❌ Multiple bash command variations (show 1 example, not 10)
- ❌ Deep technical explanations (save for inline code comments)
- ❌ Exhaustive lists of all possible options
- ❌ Detailed architecture explanations (brief overview only)
- ❌ Every possible use case (most common 2-3 only)
- ❌ Debug tools details (brief mention with reference)

**When adding content:**
1. Ask: "Does the user NEED this to get started?"
2. If NO → Don't add it
3. If YES → Can it be explained in <5 lines?
4. If adding, remove something of equal length elsewhere

**Monitoring:**
```bash
# Check README length regularly
wc -l README.md
# Should stay under 300 lines
```

**If README grows beyond 300 lines:**
- Review and remove verbose examples
- Consolidate redundant sections
- Move advanced topics to inline code documentation
- Prefer links to showing everything

## Code Formatting

### Python Code
- Always run `black` with line length 100 on Python files before committing
- Use `black py-extractor/ match/ --line-length 100`
- Follow PEP 8 style guidelines
- Use type hints where appropriate

### Go Code
- Always run `gofmt -w` on Go files before committing
- Use `gofmt -w go-extractor/main.go`
- Follow standard Go formatting conventions
- Use `go vet` to check for potential issues

## Code Quality

### General Rules
- Write clear, self-documenting code with meaningful variable names
- Add comments for complex logic and business rules
- Keep functions focused and single-purpose
- Use consistent error handling patterns

### Python Specific
- Use f-strings for string formatting
- Prefer list/dict comprehensions over loops when readable
- Use `pathlib.Path` for file operations
- Handle exceptions explicitly with try/except blocks

### Go Specific
- Use `context.Context` for long-running operations
- Return errors explicitly, don't panic
- Use meaningful variable names (avoid abbreviations)
- Group related declarations together

## Testing

### Test Structure
- Write tests for all public functions
- Use descriptive test names that explain the scenario
- Test both success and failure cases
- Mock external dependencies appropriately

### Test Data
- Use realistic test data that reflects real-world usage
- Keep test data minimal but comprehensive
- Use constants for repeated test values

## Documentation

### Code Comments
- Document public APIs with docstrings
- Explain complex algorithms and business logic
- Keep comments up-to-date with code changes
- Use TODO comments sparingly and with clear ownership

### README Updates
- Update README.md when adding new features
- Include usage examples for new functionality
- Keep installation and setup instructions current
- Document any breaking changes

## Git Workflow

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in imperative mood
- Include scope when relevant (e.g., "feat(extractor): add purpose detection")
- Reference issues when applicable

### Code Review
- Review all code changes before merging
- Check for formatting issues
- Verify tests pass
- Ensure documentation is updated

## Performance

### Optimization Guidelines
- Profile code before optimizing
- Focus on algorithmic improvements first
- Use appropriate data structures
- Consider memory usage for large datasets

### Resource Management
- Close files and connections properly
- Use context managers in Python
- Handle cleanup in defer statements in Go
- Monitor memory usage in long-running processes

## Security

### Input Validation
- Validate all external inputs
- Sanitize file paths and user data
- Use parameterized queries for database operations
- Handle sensitive data appropriately

### Dependencies
- Keep dependencies up-to-date
- Use dependency scanning tools
- Prefer well-maintained, popular packages
- Document any security-related dependencies

## Purpose-Based Filtering System

### Purpose Detection
- Add new purpose categories to both Go and Python extractors
- Update compatibility matrix when adding new purposes
- Test purpose detection with real test cases
- Document purpose patterns and keywords

### Matching Logic
- Maintain purpose compatibility rules
- Test filtering effectiveness with real data
- Monitor false positive rates
- Adjust scoring weights based on validation results

## OpenShift Integration

### Resource Mapping
- Keep Route↔Ingress mappings current
- Update SCC↔PSA mappings as needed
- Test with real OpenShift resources
- Document any new equivalence rules

### CLI Command Mapping
- Map new kubectl/oc commands to API operations
- Test command parsing with real examples
- Handle edge cases in command parsing
- Update documentation for new commands

## Comprehensive Similarity Analysis

### Language-Agnostic Analysis
- Compare all tests regardless of programming language (Go↔Go, Python↔Python, Go↔Python)
- Focus on functional similarity rather than language-specific patterns
- Identify true duplicates and consolidation opportunities
- Enable cross-pollination between different language implementations

### Match Type Distribution
- **Intra-language matches**: Find duplicates within same language (95.4% of matches)
- **Cross-language matches**: Identify patterns that can be shared between languages (4.6% of matches)
- **Perfect duplicates**: Detect 100% identical tests for immediate consolidation
- **Functional duplicates**: Find tests that differ only in parameters (96.2% similarity)

### Similarity Report Generation
- Generate comprehensive Markdown reports with match type analysis
- Include executive summary with quality indicators and duplicate ratios
- Provide strategic recommendations for test optimization
- Show score distribution and shared signals analysis
- Identify potential duplicates and complementary tests

### Validation and Quality
- Source code validation of similar tests to ensure accuracy
- Purpose-based filtering to reduce false positives
- Technology compatibility checking for meaningful matches
- Operation validation to confirm functional similarity

## Label Handling for Semantic Similarity

### Label Exclusion from Embeddings

**Decision**: Labels are **excluded from embeddings** but **preserved in metadata**.

**Rationale** (based on comprehensive analysis in `spec-md/label_recommendation.md`):
- **No cross-repository labels**: 0 labels appear in multiple repos
- **86.7% noise ratio**: 249/287 labels used ≤4 times (one-off identifiers, not categories)
- **File-specific organizational tags**: Labels are used for Ginkgo test filtering (`ginkgo --label-filter`), not semantic categorization
- **Inconsistent ontology**: Mixed kebab-case, CamelCase, and ad-hoc naming across repos

### Label Usage

**Labels ARE used for**:
- ✅ Filtering results by feature area (metadata filtering)
- ✅ Reporting label distribution in analysis
- ✅ CI/CD test selection and organization
- ✅ Intra-file test suite grouping

**Labels are NOT used for**:
- ❌ Semantic similarity embeddings (adds noise)
- ❌ Cross-file test categorization
- ❌ Multi-repo consistency matching

### Implementation Location

- **Extractor output**: Labels included in JSONL `labels` field and markdown `- **labels**:` sections
- **Metadata**: Labels preserved in `TestContext.container_labels` and `TestContext.test_labels`
- **Embeddings**: Labels excluded in `match/markdown-similarity.py::_create_combined_text()` (lines 366-371)

### Analysis Results

Based on extraction from 4 repositories:
- **287 unique labels** across 688 occurrences
- **Top label frequency**: "egress" (28 times, all in 1 file)
- **Single-use labels**: 160 (55.7%)
- **Rare labels (2-4 uses)**: 89 (31.0%)
- **Frequent labels (10+ uses)**: Only 9 (3.1%)

See `analyze_labels.py` for label extraction and `spec-md/label_analysis.json` for full frequency data.

## Cross-Repository Similarity Optimization

### Generic Assertion Filtering (Python)

**Problem**: Code-level implementation details pollute semantic embeddings, reducing similarity scores between functionally identical tests.

**Solution**: Pattern-based filtering at the embedding layer (not extraction layer).

**Implementation**:
- **File**: `match/markdown-similarity.py`
- **Functions**: `_is_generic_assertion()`, `_filter_generic_assertions()`
- **Location**: Applied in `_create_combined_text()` when creating embeddings
- **Principle**: Preserve raw data in JSONL, filter only for similarity matching

**Filtered Patterns** (generic noise):
```python
# Error handling
"operation succeeds without error"
"returns an error"

# Boolean checks (simple variables only)
"is true", "is false"

# Existence checks
"is nil", "exists", "is zero"

# Generic len() comparisons
"len(...) is greater than ..."
"len(...) has length ..."
```

**Preserved Patterns** (domain-specific signal):
```python
# Domain keyword protection list
domain_keywords = [
    "status", "state", "ready", "available", "healthy",
    "running", "pod", "node", "cluster", "operator", "service"
]

# Examples that are preserved:
"SR-IOV operator is ready"           # Has 'operator'
"pod status equals Running"          # Has 'pod', 'status'
"cluster ready is true"              # Has 'cluster', 'ready'
"Deploy SR-IOV operator"             # Specific action
```

**Key Principles**:
1. **Filter at the right layer**: Keep raw data intact, filter only for embeddings
2. **Conservative filtering**: Only remove obvious generic patterns
3. **Domain awareness**: Preserve anything with domain-specific keywords
4. **Pattern-based**: Use regex patterns, not hardcoded test suite names
5. **Testable**: Unit tests validate filtering logic (see `test_filtering.py`)

**Results**:
- 75 high-similarity matches (>0.90) between cnf-gotests ↔ eco-gotests
- Top matches >0.96 similarity (near-perfect duplicates)
- 27.4% of cross-repo matches are high-quality (>0.90)

### Custom Test Framework Wrapper Recognition (Go)

**Problem**: Test suites use custom wrapper functions (e.g., `compat_otp.By()`) that the extractor doesn't recognize, causing identical tests to have different extractions.

**Solution**: Generic AST pattern matching for any `package.Method()` pattern, not just known aliases.

**Implementation**:
- **File**: `go-extractor/spec-extractor/internal/extractor/visit.go`
- **Function**: `isCustomByCall()` (lines 222-248)
- **Integration**: `v.recog.IsBy(be) || isCustomByCall(be)` (line 148)

**Pattern Matched**:
```go
// Detects any selector expression with method name "By" and single string argument
sel, ok := call.Fun.(*ast.SelectorExpr)
sel.Sel.Name == "By"
len(call.Args) == 1
call.Args[0] is string literal
```

**Examples Recognized**:
- `compat_otp.By("step description")` ← Previously missed
- `utils.By("step description")`
- `helper.By("step description")`
- Any `<package>.By("string")` pattern

**Key Principles**:
1. **Generic pattern matching**: Match structural patterns, not specific package names
2. **No configuration needed**: Automatically works with any test suite
3. **Future-proof**: Based on AST structure, not hardcoded names
4. **Fallback approach**: Check custom patterns after standard recognition fails

**Results**:
- openshift-tests-private now extracts steps from `compat_otp.By()` calls
- +1-3 steps per test extracted that were previously missed
- Works with any future test suite using custom wrappers

### Design Principles for Similarity Optimization

**1. Preserve Raw Data, Filter for Analysis**
- JSONL contains all extracted information (unfiltered)
- Filtering applied only during embedding creation
- Enables different filtering strategies without re-extraction

**2. Pattern-Based Over Hardcoding**
- Use regex patterns and structural matching
- Avoid hardcoding specific repo names or function names
- Makes solution generic and future-proof

**3. Conservative Filtering**
- Only remove obvious noise
- When in doubt, preserve the data
- Use domain keyword lists to protect important context

**4. Validate with Unit Tests**
- Test filtering logic independently (see `test_filtering.py`)
- Validate with real-world examples from multiple repos
- Check for false positives (unrelated tests matching)

**5. Two-Layer Approach**
- **Extraction layer** (Go): Get all available information
- **Analysis layer** (Python): Filter for specific use cases
- Keeps extraction generic, allows analysis flexibility

### Monitoring and Maintenance

**Metrics to Track**:
- High-similarity match percentage (should stay >25%)
- Average cross-repo similarity (should stay >0.80)
- False positive rate (manual review of sample matches)

**Pattern Expansion**:
When new generic patterns are identified, add to `_is_generic_assertion()`:
```python
# New patterns to consider:
# - "err is nil" (but preserve "pod.Err is nil")
# - "result succeeds" (generic outcome)
# - "count equals N" (generic numeric check)
```

**Domain Keyword Updates**:
Add new domain keywords as test suites evolve:
```python
# Consider adding:
# - "deployment", "statefulset", "daemonset" (K8s resources)
# - "route", "ingress" (networking)
# - "crd", "operator" (operators)
```

## Polarion Test ID Extraction

### Overview

Polarion test IDs link test cases to their tracking items in Polarion WorkItem system. The extractor supports **two different patterns** used across OpenShift test repositories.

**Total Extraction**: 2,281 polarion links across all repositories

### Pattern 1: Decorator Pattern (885 test IDs)

**Used in**: cnf-gotests, eco-gotests

**Format**: Test ID passed as a decorator argument to `It()` between the description and the function literal.

```go
It("test description", polarion.ID("37056"), func() {
    // test body
})

// Also supports reportxml.ID():
It("deploy operator", reportxml.ID("48452"), func() {
    // test body
})
```

**Implementation**:
- **File**: `go-extractor/spec-extractor/internal/extractor/testid.go`
- **Function**: `extractTestID()` (lines 14-62)
- **Logic**: Checks `It()` call arguments BEFORE searching function body
  1. Iterate through `call.Args`
  2. Skip string literals (description) and function literals (test body)
  3. Check remaining args for `polarion.ID()` or `reportxml.ID()` call expressions
  4. Extract ID from first argument of ID call

**Output**:
```json
{"desc": "from the same policy", "test_id": "37056"}
```

```markdown
- **Test**: from the same policy
  - test_id: [OCP-37056](https://polarion.engineering.redhat.com/polarion/#/project/OSE/workitem?id=OCP-37056)
```

### Pattern 2: Author Pattern (1,396 test IDs)

**Used in**: openshift-tests, openshift-tests-private

**Format**: Test ID embedded in the test description string using `Author:username-Priority-POLARIONID-description` format.

```go
It("Author:bandrade-High-24061-have imagePullPolicy:IfNotPresent on thier deployments", func() {
    // test body
})

// Also handles prefixes:
It("ConnectedOnly-Author:jiazha-Critical-23440-can subscribe to the etcd operator", func() {
    // test body
})
```

**Implementation**:
- **File**: `go-extractor/spec-extractor/internal/extractor/constants.go`
- **Function**: `ParseTestDescription()` (lines 56-96)
- **Regex Pattern**: `Author:[^-]+-[^-]+-(\d+)-`
  - Matches: `Author:` + username + `-` + priority + `-` + **digits** + `-`
  - Captures the numeric ID (3rd field)

**Priority**: Explicit `[test_id:12345]` pattern takes precedence over Author pattern if both are present.

**Output**:
```json
{"desc": "Author:bandrade-High-24061-have imagePullPolicy:IfNotPresent...", "test_id": "24061"}
```

```markdown
- **Test**: Author:bandrade-High-24061-have imagePullPolicy:IfNotPresent on thier deployments
  - test_id: [OCP-24061](https://polarion.engineering.redhat.com/polarion/#/project/OSE/workitem?id=OCP-24061)
```

### Polarion Link Format

All numeric test IDs are prefixed with `OCP-` and linked to Polarion:

```markdown
[OCP-{ID}](https://polarion.engineering.redhat.com/polarion/#/project/OSE/workitem?id=OCP-{ID})
```

**Implementation**:
- **File**: `go-extractor/spec-extractor/internal/extractor/markdown.go`
- **Lines**: 177-191
- **Logic**:
  - Numeric IDs (e.g., "37056") → Add `OCP-` prefix → `OCP-37056`
  - Alphanumeric IDs (e.g., "C00113") → Use as-is

### Troubleshooting: If Test IDs Disappear

**Symptoms**: Markdown files missing `test_id:` lines or JSONL missing `test_id` field

**Check these files**:

1. **Decorator Pattern Extraction** (`go-extractor/spec-extractor/internal/extractor/testid.go:14-51`)
   - Verify `extractTestID()` checks `It()` call arguments FIRST
   - Should iterate through `call.Args` looking for `polarion.ID()` or `reportxml.ID()`
   - Must happen BEFORE searching function body

2. **Author Pattern Extraction** (`go-extractor/spec-extractor/internal/extractor/constants.go:56-96`)
   - Verify `ParseTestDescription()` includes Author pattern regex
   - Pattern should be: `Author:[^-]+-[^-]+-(\d+)-`
   - Should run AFTER checking for explicit `[test_id:]` pattern

3. **JSONL Export** (`go-extractor/spec-extractor/internal/extractor/jsonl.go:46-49`)
   - Verify `WritePerItJSONL()` includes `test_id` field
   - Check: `if tc.TestID != "" { rec.TestID = tc.TestID }`

4. **Markdown Rendering** (`go-extractor/spec-extractor/internal/extractor/markdown.go:176-191`)
   - Verify polarion link generation for `tc.TestID`
   - Check URL format and OCP- prefix logic

**Verification Commands**:

```bash
# Count extracted test IDs
jq -r 'select(.test_id) | .test_id' spec-md/go_specs_per_it.jsonl | wc -l
# Should be: ~2,281

# Check decorator pattern (cnf-gotests)
jq -r 'select(.file_path | contains("cnf-gotests")) | select(.test_id) | .test_id' spec-md/go_specs_per_it.jsonl | head -5

# Check Author pattern (openshift-tests)
jq -r 'select(.desc | contains("Author:")) | select(.test_id) | {desc, test_id}' spec-md/go_specs_per_it.jsonl | head -5

# Count polarion links in markdown
grep -r "polarion.engineering.redhat.com" spec-md/markdown/ | wc -l
# Should be: ~2,281
```

### Key Implementation Points

1. **Order Matters**: Decorator pattern must check `It()` arguments BEFORE function body
2. **Priority Matters**: Explicit `[test_id:]` takes precedence over Author pattern
3. **Two Layers**:
   - Extraction (Go): `extractTestID()` and `ParseTestDescription()`
   - Output (Go): JSONL export and markdown rendering
4. **Preserve Full Description**: Author pattern keeps full description including `Author:` prefix for context

### Historical Context

**2026-01-29**: Initial decorator pattern only checked inside function body, missing `polarion.ID("123")` arguments to `It()`. Fixed by checking `call.Args` first. Author pattern added to support openshift-tests format, increasing coverage from 885 to 2,281 test IDs.
