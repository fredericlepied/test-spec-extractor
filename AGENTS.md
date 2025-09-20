# AI Coding Assistent Rules for Test Spec Extractor

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
