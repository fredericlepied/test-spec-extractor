package extractor

// Model structures for extracted specs

type TestStep struct {
	Text string
}

type TestCase struct {
	Description    string
	TestID         string // Test ID from reportxml.ID("12345")
	LineNumber     int    // Line number where the test starts in source file
	Labels         []string
	PrepSteps      []TestStep // Individual test prerequisites
	SkipConditions []TestStep // Skip conditions (rendered as "Skip if" section)
	Steps          []TestStep
	Validations    []TestStep // Expect assertions with failure messages
	CleanupSteps   []TestStep // Cleanup actions and Fail messages
}

type Container struct {
	Kind           string // Describe, Context, When
	Description    string
	Labels         []string
	Children       []*Container
	PrepSteps      []TestStep // By(...) found in active Before* under this container
	SkipConditions []TestStep // Skip(...) found in Before* blocks (rendered as "Skip if" section)
	CleanupSteps   []TestStep // By(...) found in active After* under this container
	Cases          []TestCase
}

type FileSpec struct {
	FilePath string
	Root     *Container // synthetic root
}

// HasTests returns true if any Container in the tree contains at least one TestCase.
func (f *FileSpec) HasTests() bool {
	if f == nil || f.Root == nil {
		return false
	}
	return containerHasTests(f.Root)
}

func containerHasTests(c *Container) bool {
	if c == nil {
		return false
	}
	if len(c.Cases) > 0 {
		return true
	}
	for _, child := range c.Children {
		if containerHasTests(child) {
			return true
		}
	}
	return false
}
