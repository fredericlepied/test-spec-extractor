package extractor

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

type PerItRecord struct {
	Desc           string   `json:"desc"`
	TestID         string   `json:"test_id,omitempty"`
	LineNumber     int      `json:"line_number,omitempty"`
	Labels         []string `json:"labels,omitempty"`
	PrepSteps      []string `json:"prep_steps,omitempty"`
	SkipConditions []string `json:"skip_conditions,omitempty"`
	Steps          []string `json:"steps,omitempty"`
	Validations    []string `json:"validations,omitempty"`
	CleanupSteps   []string `json:"cleanup_steps,omitempty"`
	FilePath       string   `json:"file_path,omitempty"`
	K8sResources   []string `json:"k8s_resources,omitempty"`
	SourceURL      string   `json:"source_url,omitempty"`
	Repo           string   `json:"repo,omitempty"`
}

// extractRepoName extracts the repository name from an absolute file path.
// It looks for a path segment after "external/" (e.g. /home/user/external/eco-gotests/... → "eco-gotests").
// Falls back to the git repository root directory name.
func extractRepoName(filePath string) string {
	parts := strings.Split(filepath.ToSlash(filePath), "/")
	for i, p := range parts {
		if p == "external" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	if root := findGitRepoRoot(filePath); root != "" {
		return filepath.Base(root)
	}
	return ""
}

// WritePerItJSONL appends one JSON object per test case found in the FileSpec to the given writer.
func WritePerItJSONL(spec *FileSpec, filePath string) error {
	if spec == nil || !spec.HasTests() || filePath == "" {
		return nil
	}
	// Open in append mode
	f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	// Walk and emit
	var emitContainer func(c *Container)
	emitContainer = func(c *Container) {
		if c == nil {
			return
		}

		for _, tc := range c.Cases {
			// Emit original test case if it has a description
			if tc.Description != "" {
				rec := PerItRecord{
					Desc:         tc.Description,
					FilePath:     spec.FilePath,
					LineNumber:   tc.LineNumber,
					K8sResources: spec.K8sResources,
					SourceURL:    makeGitHubURL(spec.FilePath, tc.LineNumber),
					Repo:         extractRepoName(spec.FilePath),
				}
				// Add test ID if present
				if tc.TestID != "" {
					rec.TestID = tc.TestID
				}
				// Inherit container labels
				if len(c.Labels) > 0 {
					rec.Labels = append(rec.Labels, c.Labels...)
				}
				// Add test-specific labels
				if len(tc.Labels) > 0 {
					rec.Labels = append(rec.Labels, tc.Labels...)
				}
				// prepend inherited preparation steps from this container
				if len(c.PrepSteps) > 0 {
					for _, s := range c.PrepSteps {
						rec.PrepSteps = append(rec.PrepSteps, s.Text)
					}
				}
				// append test case specific preparation steps
				for _, st := range tc.PrepSteps {
					rec.PrepSteps = append(rec.PrepSteps, st.Text)
				}
				// append inherited skip conditions from this container
				if len(c.SkipConditions) > 0 {
					for _, s := range c.SkipConditions {
						rec.SkipConditions = append(rec.SkipConditions, s.Text)
					}
				}
				// append test case specific skip conditions
				for _, skip := range tc.SkipConditions {
					rec.SkipConditions = append(rec.SkipConditions, skip.Text)
				}
				// append inherited cleanup steps from this container
				if len(c.CleanupSteps) > 0 {
					for _, s := range c.CleanupSteps {
						rec.CleanupSteps = append(rec.CleanupSteps, s.Text)
					}
				}
				for _, st := range tc.Steps {
					rec.Steps = append(rec.Steps, st.Text)
				}
				// append validations with transformation applied
				for _, v := range tc.Validations {
					transformed := transformSkipMessage(v.Text)
					rec.Validations = append(rec.Validations, transformed)
				}
				// append test case specific cleanup steps
				for _, st := range tc.CleanupSteps {
					rec.CleanupSteps = append(rec.CleanupSteps, st.Text)
				}
				if b, err := json.Marshal(rec); err == nil {
					w.Write(b)
					w.WriteByte('\n')
				}
			}
		}
		for _, child := range c.Children {
			emitContainer(child)
		}
	}
	for _, c := range spec.Root.Children {
		emitContainer(c)
	}
	return nil
}
