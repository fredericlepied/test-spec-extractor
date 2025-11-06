package extractor

import (
	"bufio"
	"encoding/json"
	"os"
)

type PerItRecord struct {
	Desc         string   `json:"desc"`
	Labels       []string `json:"labels,omitempty"`
	PrepSteps    []string `json:"prep_steps,omitempty"`
	Steps        []string `json:"steps,omitempty"`
	CleanupSteps []string `json:"cleanup_steps,omitempty"`
	FilePath     string   `json:"file_path,omitempty"`
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
			rec := PerItRecord{Desc: tc.Description, FilePath: spec.FilePath}
			if len(tc.Labels) > 0 {
				rec.Labels = append(rec.Labels, tc.Labels...)
			}
			// prepend inherited preparation steps from this container
			if len(c.PrepSteps) > 0 {
				for _, s := range c.PrepSteps {
					rec.PrepSteps = append(rec.PrepSteps, s.Text)
				}
			}
			// append test case specific preparation steps (including Skip conditions)
			for _, st := range tc.PrepSteps {
				rec.PrepSteps = append(rec.PrepSteps, st.Text)
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
			// append test case specific cleanup steps
			for _, st := range tc.CleanupSteps {
				rec.CleanupSteps = append(rec.CleanupSteps, st.Text)
			}
			if b, err := json.Marshal(rec); err == nil {
				w.Write(b)
				w.WriteByte('\n')
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
