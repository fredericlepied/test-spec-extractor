package extractor

import (
	"bytes"
	"fmt"
	"strings"
)

func RenderMarkdown(spec *FileSpec) []byte {
	var b bytes.Buffer
	fmt.Fprintf(&b, "## %s\n\n", spec.FilePath)
	// Walk containers from root
	for _, c := range spec.Root.Children {
		renderContainer(&b, c, 0)
	}
	return b.Bytes()
}

func renderContainer(b *bytes.Buffer, c *Container, depth int) {
	// Heading level by depth: 0=>###, 1=>####, 2=>#####
	level := 3 + depth
	if level > 6 {
		level = 6
	}
	heading := strings.Repeat("#", level)
	fmt.Fprintf(b, "%s %s: %s\n\n", heading, c.Kind, safe(c.Description))
	if len(c.Labels) > 0 {
		fmt.Fprintf(b, "- **labels**: %s\n", strings.Join(c.Labels, ", "))
	}
	if len(c.PrepSteps) > 0 {
		fmt.Fprintf(b, "- **preparation**:\n")
		for _, s := range c.PrepSteps {
			fmt.Fprintf(b, "  - %s\n", safe(s.Text))
		}
	}
	if len(c.Cases) > 0 {
		for _, tc := range c.Cases {
			fmt.Fprintf(b, "- **It**: %s\n", safe(tc.Description))
			if len(tc.Labels) > 0 {
				fmt.Fprintf(b, "  - labels: %s\n", strings.Join(tc.Labels, ", "))
			}
			if len(tc.Steps) > 0 {
				fmt.Fprintf(b, "  - steps:\n")
				for _, s := range tc.Steps {
					fmt.Fprintf(b, "    - %s\n", safe(s.Text))
				}
			}
		}
		fmt.Fprintln(b)
	}
	for _, child := range c.Children {
		renderContainer(b, child, depth+1)
	}
}

func safe(s string) string {
	// naive escaping for markdown bullets
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
