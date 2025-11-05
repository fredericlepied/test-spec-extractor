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
		renderContainerWithConditions(&b, c, 0, []string{})
	}
	return b.Bytes()
}

func renderContainerWithConditions(b *bytes.Buffer, c *Container, depth int, whenConditions []string) {
	// Heading level by depth: 0=>###, 1=>####, 2=>#####
	level := 3 + depth
	if level > 6 {
		level = 6
	}
	heading := strings.Repeat("#", level)
	fmt.Fprintf(b, "%s %s: %s\n\n", heading, c.Kind, safe(c.Description))

	// Show inherited When conditions as prerequisites
	if len(whenConditions) > 0 {
		fmt.Fprintf(b, "- **when**: %s\n", strings.Join(whenConditions, ", "))
	}

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
			fmt.Fprintf(b, "- **Test**: %s\n", safe(tc.Description))
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

	// Pass down When conditions to children, adding current one if this is a When block
	childWhenConditions := make([]string, len(whenConditions))
	copy(childWhenConditions, whenConditions)
	if c.Kind == "When" && c.Description != "" {
		childWhenConditions = append(childWhenConditions, safe(c.Description))
	}

	for _, child := range c.Children {
		renderContainerWithConditions(b, child, depth+1, childWhenConditions)
	}
}

func safe(s string) string {
	// naive escaping for markdown bullets
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
