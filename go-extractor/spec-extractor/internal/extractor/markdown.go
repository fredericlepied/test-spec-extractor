package extractor

import (
	"bytes"
	"fmt"
	"strings"
)

// transformSkipMessage transforms negative skip/expect messages to positive test descriptions
// by removing "No " / "Not " prefixes and replacing negative contractions with positive forms
func transformSkipMessage(msg string) string {
	// Remove "No " prefix (case-insensitive)
	if len(msg) > 3 && strings.ToLower(msg[:3]) == "no " {
		msg = msg[3:]
	}

	// Remove "Not all " prefix (case-insensitive) -> "all "
	if len(msg) > 8 && strings.ToLower(msg[:8]) == "not all " {
		msg = msg[4:] // Keep "all " and remove "Not "
	} else if len(msg) > 4 && strings.ToLower(msg[:4]) == "not " {
		// Remove "Not " prefix for other cases (case-insensitive)
		msg = msg[4:]
	}

	// Remove "Unable to " prefix and replace with "able to "
	if len(msg) > 10 && strings.ToLower(msg[:10]) == "unable to " {
		// Preserve original case
		if msg[0] >= 'A' && msg[0] <= 'Z' {
			msg = "Able to " + msg[10:]
		} else {
			msg = "able to " + msg[10:]
		}
	}

	// Remove "Failed to " prefix and replace with "successfully "
	if len(msg) > 10 && strings.ToLower(msg[:10]) == "failed to " {
		// Preserve original case
		if msg[0] >= 'A' && msg[0] <= 'Z' {
			msg = "Successfully " + msg[10:]
		} else {
			msg = "successfully " + msg[10:]
		}
	}

	// Remove "Cannot " / "Can not " prefix and replace with "can "
	if len(msg) > 7 && strings.ToLower(msg[:7]) == "cannot " {
		if msg[0] >= 'A' && msg[0] <= 'Z' {
			msg = "Can " + msg[7:]
		} else {
			msg = "can " + msg[7:]
		}
	} else if len(msg) > 8 && strings.ToLower(msg[:8]) == "can not " {
		if msg[0] >= 'A' && msg[0] <= 'Z' {
			msg = "Can " + msg[8:]
		} else {
			msg = "can " + msg[8:]
		}
	}

	// Remove "Could not " prefix and replace with "could "
	if len(msg) > 11 && strings.ToLower(msg[:11]) == "could not " {
		if msg[0] >= 'A' && msg[0] <= 'Z' {
			msg = "Could " + msg[11:]
		} else {
			msg = "could " + msg[11:]
		}
	}

	// Replace negative phrases with positive forms
	// Handle "is not" / "are not" first (before contractions)
	negativePatterns := map[string]string{
		" should not ": " should ",
		" is not ":     " is ",
		" are not ":    " are ",
		" was not ":    " was ",
		" were not ":   " were ",
		" has not ":    " has ",
		" have not ":   " have ",
		" does not ":   " does ",
		" do not ":     " do ",
		" can not ":    " can ",
		" could not ":  " could ",
		" will not ":   " will ",
		" would not ":  " would ",
		"shouldn't":    "should",
		"doesn't":      "does",
		"don't":        "do",
		"isn't":        "is",
		"aren't":       "are",
		"can't":        "can",
		"won't":        "will",
		"hasn't":       "has",
		"haven't":      "have",
		"wasn't":       "was",
		"weren't":      "were",
	}

	// Process each pattern
	for negative, positive := range negativePatterns {
		// Case-insensitive replacement while preserving original case
		lowerMsg := strings.ToLower(msg)
		negLower := strings.ToLower(negative)

		if idx := strings.Index(lowerMsg, negLower); idx != -1 {
			// Preserve the case of the first letter
			replacement := positive
			if idx < len(msg) && msg[idx] >= 'A' && msg[idx] <= 'Z' {
				// Original was capitalized
				replacement = strings.ToUpper(positive[:1]) + positive[1:]
			}
			msg = msg[:idx] + replacement + msg[idx+len(negative):]
			// Only replace the first occurrence to avoid over-processing
			break
		}
	}

	return msg
}

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
	if len(c.CleanupSteps) > 0 {
		fmt.Fprintf(b, "- **cleanup**:\n")
		for _, s := range c.CleanupSteps {
			fmt.Fprintf(b, "  - %s\n", safe(s.Text))
		}
	}

	// Show container-level Skip conditions (from Before* blocks)
	if len(c.SkipConditions) > 0 {
		fmt.Fprintf(b, "- **Skip if**:\n")
		for _, skip := range c.SkipConditions {
			fmt.Fprintf(b, "  - %s\n", safe(skip.Text))
		}
	}

	if len(c.Cases) > 0 {
		for _, tc := range c.Cases {
			// Render original test case if it has a description
			if tc.Description != "" {
				fmt.Fprintf(b, "- **Test**: %s\n", safe(tc.Description))
				if len(tc.Labels) > 0 {
					fmt.Fprintf(b, "  - labels: %s\n", strings.Join(tc.Labels, ", "))
				}
				if len(tc.PrepSteps) > 0 {
					fmt.Fprintf(b, "  - preparation:\n")
					for _, s := range tc.PrepSteps {
						fmt.Fprintf(b, "    - %s\n", safe(s.Text))
					}
				}
				if len(tc.SkipConditions) > 0 {
					fmt.Fprintf(b, "  - Skip if:\n")
					for _, skip := range tc.SkipConditions {
						fmt.Fprintf(b, "    - %s\n", safe(skip.Text))
					}
				}
				if len(tc.Steps) > 0 {
					fmt.Fprintf(b, "  - steps:\n")
					for _, s := range tc.Steps {
						fmt.Fprintf(b, "    - %s\n", safe(s.Text))
					}
				}
				if len(tc.Validations) > 0 {
					fmt.Fprintf(b, "  - validations:\n")
					for _, v := range tc.Validations {
						// Apply transformation to flip negative to positive
						transformed := transformSkipMessage(v.Text)
						fmt.Fprintf(b, "    - %s\n", safe(transformed))
					}
				}
				if len(tc.CleanupSteps) > 0 {
					fmt.Fprintf(b, "  - cleanup:\n")
					for _, s := range tc.CleanupSteps {
						fmt.Fprintf(b, "    - %s\n", safe(s.Text))
					}
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
	// naive escaping for markdown bullets and underscores
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "_", "\\_")
	return s
}
