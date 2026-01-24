package extractor

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// ExpectInfo holds parsed information from an Expect assertion
type ExpectInfo struct {
	Subject     string // What's being tested (e.g., "err", "ready", "len(workerNodeList)")
	Matcher     string // The matcher name (e.g., "HaveOccurred", "Equal", "BeNumerically")
	Expected    string // Expected value if any
	IsNegated   bool   // true for ToNot/NotTo, false for To
	Message     string // Explicit failure message if provided
	Description string // Generated human-readable description
}

// parseExpectAssertion extracts information from an Expect assertion chain
func parseExpectAssertion(call *ast.CallExpr, recog *Recognizer) *ExpectInfo {
	// Check if this is a .To()/.ToNot()/.NotTo() call on an Expect
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil
	}

	methodName := sel.Sel.Name
	var isNegated bool
	switch methodName {
	case "To":
		isNegated = false
	case "ToNot", "NotTo":
		isNegated = true
	default:
		return nil
	}

	// Get the Expect call (the receiver)
	expectCall, ok := sel.X.(*ast.CallExpr)
	if !ok || !recog.IsExpect(expectCall) {
		return nil
	}

	info := &ExpectInfo{
		IsNegated: isNegated,
	}

	// Extract the subject (what's being tested) - first argument to Expect()
	if len(expectCall.Args) > 0 {
		info.Subject = exprToString(expectCall.Args[0])
	}

	// Extract the matcher and expected value - first argument to .To()/.ToNot()
	if len(call.Args) > 0 {
		if matcherCall, ok := call.Args[0].(*ast.CallExpr); ok {
			info.Matcher = exprToString(matcherCall.Fun)
			// Extract matcher arguments (expected values)
			if len(matcherCall.Args) > 0 {
				// For matchers like Equal(x), BeNumerically(">", 1), etc.
				parts := make([]string, len(matcherCall.Args))
				for i, arg := range matcherCall.Args {
					parts[i] = exprToString(arg)
				}
				info.Expected = strings.Join(parts, ", ")
			}
		}
	}

	// Extract explicit failure message (last string argument to .To()/.ToNot())
	info.Message = lastStringArg(call)

	// Generate human-readable description
	info.Description = generateDescription(info)

	return info
}

// generateDescription creates a human-readable validation description
func generateDescription(info *ExpectInfo) string {
	// If there's an explicit message, apply the same transformation as Skip messages
	// This converts negative phrasing to positive (e.g., "Failed to" → "Successfully")
	// Both To and NotTo use the same transformation
	if info.Message != "" {
		return transformSkipMessage(info.Message)
	}

	// Generate description based on matcher type
	subject := cleanSubject(info.Subject)
	matcher := cleanMatcher(info.Matcher)

	switch matcher {
	case "HaveOccurred":
		if info.IsNegated {
			return fmt.Sprintf("%s succeeds without error", subject)
		}
		return fmt.Sprintf("%s returns an error", subject)

	case "Succeed":
		if info.IsNegated {
			return fmt.Sprintf("%s fails", subject)
		}
		return fmt.Sprintf("%s succeeds", subject)

	case "Equal":
		value := info.Expected
		if info.IsNegated {
			// Use positive phrasing for negated equality
			return fmt.Sprintf("%s differs from %s", subject, value)
		}
		// Special handling for boolean values
		if value == "true" {
			return fmt.Sprintf("%s is true", subject)
		} else if value == "false" {
			return fmt.Sprintf("%s is false", subject)
		}
		return fmt.Sprintf("%s equals %s", subject, value)

	case "BeTrue":
		if info.IsNegated {
			return fmt.Sprintf("%s is false", subject)
		}
		return fmt.Sprintf("%s is true", subject)

	case "BeFalse":
		if info.IsNegated {
			return fmt.Sprintf("%s is true", subject)
		}
		return fmt.Sprintf("%s is false", subject)

	case "BeNumerically":
		// Expected format: "operator, value" like ">, 1"
		parts := strings.Split(info.Expected, ", ")
		if len(parts) >= 2 {
			op := strings.Trim(parts[0], "\"")
			val := parts[1]
			opText := operatorToText(op, info.IsNegated)
			return fmt.Sprintf("%s %s %s", subject, opText, val)
		}
		return fmt.Sprintf("%s matches numeric expectation", subject)

	case "BeEmpty":
		if info.IsNegated {
			return fmt.Sprintf("%s contains data", subject)
		}
		return fmt.Sprintf("%s is empty", subject)

	case "HaveLen":
		if info.IsNegated {
			return fmt.Sprintf("%s has different length than %s", subject, info.Expected)
		}
		return fmt.Sprintf("%s has length %s", subject, info.Expected)

	case "ContainElement", "ContainSubstring", "Contain":
		if info.IsNegated {
			return fmt.Sprintf("%s excludes %s", subject, info.Expected)
		}
		return fmt.Sprintf("%s contains %s", subject, info.Expected)

	case "MatchRegexp":
		if info.IsNegated {
			return fmt.Sprintf("%s differs from pattern %s", subject, info.Expected)
		}
		return fmt.Sprintf("%s matches pattern %s", subject, info.Expected)

	case "BeNil":
		if info.IsNegated {
			return fmt.Sprintf("%s exists", subject)
		}
		return fmt.Sprintf("%s is nil", subject)

	case "BeZero":
		if info.IsNegated {
			return fmt.Sprintf("%s has a value", subject)
		}
		return fmt.Sprintf("%s is zero", subject)

	default:
		// Generic fallback - use positive phrasing even for negated matchers
		if info.IsNegated {
			return fmt.Sprintf("%s differs from %s expectation", subject, matcher)
		}
		return fmt.Sprintf("%s satisfies %s", subject, matcher)
	}
}

// cleanSubject makes the subject more readable
func cleanSubject(subject string) string {
	// Common patterns to clean up
	subject = strings.TrimSpace(subject)

	// If it's "err", make it more descriptive
	if subject == "err" {
		return "operation"
	}

	// Remove package prefixes for readability
	if idx := strings.LastIndex(subject, "."); idx != -1 && idx < len(subject)-1 {
		// Keep only the last part if it looks like a package.Function call
		parts := strings.Split(subject, ".")
		if len(parts) > 2 {
			subject = strings.Join(parts[len(parts)-2:], ".")
		}
	}

	return subject
}

// cleanMatcher extracts the matcher name without package prefix
func cleanMatcher(matcher string) string {
	matcher = strings.TrimSpace(matcher)
	// Remove package prefix if present
	if idx := strings.LastIndex(matcher, "."); idx != -1 && idx < len(matcher)-1 {
		return matcher[idx+1:]
	}
	return matcher
}

// operatorToText converts comparison operators to text (using positive phrasing)
func operatorToText(op string, negated bool) string {
	switch op {
	case ">":
		if negated {
			return "is less than or equal to"
		}
		return "is greater than"
	case ">=":
		if negated {
			return "is less than"
		}
		return "is greater than or equal to"
	case "<":
		if negated {
			return "is greater than or equal to"
		}
		return "is less than"
	case "<=":
		if negated {
			return "is greater than"
		}
		return "is less than or equal to"
	case "==", "=":
		if negated {
			return "differs from"
		}
		return "equals"
	case "!=":
		if negated {
			return "equals"
		}
		return "differs from"
	default:
		return "compares to"
	}
}

// exprToString converts an AST expression to a string representation
func exprToString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		return exprToString(e.X) + "." + e.Sel.Name
	case *ast.CallExpr:
		name := exprToString(e.Fun)
		// For function calls, show the function name and arg count
		if len(e.Args) > 0 {
			args := make([]string, len(e.Args))
			for i, arg := range e.Args {
				args[i] = exprToString(arg)
			}
			return name + "(" + strings.Join(args, ", ") + ")"
		}
		return name + "()"
	case *ast.BasicLit:
		if e.Kind == token.STRING {
			return unquote(e.Value)
		}
		return e.Value
	case *ast.UnaryExpr:
		return e.Op.String() + exprToString(e.X)
	case *ast.BinaryExpr:
		return exprToString(e.X) + " " + e.Op.String() + " " + exprToString(e.Y)
	case *ast.IndexExpr:
		return exprToString(e.X) + "[" + exprToString(e.Index) + "]"
	case *ast.ParenExpr:
		return "(" + exprToString(e.X) + ")"
	default:
		return "<expr>"
	}
}

func lastStringArg(call *ast.CallExpr) string {
	// Find the last string argument (used for Expect failure messages)
	var lastStr string
	for _, a := range call.Args {
		if bl, ok := a.(*ast.BasicLit); ok && bl.Kind == token.STRING {
			lastStr = unquote(bl.Value)
		}
	}
	return lastStr
}

func unquote(s string) string {
	if s == "" {
		return s
	}
	if uq, err := strconv.Unquote(s); err == nil {
		return uq
	}
	return s
}
