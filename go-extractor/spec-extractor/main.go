package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"kubespec-extractor/spec-extractor/internal/extractor"
)

type aliasMap map[string][]string

type config struct {
	root    string
	outDir  string
	include []string
	exclude []string
	aliases aliasMap
	jsonl   string
}

func parseFlags() config {
	var root string
	var out string
	var includeCSV string
	var excludeCSV string
	var aliasCSV multiValue
	var jsonlOut string

	flag.StringVar(&root, "root", ".", "Root directory to scan for Go tests")
	flag.StringVar(&out, "out", "spec-out", "Output directory for generated markdown")
	flag.StringVar(&includeCSV, "include", "**/*.go", "Comma-separated include globs")
	flag.StringVar(&excludeCSV, "exclude", "vendor/**,**/*_testdata/**", "Comma-separated exclude globs")
	flag.Var(&aliasCSV, "alias", "Repeatable alias mappings: Name=Alt1,Alt2 (can be specified multiple times)")
	flag.StringVar(&jsonlOut, "jsonl", "", "Optional path to write per-test JSONL records")
	flag.Parse()

	cfg := config{
		root:    root,
		outDir:  out,
		include: splitCSV(includeCSV),
		exclude: splitCSV(excludeCSV),
		aliases: make(aliasMap),
		jsonl:   jsonlOut,
	}

	for _, entry := range aliasCSV {
		name, alts, ok := parseAlias(entry)
		if !ok {
			fmt.Fprintf(os.Stderr, "invalid --alias: %s\n", entry)
			continue
		}
		cfg.aliases[name] = append(cfg.aliases[name], alts...)
	}

	return cfg
}

type multiValue []string

func (m *multiValue) String() string { return strings.Join(*m, ",") }

func (m *multiValue) Set(value string) error {
	*m = append(*m, value)
	return nil
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseAlias(s string) (string, []string, bool) {
	eq := strings.IndexByte(s, '=')
	if eq <= 0 || eq >= len(s)-1 {
		return "", nil, false
	}
	name := s[:eq]
	alts := splitCSV(s[eq+1:])
	if name == "" || len(alts) == 0 {
		return "", nil, false
	}
	return name, alts, true
}

func main() {
	cfg := parseFlags()

	// Create output directory if missing
	if err := os.MkdirAll(cfg.outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "failed creating output dir: %v\n", err)
		os.Exit(1)
	}

	var files []string
	err := filepath.Walk(cfg.root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			base := filepath.Base(path)
			if base == "vendor" || strings.HasSuffix(base, "_testdata") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "walk error: %v\n", err)
		os.Exit(1)
	}

	// Parse files to build import alias maps (and verify parsable)
	parsed := 0
	for _, f := range files {
		res, err := extractor.ParseFile(f)
		if err != nil {
			// Best-effort: skip unparsable files with a warning
			fmt.Fprintf(os.Stderr, "warn: parse error in %s: %v\n", f, err)
			continue
		}
		// Build spec, and only write if the file contains at least one test case
		spec := extractor.BuildFileSpec(res, cfg.aliases)
		if spec.HasTests() {
			rel, err := filepath.Rel(cfg.root, f)
			if err != nil {
				rel = filepath.Base(f)
			}
			outPath := filepath.Join(cfg.outDir, strings.TrimSuffix(rel, filepath.Ext(rel))+".md")
			if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
				fmt.Fprintf(os.Stderr, "warn: cannot create dir for %s: %v\n", outPath, err)
				continue
			}
			if err := os.WriteFile(outPath, extractor.RenderMarkdown(spec), 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "warn: write failed for %s: %v\n", outPath, err)
				continue
			}
			if cfg.jsonl != "" {
				if err := extractor.WritePerItJSONL(spec, cfg.jsonl); err != nil {
					fmt.Fprintf(os.Stderr, "warn: jsonl write failed for %s: %v\n", f, err)
				}
			}
		}
		parsed++
	}

	fmt.Printf("Parsed %d/%d Go files under %s. Output will be written to %s\n", parsed, len(files), cfg.root, cfg.outDir)
}
