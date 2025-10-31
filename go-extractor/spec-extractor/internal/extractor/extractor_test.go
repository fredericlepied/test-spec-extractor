package extractor

import (
	"go/parser"
	"go/token"
	"testing"
)

func TestBuildFileSpec_Basic(t *testing.T) {
	src := `package sample
import g "github.com/onsi/ginkgo/v2"

var _ = g.Describe("top", func() {
    g.BeforeEach(func(){ g.By("prep step") })
    g.When("cond A", func(){
        g.It("does X", func(){
            g.By("step one")
            g.By("step two")
        }, g.Label("slow","e2e"))
    })
})`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "sample.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fr := &FileResult{FilePath: "sample.go", Package: f.Name.Name, ImportMap: map[string]string{"g": "github.com/onsi/ginkgo/v2"}, AST: f, FileSet: fset}
	spec := BuildFileSpec(fr, nil)

	if len(spec.Root.Children) != 1 {
		t.Fatalf("expected 1 top container, got %d", len(spec.Root.Children))
	}
	top := spec.Root.Children[0]
	if top.Description != "top" {
		t.Fatalf("unexpected top desc: %q", top.Description)
	}
	if len(top.PrepSteps) != 1 || top.PrepSteps[0].Text != "prep step" {
		t.Fatalf("unexpected prep steps: %+v", top.PrepSteps)
	}
	if len(top.Children) != 1 || top.Children[0].Kind != "When" {
		t.Fatalf("expected When child, got: %+v", top.Children)
	}
	when := top.Children[0]
	if len(when.Cases) != 1 {
		t.Fatalf("expected 1 case, got %d", len(when.Cases))
	}
	it := when.Cases[0]
	if it.Description != "does X" {
		t.Fatalf("unexpected it desc: %q", it.Description)
	}
	if len(it.Labels) != 2 {
		t.Fatalf("expected 2 labels, got %v", it.Labels)
	}
	if len(it.Steps) != 2 || it.Steps[0].Text != "step one" {
		t.Fatalf("unexpected steps: %+v", it.Steps)
	}
}
