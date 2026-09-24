package services

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The frontend can call ReadResult with any string, so what it will read is
// the thing to hold down: a JSON result a job of this app left under its own
// scratch folder, and nothing else on the disk.
func TestReadResultReadsOnlyWhatItsJobsMade(t *testing.T) {
	s := &RuntimeService{}

	dir := filepath.Join(scratchRoot(), "readresult-test")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	want := `{"version":1}`
	if err := os.WriteFile(filepath.Join(dir, "trace.json"), []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.py"), []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := s.ReadResult(filepath.Join(dir, "trace.json"))
	if err != nil || got != want {
		t.Fatalf("its own trace: got %q, %v", got, err)
	}
	// A run record, in the folder smoke-train writes to, is a result too.
	if err := os.MkdirAll(filepath.Join(dir, "runs"), 0o755); err != nil {
		t.Fatal(err)
	}
	record := filepath.Join(dir, "runs", "run.json")
	if err := os.WriteFile(record, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := s.ReadResult(record); err != nil || got != want {
		t.Fatalf("its own run record: got %q, %v", got, err)
	}

	outside := filepath.Join(t.TempDir(), "trace.json")
	if err := os.WriteFile(outside, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, path := range map[string]string{
		"another file beside it":         filepath.Join(dir, "model.py"),
		"a trace outside its folder":     outside,
		"a climb out of the folder":      filepath.Join(scratchRoot(), "..", "trace.json"),
		"the folder itself":              scratchRoot(),
		"a trace that was never written": filepath.Join(scratchRoot(), "nobody", "trace.json"),
		"the step log beside the record": filepath.Join(dir, "runs", "run.jsonl"),
	} {
		if _, err := s.ReadResult(path); err == nil {
			t.Errorf("%s: read %s, which it should have refused", name, path)
		}
	}
}

// Generated paths are relative by construction, and one that climbs out of
// the scratch folder is refused before anything is written.
func TestWriteFilesStaysInsideItsFolder(t *testing.T) {
	target := t.TempDir()
	if _, err := writeFiles(target, []GeneratedFile{{Path: "../escape.py", Contents: "x"}}); err == nil {
		t.Fatal("wrote a file above its folder")
	}
	result, err := writeFiles(target, []GeneratedFile{
		{Path: "model.py", Contents: "a"},
		{Path: "design.tensorcad.json", Contents: "{}"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Written) != 2 || !strings.HasPrefix(result.Written[0], target) {
		t.Fatalf("wrote %v", result.Written)
	}
}

// A design's trace lives beside it under the same name, and only a design
// file has one: anything else is refused rather than guessed at.
func TestTraceIsKeptBesideItsDesign(t *testing.T) {
	s := &DesignService{}
	dir := t.TempDir()
	design := filepath.Join(dir, "gpt.tensorcad.json")

	if got, err := s.ReadTrace(design); err != nil || got != "" {
		t.Fatalf("no trace yet: got %q, %v", got, err)
	}
	written, err := s.SaveTrace(design, `{"version":1}`)
	if err != nil {
		t.Fatal(err)
	}
	if written != filepath.Join(dir, "gpt.trace.json") {
		t.Fatalf("wrote %s, not beside the design", written)
	}
	if got, err := s.ReadTrace(design); err != nil || got != `{"version":1}` {
		t.Fatalf("read back %q, %v", got, err)
	}
	if _, err := os.Stat(written + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("left its temporary file behind")
	}

	for _, notADesign := range []string{filepath.Join(dir, "notes.txt"), filepath.Join(dir, "gpt.trace.json")} {
		if _, err := s.ReadTrace(notADesign); err == nil {
			t.Errorf("read a trace for %s, which is not a design", notADesign)
		}
		if _, err := s.SaveTrace(notADesign, "{}"); err == nil {
			t.Errorf("wrote a trace for %s, which is not a design", notADesign)
		}
	}
	if _, err := s.SaveTrace(design, "  "); err == nil {
		t.Error("wrote an empty trace")
	}
}
