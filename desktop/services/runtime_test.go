package services

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The frontend can call ReadTrace with any string, so what it will read is
// the thing to hold down: a trace.json this service wrote, under its own
// scratch folder, and nothing else on the disk.
func TestReadTraceReadsOnlyItsOwnTraces(t *testing.T) {
	s := &RuntimeService{}

	dir := filepath.Join(traceRoot(), "readtrace-test")
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

	got, err := s.ReadTrace(filepath.Join(dir, "trace.json"))
	if err != nil || got != want {
		t.Fatalf("its own trace: got %q, %v", got, err)
	}

	outside := filepath.Join(t.TempDir(), "trace.json")
	if err := os.WriteFile(outside, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, path := range map[string]string{
		"another file beside it":         filepath.Join(dir, "model.py"),
		"a trace outside its folder":     outside,
		"a climb out of the folder":      filepath.Join(traceRoot(), "..", "trace.json"),
		"the folder itself":              traceRoot(),
		"a trace that was never written": filepath.Join(traceRoot(), "nobody", "trace.json"),
	} {
		if _, err := s.ReadTrace(path); err == nil {
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
