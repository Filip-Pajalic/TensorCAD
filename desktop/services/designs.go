// Package services holds the Go side of TensorCAD: everything that needs the
// operating system rather than the browser.
//
// The design document itself is JSON produced and consumed by the TypeScript
// core, so this package treats it as opaque bytes. It owns where a design
// lives, not what it means.
package services

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/wailsapp/wails/v3/pkg/application"
)

// Extension every design file carries. Checked on save so a design opened from
// the recent list is always the thing it claims to be.
const DesignExtension = ".tensorcad.json"

// DesignFile is a design on disk, with the metadata the UI shows about it.
type DesignFile struct {
	// Path is absolute. Empty means the design has never been saved.
	Path string `json:"path"`
	// Name is the file's base name without the extension.
	Name string `json:"name"`
	// Contents is the raw document JSON.
	Contents string `json:"contents"`
	// ModifiedAt is the file's modification time, RFC 3339.
	ModifiedAt string `json:"modifiedAt"`
	// Size in bytes.
	Size int64 `json:"size"`
}

// RecentEntry is one row of the recent-files list.
type RecentEntry struct {
	Path string `json:"path"`
	Name string `json:"name"`
	// OpenedAt is when this design was last opened, RFC 3339.
	OpenedAt string `json:"openedAt"`
	// Missing is true when the file is no longer where it was.
	Missing bool `json:"missing"`
}

// DesignService opens, saves and remembers design files.
type DesignService struct {
	app *application.App

	mu     sync.Mutex
	recent []RecentEntry
	// stateDir holds the recent list and any other small app state.
	stateDir string
}

// NewDesignService wires the service to the running application.
func NewDesignService(app *application.App, stateDir string) *DesignService {
	s := &DesignService{app: app, stateDir: stateDir}
	s.loadRecent()
	return s
}

// ServiceName is what Wails calls this service in logs.
func (s *DesignService) ServiceName() string { return "Designs" }

// Open asks for a file and returns it. An empty path means the user cancelled,
// which is not an error.
func (s *DesignService) Open() (*DesignFile, error) {
	dialog := s.app.Dialog.OpenFile()
	dialog.SetTitle("Open design")
	dialog.AddFilter("TensorCAD design", "*.tensorcad.json;*.json")
	dialog.CanChooseFiles(true)
	dialog.CanChooseDirectories(false)

	path, err := dialog.PromptForSingleSelection()
	if err != nil {
		return nil, fmt.Errorf("open dialog: %w", err)
	}
	if path == "" {
		return nil, nil
	}
	return s.OpenPath(path)
}

// OpenPath reads a design from a known path, which is how the recent list and
// the command line open one.
func (s *DesignService) OpenPath(path string) (*DesignFile, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve %q: %w", path, err)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			s.markMissing(abs)
			return nil, fmt.Errorf("%s no longer exists", abs)
		}
		return nil, fmt.Errorf("read %s: %w", abs, err)
	}
	// Parse only far enough to reject something that is not a design. The
	// TypeScript core does the real validation.
	var probe struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("%s is not valid JSON: %w", filepath.Base(abs), err)
	}
	if probe.Version == 0 {
		return nil, fmt.Errorf("%s has no document version, so it is probably not a design", filepath.Base(abs))
	}

	info, err := os.Stat(abs)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", abs, err)
	}

	s.remember(abs)
	return &DesignFile{
		Path:       abs,
		Name:       designName(abs),
		Contents:   string(data),
		ModifiedAt: info.ModTime().Format(time.RFC3339),
		Size:       info.Size(),
	}, nil
}

// Save writes a design. When path is empty it asks where to put it. It returns
// the path written, or empty when the user cancelled.
func (s *DesignService) Save(path string, contents string, suggestedName string) (string, error) {
	if strings.TrimSpace(contents) == "" {
		return "", errors.New("refusing to write an empty design")
	}

	if path == "" {
		dialog := s.app.Dialog.SaveFile()
		dialog.SetMessage("Save design")
		if suggestedName == "" {
			suggestedName = "design"
		}
		dialog.SetFilename(suggestedName + DesignExtension)
		dialog.AddFilter("TensorCAD design", "*.tensorcad.json")
		chosen, err := dialog.PromptForSingleSelection()
		if err != nil {
			return "", fmt.Errorf("save dialog: %w", err)
		}
		if chosen == "" {
			return "", nil
		}
		path = chosen
	}

	if !strings.HasSuffix(strings.ToLower(path), DesignExtension) {
		path = strings.TrimSuffix(path, filepath.Ext(path)) + DesignExtension
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("create folder: %w", err)
	}
	// Write to a temporary file and rename, so an interrupted save never
	// truncates a design that was already on disk.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(contents), 0o644); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("replace %s: %w", path, err)
	}

	s.remember(path)
	return path, nil
}

// TraceExtension is what a design's trace is saved as, beside the design:
// `gpt.tensorcad.json` keeps its values in `gpt.trace.json`.
const TraceExtension = ".trace.json"

// maxTraceBytes is the largest trace ReadTrace returns. The runtime refuses to
// trace anything that would come near it.
const maxTraceBytes = 64 << 20

// tracePath is where a design's trace is kept: the same folder and name, so the
// two are copied, sent and archived together.
func tracePath(designPath string) (string, error) {
	if !strings.HasSuffix(strings.ToLower(designPath), DesignExtension) {
		return "", fmt.Errorf("%s is not a design file", filepath.Base(designPath))
	}
	return designPath[:len(designPath)-len(DesignExtension)] + TraceExtension, nil
}

// ReadTrace returns the trace kept beside a design, or an empty string when
// there is none — which is the usual case, not an error.
func (s *DesignService) ReadTrace(designPath string) (string, error) {
	path, err := tracePath(designPath)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("stat %s: %w", path, err)
	}
	if info.Size() > maxTraceBytes {
		return "", fmt.Errorf("%s is %d bytes, more than the %d this reads", filepath.Base(path), info.Size(), maxTraceBytes)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	return string(data), nil
}

// SaveTrace keeps a trace beside the design it describes, replacing any that
// was there. Written the way Save writes a design, so an interrupted save
// never leaves half a trace for the next open to choke on.
func (s *DesignService) SaveTrace(designPath string, trace string) (string, error) {
	path, err := tracePath(designPath)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(trace) == "" {
		return "", errors.New("refusing to write an empty trace")
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(trace), 0o644); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("replace %s: %w", path, err)
	}
	return path, nil
}

// Recent returns the recent designs, most recently opened first, each flagged
// if it has since been moved or deleted.
func (s *DesignService) Recent() []RecentEntry {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]RecentEntry, 0, len(s.recent))
	for _, e := range s.recent {
		_, err := os.Stat(e.Path)
		e.Missing = err != nil
		out = append(out, e)
	}
	return out
}

// ClearRecent forgets the recent list.
func (s *DesignService) ClearRecent() error {
	s.mu.Lock()
	s.recent = nil
	s.mu.Unlock()
	return s.saveRecent()
}

// RevealInFolder opens the platform's file manager at the design's folder.
func (s *DesignService) RevealInFolder(path string) error {
	if path == "" {
		return errors.New("this design has not been saved yet")
	}
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("%s no longer exists", path)
	}
	// Wails hands the path to the desktop shell.
	return s.app.Browser.OpenURL("file://" + filepath.ToSlash(filepath.Dir(path)))
}

// --- recent list persistence ------------------------------------------------

const maxRecent = 12

func (s *DesignService) recentPath() string {
	return filepath.Join(s.stateDir, "recent.json")
}

func (s *DesignService) remember(path string) {
	s.mu.Lock()
	entry := RecentEntry{Path: path, Name: designName(path), OpenedAt: time.Now().Format(time.RFC3339)}
	kept := []RecentEntry{entry}
	for _, e := range s.recent {
		if strings.EqualFold(e.Path, path) {
			continue
		}
		kept = append(kept, e)
		if len(kept) >= maxRecent {
			break
		}
	}
	s.recent = kept
	s.mu.Unlock()
	_ = s.saveRecent()
}

func (s *DesignService) markMissing(path string) {
	s.mu.Lock()
	for i := range s.recent {
		if strings.EqualFold(s.recent[i].Path, path) {
			s.recent[i].Missing = true
		}
	}
	s.mu.Unlock()
}

func (s *DesignService) loadRecent() {
	data, err := os.ReadFile(s.recentPath())
	if err != nil {
		return
	}
	var list []RecentEntry
	if err := json.Unmarshal(data, &list); err != nil {
		return
	}
	sort.SliceStable(list, func(i, j int) bool { return list[i].OpenedAt > list[j].OpenedAt })
	if len(list) > maxRecent {
		list = list[:maxRecent]
	}
	s.mu.Lock()
	s.recent = list
	s.mu.Unlock()
}

func (s *DesignService) saveRecent() error {
	s.mu.Lock()
	data, err := json.MarshalIndent(s.recent, "", "  ")
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.stateDir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(s.recentPath(), data, 0o644)
}

// designName strips both the .json and the .tensorcad halves of the extension.
func designName(path string) string {
	base := filepath.Base(path)
	base = strings.TrimSuffix(base, ".json")
	base = strings.TrimSuffix(base, ".tensorcad")
	return base
}
