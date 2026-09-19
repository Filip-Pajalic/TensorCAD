package services

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/wailsapp/wails/v3/pkg/application"
)

// WorkspaceService writes what the design produces: generated PyTorch, the
// design that produced it, and a training script. The TypeScript core decides
// what the files contain; this decides where they go and reports it back.
type WorkspaceService struct {
	app *application.App
	// root is where generated output is placed when the user does not choose.
	root string
}

// GeneratedFile is one file the core produced.
type GeneratedFile struct {
	Path     string `json:"path"`
	Contents string `json:"contents"`
}

// WriteResult says what landed on disk.
type WriteResult struct {
	Directory string   `json:"directory"`
	Written   []string `json:"written"`
	Bytes     int64    `json:"bytes"`
}

func NewWorkspaceService(app *application.App, root string) *WorkspaceService {
	return &WorkspaceService{app: app, root: root}
}

func (s *WorkspaceService) ServiceName() string { return "Workspace" }

// DefaultRoot is where generated output goes unless the user picks elsewhere.
func (s *WorkspaceService) DefaultRoot() string { return s.root }

// ChooseDirectory asks for a folder. An empty result means the user cancelled.
func (s *WorkspaceService) ChooseDirectory(title string) (string, error) {
	dialog := s.app.Dialog.OpenFile()
	if title == "" {
		title = "Choose a folder"
	}
	dialog.SetTitle(title)
	dialog.CanChooseFiles(false)
	dialog.CanChooseDirectories(true)
	path, err := dialog.PromptForSingleSelection()
	if err != nil {
		return "", fmt.Errorf("folder dialog: %w", err)
	}
	return path, nil
}

// Write puts generated files into a folder named after the design, under the
// given directory. Passing an empty directory uses the default root.
func (s *WorkspaceService) Write(directory string, designName string, files []GeneratedFile) (*WriteResult, error) {
	if len(files) == 0 {
		return nil, errors.New("nothing to write")
	}
	if directory == "" {
		directory = s.root
	}
	if designName == "" {
		designName = "design"
	}
	target := filepath.Join(directory, safeName(designName))
	if err := os.MkdirAll(target, 0o755); err != nil {
		return nil, fmt.Errorf("create %s: %w", target, err)
	}

	result := &WriteResult{Directory: target}
	for _, f := range files {
		// A generated path is relative by construction; refuse anything that
		// tries to climb out of the target folder.
		clean := filepath.Clean(filepath.FromSlash(f.Path))
		if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") {
			return nil, fmt.Errorf("refusing to write outside the target folder: %s", f.Path)
		}
		full := filepath.Join(target, clean)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return nil, fmt.Errorf("create folder for %s: %w", clean, err)
		}
		if err := os.WriteFile(full, []byte(f.Contents), 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", clean, err)
		}
		result.Written = append(result.Written, full)
		result.Bytes += int64(len(f.Contents))
	}
	sort.Strings(result.Written)
	return result, nil
}

// Reveal opens the platform file manager at a folder.
func (s *WorkspaceService) Reveal(path string) error {
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("%s does not exist", path)
	}
	return s.app.Browser.OpenURL("file://" + filepath.ToSlash(path))
}

// ReadTextFile returns a small text file, for showing generated code in the UI
// without a second round trip through the filesystem.
func (s *WorkspaceService) ReadTextFile(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("%s does not exist", path)
	}
	const limit = 4 << 20
	if info.Size() > limit {
		return "", fmt.Errorf("%s is %d bytes, larger than the %d byte limit", filepath.Base(path), info.Size(), limit)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	return string(data), nil
}

// safeName keeps a design name usable as a folder name on every platform.
func safeName(name string) string {
	replacer := strings.NewReplacer(
		"/", "-", "\\", "-", ":", "-", "*", "-",
		"?", "-", "\"", "-", "<", "-", ">", "-", "|", "-",
	)
	cleaned := strings.TrimSpace(replacer.Replace(name))
	cleaned = strings.Trim(cleaned, ".")
	if cleaned == "" {
		return "design"
	}
	return cleaned
}
