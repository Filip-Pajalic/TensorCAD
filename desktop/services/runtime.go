package services

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/wailsapp/wails/v3/pkg/application"
)

// RuntimeService runs the Python side of TensorCAD: verifying a generated model
// against PyTorch, and training a scaled-down design on the local GPU.
//
// These are long jobs that print progress. Rather than block a call for
// minutes, a job is started, given an id, and its output is emitted as events
// the frontend subscribes to. That is the whole reason this belongs in Go
// rather than in the browser.
type RuntimeService struct {
	app *application.App

	mu   sync.Mutex
	jobs map[string]*job
	seq  int
}

type job struct {
	id     string
	cancel context.CancelFunc
	cmd    *exec.Cmd
}

// JobStarted is returned when a job is accepted.
type JobStarted struct {
	ID      string `json:"id"`
	Command string `json:"command"`
}

// JobLine is one line of output from a running job.
type JobLine struct {
	ID string `json:"id"`
	// Stream is "stdout" or "stderr".
	Stream string `json:"stream"`
	Text   string `json:"text"`
}

// JobDone reports how a job ended.
type JobDone struct {
	ID       string `json:"id"`
	ExitCode int    `json:"exitCode"`
	// Result is the final JSON object the runtime printed on stdout, if any.
	Result map[string]any `json:"result,omitempty"`
	Error  string         `json:"error,omitempty"`
	// Seconds the job ran for.
	Seconds float64 `json:"seconds"`
}

// Environment describes whether the Python side is usable, so the UI can say
// what is missing instead of failing when a button is pressed.
type Environment struct {
	Python        string `json:"python"`
	PythonVersion string `json:"pythonVersion"`
	RuntimeFound  bool   `json:"runtimeFound"`
	TorchVersion  string `json:"torchVersion"`
	CudaAvailable bool   `json:"cudaAvailable"`
	DeviceName    string `json:"deviceName"`
	Detail        string `json:"detail"`
}

func NewRuntimeService(app *application.App) *RuntimeService {
	return &RuntimeService{app: app, jobs: map[string]*job{}}
}

func (s *RuntimeService) ServiceName() string { return "Runtime" }

// Probe reports what the Python environment can do. It runs a short script, so
// it is safe to call on startup.
func (s *RuntimeService) Probe() Environment {
	env := Environment{}

	python := findPython()
	if python == "" {
		env.Detail = "No Python interpreter was found on PATH."
		return env
	}
	env.Python = python

	if out, err := runShort(python, "--version"); err == nil {
		env.PythonVersion = strings.TrimSpace(strings.TrimPrefix(out, "Python "))
	}

	const script = `
import json, importlib.util
out = {"runtime": importlib.util.find_spec("tensorcad_runtime") is not None}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    out["device"] = torch.cuda.get_device_name(0) if out["cuda"] else ""
except Exception as e:
    out["torch"] = ""
    out["cuda"] = False
    out["device"] = ""
    out["error"] = str(e)[:200]
print(json.dumps(out))
`
	out, err := runShort(python, "-c", script)
	if err != nil {
		env.Detail = "Python is present but could not be queried: " + err.Error()
		return env
	}
	var probe struct {
		Runtime bool   `json:"runtime"`
		Torch   string `json:"torch"`
		Cuda    bool   `json:"cuda"`
		Device  string `json:"device"`
		Error   string `json:"error"`
	}
	if err := json.Unmarshal([]byte(lastJSONLine(out)), &probe); err != nil {
		env.Detail = "Could not read the environment probe: " + err.Error()
		return env
	}

	env.RuntimeFound = probe.Runtime
	env.TorchVersion = probe.Torch
	env.CudaAvailable = probe.Cuda
	env.DeviceName = probe.Device

	switch {
	case !probe.Runtime:
		env.Detail = "tensorcad_runtime is not installed. Run: pip install -e python/tensorcad_runtime"
	case probe.Torch == "":
		env.Detail = "PyTorch is not installed. " + probe.Error
	case !probe.Cuda:
		env.Detail = "PyTorch is installed but no CUDA device is available; jobs will run on the CPU."
	default:
		env.Detail = fmt.Sprintf("Ready: torch %s on %s", probe.Torch, probe.Device)
	}
	return env
}

// Verify instantiates a generated model and checks its parameter count against
// what the design says.
func (s *RuntimeService) Verify(modelPath string, batch int, seq int) (*JobStarted, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("%s does not exist", modelPath)
	}
	args := []string{"-m", "tensorcad_runtime", "verify", modelPath}
	if batch > 0 {
		args = append(args, "--batch", fmt.Sprint(batch))
	}
	if seq > 0 {
		args = append(args, "--seq", fmt.Sprint(seq))
	}
	return s.start(args, filepath.Dir(modelPath))
}

// SmokeTrain trains a generated model briefly and streams its loss curve.
func (s *RuntimeService) SmokeTrain(modelPath string, steps int, batch int, seq int) (*JobStarted, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("%s does not exist", modelPath)
	}
	args := []string{"-m", "tensorcad_runtime", "smoke-train", modelPath}
	if steps > 0 {
		args = append(args, "--steps", fmt.Sprint(steps))
	}
	if batch > 0 {
		args = append(args, "--batch", fmt.Sprint(batch))
	}
	if seq > 0 {
		args = append(args, "--seq", fmt.Sprint(seq))
	}
	return s.start(args, filepath.Dir(modelPath))
}

// TraceStarted is a trace job, and where the trace will be once it is done.
type TraceStarted struct {
	ID      string `json:"id"`
	Command string `json:"command"`
	Out     string `json:"out"`
}

// Trace runs a design small enough to look at and records what it computes,
// for the editor's volume view and walkthrough.
//
// The editor generates the model and hands the files over; they go into a
// scratch folder of this service's own rather than anywhere the user chose,
// because a trace is something to look at, not something to keep — the
// runtime can make it again in seconds. When the job is done the frontend
// reads the result with ReadTrace.
func (s *RuntimeService) Trace(designName string, files []GeneratedFile) (*TraceStarted, error) {
	if len(files) == 0 {
		return nil, errors.New("nothing to trace: generate the model first")
	}
	written, err := writeFiles(filepath.Join(traceRoot(), safeName(designName)), files)
	if err != nil {
		return nil, err
	}
	dir := written.Directory
	model := filepath.Join(dir, "model.py")
	if _, err := os.Stat(model); err != nil {
		return nil, errors.New("the generated files have no model.py")
	}
	out := filepath.Join(dir, "trace.json")
	// Last run's, which would otherwise be read if this one fails early.
	_ = os.Remove(out)
	started, err := s.start([]string{"-m", "tensorcad_runtime", "trace", model, "--out", out}, dir)
	if err != nil {
		return nil, err
	}
	return &TraceStarted{ID: started.ID, Command: started.Command, Out: out}, nil
}

// maxTraceBytes is what ReadTrace will return. A trace of the largest design
// the runtime agrees to trace, at the longest input, is well under it.
const maxTraceBytes = 64 << 20

// ReadTrace returns a trace this service wrote, and nothing else: the path has
// to be a trace.json under its own scratch folder. The frontend can reach this
// with any string, so it is not a way to read an arbitrary file.
func (s *RuntimeService) ReadTrace(path string) (string, error) {
	clean := filepath.Clean(path)
	rel, err := filepath.Rel(traceRoot(), clean)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) || filepath.Base(clean) != "trace.json" {
		return "", fmt.Errorf("%s is not a trace this app made", path)
	}
	info, err := os.Stat(clean)
	if err != nil {
		return "", fmt.Errorf("no trace was written: %s does not exist", clean)
	}
	if info.Size() > maxTraceBytes {
		return "", fmt.Errorf("the trace is %d bytes, more than the %d this reads", info.Size(), maxTraceBytes)
	}
	data, err := os.ReadFile(clean)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", clean, err)
	}
	return string(data), nil
}

// traceRoot is where traces are made: a folder of this app's own under the
// system's temporary directory.
func traceRoot() string {
	return filepath.Join(os.TempDir(), "tensorcad-trace")
}

// Cancel stops a running job.
func (s *RuntimeService) Cancel(id string) error {
	s.mu.Lock()
	j, ok := s.jobs[id]
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("no job %s is running", id)
	}
	j.cancel()
	return nil
}

// Running lists the jobs currently in flight.
func (s *RuntimeService) Running() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.jobs))
	for id := range s.jobs {
		ids = append(ids, id)
	}
	return ids
}

func (s *RuntimeService) start(args []string, workDir string) (*JobStarted, error) {
	python := findPython()
	if python == "" {
		return nil, errors.New("no Python interpreter was found on PATH")
	}

	s.mu.Lock()
	s.seq++
	id := fmt.Sprintf("job-%d", s.seq)
	s.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, python, args...)
	cmd.Dir = workDir
	// Unbuffered, or progress arrives in one lump when the job ends.
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("capture stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("capture stderr: %w", err)
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start %s: %w", strings.Join(args, " "), err)
	}

	s.mu.Lock()
	s.jobs[id] = &job{id: id, cancel: cancel, cmd: cmd}
	s.mu.Unlock()

	started := time.Now()
	var lastStdout string
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			line := sc.Text()
			lastStdout = line
			s.app.Event.Emit("runtime:line", JobLine{ID: id, Stream: "stdout", Text: line})
		}
	}()
	go func() {
		defer wg.Done()
		sc := bufio.NewScanner(stderr)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			s.app.Event.Emit("runtime:line", JobLine{ID: id, Stream: "stderr", Text: sc.Text()})
		}
	}()

	go func() {
		wg.Wait()
		runErr := cmd.Wait()
		cancel()

		s.mu.Lock()
		delete(s.jobs, id)
		s.mu.Unlock()

		done := JobDone{ID: id, Seconds: time.Since(started).Seconds()}
		if runErr != nil {
			done.Error = runErr.Error()
		}
		if cmd.ProcessState != nil {
			done.ExitCode = cmd.ProcessState.ExitCode()
		}
		// The runtime prints one JSON object as its last line of stdout.
		if lastStdout != "" {
			var result map[string]any
			if json.Unmarshal([]byte(lastStdout), &result) == nil {
				done.Result = result
			}
		}
		s.app.Event.Emit("runtime:done", done)
	}()

	return &JobStarted{ID: id, Command: python + " " + strings.Join(args, " ")}, nil
}

// --- helpers ----------------------------------------------------------------

func findPython() string {
	for _, name := range []string{"python", "python3", "py"} {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}
	return ""
}

func runShort(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
	return string(out), err
}

// lastJSONLine returns the last line that looks like a JSON object, which is
// how every tensorcad_runtime command reports its result.
func lastJSONLine(out string) string {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "{") && strings.HasSuffix(line, "}") {
			return line
		}
	}
	return "{}"
}
