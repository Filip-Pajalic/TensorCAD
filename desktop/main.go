// TensorCAD desktop.
//
// A Wails application: Go owns the parts that need the operating system, and
// the drawing surface stays in the frontend where it has to be. Go owns files,
// native dialogs and menus, the theme, the Python jobs that verify and train a
// design, and — as the port lands — the analysis engine itself.
//
// The engine is registered as a service here but the window still computes its
// own numbers, because the analysis reruns on every keystroke and a process
// boundary in that loop has to be shown not to cost anything before it is put
// there. Until then the Engine service is what the CLI, the MCP server and any
// out-of-window caller use.
package main

import (
	"embed"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/tensorcad/desktop/services"
	"github.com/wailsapp/wails/v3/pkg/application"
	"github.com/wailsapp/wails/v3/pkg/events"
)

//go:embed all:frontend/dist
var assets embed.FS

func init() {
	// Events the frontend subscribes to. Registering them here gives the
	// binding generator enough to emit typed helpers.
	application.RegisterEvent[services.JobLine]("runtime:line")
	application.RegisterEvent[services.JobDone]("runtime:done")
	application.RegisterEvent[string]("menu:command")
	application.RegisterEvent[string]("theme:changed")
}

func main() {
	stateDir := appStateDir()
	workspaceRoot := defaultWorkspaceRoot()

	app := application.New(application.Options{
		Name:        "TensorCAD",
		Description: "Design transformer architectures, check them, and generate the model.",
		Assets: application.AssetOptions{
			Handler: application.AssetFileServerFS(assets),
		},
		Mac: application.MacOptions{
			ApplicationShouldTerminateAfterLastWindowClosed: true,
		},
	})

	designs := services.NewDesignService(app, stateDir)
	app.RegisterService(application.NewService(designs))
	app.RegisterService(application.NewService(services.NewRuntimeService(app)))
	app.RegisterService(application.NewService(services.NewWorkspaceService(app, workspaceRoot)))
	// The analysis engine itself, so the window can ask this process for the
	// numbers rather than computing them in JavaScript.
	app.RegisterService(application.NewService(services.NewEngineService()))

	buildMenu(app)

	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Title:  "TensorCAD",
		Width:  1600,
		Height: 1000,
		// Below this the three docks and the sheet stop coexisting.
		MinWidth:  1100,
		MinHeight: 680,
		Mac: application.MacWindow{
			InvisibleTitleBarHeight: 50,
			Backdrop:                application.MacBackdropTranslucent,
			TitleBar:                application.MacTitleBarHiddenInset,
		},
		Windows: application.WindowsWindow{
			// Follow the system accent and theme rather than fighting them.
			Theme: application.SystemDefault,
		},
		// Matches the light sheet so the window does not flash dark on open.
		BackgroundColour: application.NewRGB(216, 218, 222),
		URL:              "/",
	})

	// Tell the frontend which way the system theme went, so the drawing can
	// follow it without the web layer having to guess.
	app.Event.OnApplicationEvent(events.Common.ThemeChanged, func(*application.ApplicationEvent) {
		app.Event.Emit("theme:changed", themeName(app))
	})

	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}

// buildMenu puts the commands where the platform expects them. Each item emits
// one event; the frontend owns what the command means, because it owns the
// document.
func buildMenu(app *application.App) {
	menu := app.Menu.New()

	if runtime.GOOS == "darwin" {
		menu.AddRole(application.AppMenu)
	}

	file := menu.AddSubmenu("File")
	file.Add("New design").SetAccelerator("CmdOrCtrl+n").OnClick(command(app, "new"))
	file.Add("Open…").SetAccelerator("CmdOrCtrl+o").OnClick(command(app, "open"))
	file.Add("Save").SetAccelerator("CmdOrCtrl+s").OnClick(command(app, "save"))
	file.Add("Save as…").SetAccelerator("CmdOrCtrl+shift+s").OnClick(command(app, "save-as"))
	file.AddSeparator()
	file.Add("Generate PyTorch…").SetAccelerator("CmdOrCtrl+g").OnClick(command(app, "generate"))
	if runtime.GOOS != "darwin" {
		file.AddSeparator()
		file.Add("Quit").SetAccelerator("CmdOrCtrl+q").OnClick(func(*application.Context) { app.Quit() })
	}

	edit := menu.AddSubmenu("Edit")
	edit.Add("Undo").SetAccelerator("CmdOrCtrl+z").OnClick(command(app, "undo"))
	edit.Add("Redo").SetAccelerator("CmdOrCtrl+shift+z").OnClick(command(app, "redo"))
	edit.AddSeparator()
	edit.Add("Lock selection").SetAccelerator("CmdOrCtrl+l").OnClick(command(app, "lock"))
	edit.Add("Delete selection").SetAccelerator("delete").OnClick(command(app, "delete"))

	view := menu.AddSubmenu("View")
	view.Add("Zoom to fit").SetAccelerator("CmdOrCtrl+0").OnClick(command(app, "fit"))
	view.Add("Arrange").SetAccelerator("CmdOrCtrl+shift+a").OnClick(command(app, "arrange"))
	view.AddSeparator()
	view.Add("Annotations").SetAccelerator("CmdOrCtrl+1").OnClick(command(app, "toggle-callouts"))
	view.Add("Numeric shapes").SetAccelerator("CmdOrCtrl+2").OnClick(command(app, "toggle-shape-mode"))
	view.AddSeparator()
	view.Add("Light theme").OnClick(command(app, "theme:light"))
	view.Add("Dark theme").OnClick(command(app, "theme:dark"))
	view.Add("Follow system").OnClick(command(app, "theme:system"))

	design := menu.AddSubmenu("Design")
	design.Add("Run checks").SetAccelerator("F5").OnClick(command(app, "validate"))
	design.Add("Verify against PyTorch").OnClick(command(app, "verify"))
	design.Add("Smoke train…").OnClick(command(app, "smoke-train"))
	design.Add("Trace this design").OnClick(command(app, "trace"))

	help := menu.AddSubmenu("Help")
	help.Add("Documentation").OnClick(func(*application.Context) {
		app.Browser.OpenURL("https://github.com/Filip-Pajalic/TensorCAD")
	})

	app.Menu.Set(menu)
}

func command(app *application.App, name string) func(*application.Context) {
	return func(*application.Context) { app.Event.Emit("menu:command", name) }
}

func themeName(app *application.App) string {
	if app.Env.IsDarkMode() {
		return "dark"
	}
	return "light"
}

// appStateDir is where the recent list and window state live.
func appStateDir() string {
	base, err := os.UserConfigDir()
	if err != nil {
		base, _ = os.UserHomeDir()
	}
	dir := filepath.Join(base, "TensorCAD")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

// defaultWorkspaceRoot is where generated models go unless told otherwise.
func defaultWorkspaceRoot() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	dir := filepath.Join(home, "TensorCAD")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}
