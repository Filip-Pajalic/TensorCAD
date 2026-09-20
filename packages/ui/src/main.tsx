import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./app/app.css";
import { loadEngine } from "./engine.js";
import { initTheme } from "./state/theme.js";

// Set the theme before the first paint.
initTheme();

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from index.html");
const root = createRoot(container);

/**
 * Nothing is loaded until the engine is.
 *
 * The app is imported after the engine rather than beside it, and that is not
 * tidiness: the store builds a starting design the moment its module runs, and
 * a design comes from the engine. Importing the tree first would run that line
 * before there was anything to ask.
 *
 * There is no useful half-loaded editor either. Every panel reads the analysis,
 * the palette reads the catalog and the canvas reads the shapes, so a frame
 * without them would be a frame of empty boxes.
 */
loadEngine()
  .then(async () => {
    const { default: App } = await import("./app/App.js");
    root.render(<App />);

    // After the first frame, not before it. Looking for an agent means four
    // fetches that will usually find nothing, and nothing about the editor
    // waits on the answer — the status bar says "none" until it says
    // otherwise.
    const { connect } = await import("./state/bridge.js");
    void connect();
  })
  .catch((error: unknown) => {
    root.render(
      <div className="flex h-dvh items-center justify-center p-8 text-center text-sm">
        <div>
          <p className="font-medium">The analysis engine did not load.</p>
          <p className="mt-2 opacity-70">{String(error)}</p>
        </div>
      </div>,
    );
  });
