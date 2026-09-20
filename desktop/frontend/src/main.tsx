/**
 * Desktop entry point.
 *
 * Mounts the same editor the browser build uses, then connects it to the Go
 * shell. The theme starts before the first paint so the window never flashes
 * the wrong one.
 */

import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "@tensor-cad/ui/style.css";
import { App, initTheme } from "@tensor-cad/ui";
import { connect } from "./bridge.js";

initTheme();

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from index.html");
createRoot(container).render(<App />);

connect();
