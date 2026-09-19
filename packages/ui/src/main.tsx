import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./app/app.css";
import App from "./app/App.js";
import { initTheme } from "./state/theme.js";

// Set the theme before the first paint.
initTheme();

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from index.html");
createRoot(container).render(<App />);
