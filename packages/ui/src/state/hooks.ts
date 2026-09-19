import { useMemo } from "react";
import { useEditor } from "./store.js";
import { derive, type Derived } from "./derive.js";
import { resolveLevel, type Level } from "./level.js";

export function useDerived(): Derived {
  const doc = useEditor((s) => s.doc);
  const operating = useEditor((s) => s.operating);
  return useMemo(() => derive(doc, operating), [doc, operating]);
}

export function useLevel(): { level: Level; derived: Derived } {
  const doc = useEditor((s) => s.doc);
  const path = useEditor((s) => s.path);
  const derived = useDerived();
  const level = useMemo(() => resolveLevel(doc, path, derived), [doc, path, derived]);
  return { level, derived };
}
