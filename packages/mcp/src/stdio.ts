#!/usr/bin/env bun
/**
 * The program. `index.ts` is the library; this is the thing you run.
 *
 * Two files rather than one with a guard in it, because a module that starts a
 * server when it happens to be the entry point is a module whose behaviour
 * depends on how it was loaded — which is exactly what broke when it was first
 * put through a bundler for the desktop bundle.
 */

import { serve } from "./serve.js";

await serve();
