/**
 * Cross-check the analysis against PyTorch itself.
 *
 * These tests shell out to `tensorcad-runtime verify`, which imports the generated
 * module, instantiates it on the meta device and counts parameters. That closes
 * the loop the rest of the suite cannot: everywhere else we compare our own
 * numbers to our own numbers.
 *
 * The Python runtime and PyTorch are optional, so the whole block skips (it does
 * not fail) when either is missing. See python/README.md for install steps.
 */
export {};
