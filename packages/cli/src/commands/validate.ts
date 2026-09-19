/**
 * `tensorcad validate <file|preset>` — the design-rule check.
 *
 * Exits 1 when anything is an error, which makes it usable in CI.
 */

import { formatCount, validate, type Finding, type Severity } from "@tensorcad/core";
import { bool, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { analysisOptions, impliedGpus } from "../options.js";
import { bold, dim, green, heading, red, writeOut, yellow, blue } from "../format.js";

const ORDER: Severity[] = ["error", "warning", "info"];

const PAINT: Record<Severity, (s: string) => string> = {
  error: red,
  warning: yellow,
  info: blue,
};

function renderGroup(severity: Severity, findings: Finding[]): string[] {
  if (findings.length === 0) return [];
  const paint = PAINT[severity];
  const out: string[] = [heading(paint(`${severity}${findings.length === 1 ? "" : "s"} (${findings.length})`))];
  for (const f of findings) {
    const where = f.path ? `${f.path}${f.port ? `:${f.port}` : ""}` : "";
    out.push(`  ${paint("*")} ${f.message}`);
    out.push(`    ${dim(`${f.rule}${where ? `  ${where}` : ""}`)}`);
    if (f.hint) out.push(`    ${dim("fix:")} ${f.hint}`);
  }
  return out;
}

export function cmdValidate(args: Args): number {
  const { doc } = loadDesign(args._[0]);
  const options = impliedGpus(analysisOptions(args));
  const report = validate(doc, options);

  if (bool(args, "json")) {
    writeOut(
      JSON.stringify(
        {
          name: report.name,
          ok: report.ok,
          counts: report.counts,
          findings: report.findings,
          params: report.analysis.params.total,
        },
        null,
        2,
      ),
    );
    return report.ok ? 0 : 1;
  }

  const out: string[] = [];
  out.push(
    `${bold(report.name)}  ${dim(`${formatCount(report.analysis.params.total)} parameters, T=${report.analysis.options.T}`)}`,
  );

  for (const severity of ORDER) {
    out.push(...renderGroup(severity, report.findings.filter((f) => f.severity === severity)));
  }

  const { error, warning, info } = report.counts;
  out.push("");
  if (report.findings.length === 0) {
    out.push(green("No findings. The design passes every rule."));
  } else {
    const parts = [
      error > 0 ? red(`${error} error${error === 1 ? "" : "s"}`) : null,
      warning > 0 ? yellow(`${warning} warning${warning === 1 ? "" : "s"}`) : null,
      info > 0 ? blue(`${info} info`) : null,
    ].filter(Boolean);
    out.push(`${report.ok ? green("ok") : red("failed")}  ${parts.join(", ")}`);
  }

  writeOut(out.join("\n"));
  return report.ok ? 0 : 1;
}
