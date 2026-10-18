// Lee por stdin una lista JSON [{tex, display}] y escribe por stdout la lista de HTML generados por KaTeX.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const katex = require("katex");
const items = JSON.parse(readFileSync(0, "utf8"));
const out = items.map(({ tex, display }) =>
  katex.renderToString(tex, { displayMode: !!display, throwOnError: false, output: "htmlAndMathml", strict: "ignore" }));
process.stdout.write(JSON.stringify(out));
