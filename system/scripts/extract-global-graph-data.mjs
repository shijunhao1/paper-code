import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const sourcePath = path.join(rootDir, "src", "data", "global_graph_source.html");
const outputPath = path.join(rootDir, "src", "data", "globalGraphData.json");

const html = await fs.readFile(sourcePath, "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const inlineScript = scripts.at(-1)?.[1];

if (!inlineScript) {
  throw new Error("Failed to locate inline graph script in global_graph_source.html");
}

const sandbox = {
  window: {},
  document: {
    getElementById() {
      return {
        innerHTML: "",
      };
    },
  },
  console,
};

vm.createContext(sandbox);
vm.runInContext(inlineScript, sandbox, {
  timeout: 15000,
});

const nodesData = vm.runInContext("nodesData", sandbox);
const edgesData = vm.runInContext("edgesData", sandbox);

await fs.writeFile(
  outputPath,
  JSON.stringify({ nodesData, edgesData }, null, 2),
  "utf8",
);

console.log(
  `Extracted ${nodesData.length} nodes and ${edgesData.length} edges to globalGraphData.json.`,
);
