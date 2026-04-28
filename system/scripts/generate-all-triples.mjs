import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildBatchExtractionPayload } from "../src/utils/batchExtraction.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const inputPath = path.join(rootDir, "input.json");
const outputPath = path.join(rootDir, "src", "data", "allDocumentTriples.json");

const raw = await fs.readFile(inputPath, "utf8");
const rawDocuments = JSON.parse(raw);

const documents = rawDocuments.map((item, index) => ({
  id: item.id || String(index + 1),
  eventList: item.event_list || [],
  entityMentions: item.entity_mentions || [],
  relationTriples: item.relation_triples || [],
}));

const payload = buildBatchExtractionPayload(documents);

await fs.writeFile(outputPath, JSON.stringify(payload, null, 2), "utf8");

console.log(
  `Generated ${payload.totalUniqueTriples} unique triples from ${payload.totalDocuments} documents.`,
);
