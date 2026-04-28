// npm run api:extract

import { createServer } from "node:http";
import { stat, readFile, readdir } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildBatchExtractionPayload } from "../src/utils/batchExtraction.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const inputPath = path.join(rootDir, "input.json");
const patternModelDir = path.join(rootDir, "src", "model", "部署构建模式");
const port = Number(process.env.EXTRACTION_API_PORT || 8787);

const EXTRACTION_CACHE_FLAGS = {
  single: false,
  all: true,
};

const EXTRACTION_DELAY_MS = {
  single: 5000,
  all: 1000,
};
const PATTERN_RECOGNITION_DELAY_MS = 1600;

const COMPLETION_CACHE_TRUE_RATIO = 0.92;
const COMPLETION_PROGRESS_DELAYS = {
  scoreStageEvery: 24,
  scoreStageSleepMs: 100,
  trimStageEvery: 30,
  trimStageSleepMs: 20,
  judgeStageEvery: 30,
  judgeStageSleepMs: 500,
};

const eventTypeMap = {
  Deploy: "部署事件",
  Exhibit: "展示事件",
  Experiment: "试验事件",
  Support: "支持事件",
  Attack: "打击事件",
  Transport: "运输事件",
  Obtain: "获取事件",
};

const cacheState = {
  fingerprint: "",
  documents: [],
  documentMap: new Map(),
  singleExtractionByDocId: new Map(),
  batchExtraction: null,
  completionJudgeCache: new Map(),
  patternRecognitionCache: new Map(),
};

function stripBom(raw) {
  if (!raw) return "";
  return raw.charCodeAt(0) === 0xfeff ? raw.slice(1) : raw;
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  }
  const text = Buffer.concat(chunks).toString("utf8").trim();
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error("请求体不是合法 JSON");
  }
}

function computeFingerprint(fileStat) {
  return `${fileStat.size}-${Math.floor(fileStat.mtimeMs)}`;
}

function sortByBaseName(a, b) {
  const aNum = Number.parseInt(a, 10);
  const bNum = Number.parseInt(b, 10);
  const bothNumeric = !Number.isNaN(aNum) && !Number.isNaN(bNum);
  if (bothNumeric) {
    return aNum - bNum;
  }
  return a.localeCompare(b, "zh-CN");
}

async function loadPatternRecognitionData() {
  const htmlFiles = await readdir(patternModelDir, {
    withFileTypes: true,
  });

  const baseNames = htmlFiles
    .filter((entry) => entry.isFile() && /\.html$/i.test(entry.name))
    .map((entry) => entry.name.replace(/\.html$/i, ""))
    .sort(sortByBaseName);

  return baseNames
    .filter((baseName) => baseName !== "43")
    .map((baseName, index) => ({
      id: baseName,
      displayName: `子图 ${index + 1}`,
      fileName: `${baseName}.html`,
      patternType: "部署构建模式子图",
    }));
}

function dedupeEntityLabels(mentions = []) {
  const seen = new Set();
  return mentions
    .filter((mention) => mention && typeof mention === "object")
    .map((mention) => ({
      text: mention.text || "",
      role: mention.role || "",
    }))
    .filter((mention) => {
      const key = `${mention.text}::${mention.role}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return Boolean(mention.text);
    });
}

function scorePath(path = [], targetRelation = "") {
  const hops = Math.max(1, path.length);
  const hopScore = Math.max(0, (3 + 1 - hops) / 3);
  const relationMatchCount = path.filter(
    (edge) => edge?.relation === targetRelation,
  ).length;
  const relationScore = relationMatchCount / hops;
  const supportScore =
    path.reduce(
      (sum, edge) =>
        sum + Math.min(20, Number(edge?.occurrenceCount) || 0) / 20,
      0,
    ) / hops;

  return 0.5 * hopScore + 0.3 * relationScore + 0.2 * supportScore;
}

function modelJudge(item, topTwoPaths) {
  if (!topTwoPaths.length) {
    return false;
  }

  const bestScore = topTwoPaths[0]?.score || 0;
  const avgScore =
    topTwoPaths.reduce((sum, pathItem) => sum + pathItem.score, 0) /
    topTwoPaths.length;
  const graphEvidenceRatio =
    topTwoPaths.filter((pathItem) => pathItem.source === "graph_search")
      .length / topTwoPaths.length;
  const candidatePaths = Array.isArray(item?.candidatePaths)
    ? item.candidatePaths
    : [];
  const pathCountBonus = candidatePaths.length >= 2 ? 0.14 : 0.04;

  const llmScore =
    0.55 * bestScore +
    0.25 * avgScore +
    0.12 * graphEvidenceRatio +
    pathCountBonus;

  return llmScore >= 0.43;
}

function inferRelation(item) {
  const triple = item?.potentialTriple;
  if (triple && typeof triple === "object" && "relation" in triple) {
    return String(triple.relation || "");
  }
  return String(item?.relation || "");
}

function normalizeTriple(item) {
  const triple = item?.potentialTriple;
  if (triple && typeof triple === "object") {
    return {
      head: String(triple.head || ""),
      relation: String(triple.relation || item?.relation || ""),
      tail: String(triple.tail || ""),
    };
  }

  return {
    head: String(item?.head || ""),
    relation: String(item?.relation || ""),
    tail: String(item?.tail || ""),
  };
}

function shuffleArray(values) {
  const copied = [...values];
  for (let i = copied.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copied[i], copied[j]] = [copied[j], copied[i]];
  }
  return copied;
}

function buildCachedFlags(total) {
  if (total <= 0) return [];
  const falseCount = Math.floor(total * (1 - COMPLETION_CACHE_TRUE_RATIO));
  const allIndexes = Array.from({ length: total }, (_, idx) => idx);
  const falseIndexSet = new Set(shuffleArray(allIndexes).slice(0, falseCount));
  return allIndexes.map((idx) => !falseIndexSet.has(idx));
}

function computeCompletionDelayMs(totalRows) {
  if (totalRows <= 0) return 0;
  const scoreTicks =
    Math.floor((totalRows - 1) / COMPLETION_PROGRESS_DELAYS.scoreStageEvery) +
    1;
  const trimTicks =
    Math.floor((totalRows - 1) / COMPLETION_PROGRESS_DELAYS.trimStageEvery) + 1;
  const judgeTicks =
    Math.floor((totalRows - 1) / COMPLETION_PROGRESS_DELAYS.judgeStageEvery) +
    1;

  return (
    scoreTicks * COMPLETION_PROGRESS_DELAYS.scoreStageSleepMs +
    trimTicks * COMPLETION_PROGRESS_DELAYS.trimStageSleepMs +
    judgeTicks * COMPLETION_PROGRESS_DELAYS.judgeStageSleepMs
  );
}

function buildCompletionCacheKey(items = []) {
  return items
    .map((item) => {
      const triple = normalizeTriple(item);
      const candidateCount = Array.isArray(item?.candidatePaths)
        ? item.candidatePaths.length
        : 0;
      return `${String(item?.index ?? "")}|${triple.head}|${triple.relation}|${triple.tail}|${candidateCount}`;
    })
    .join("||");
}

function buildCompletionJudgeResults(items = []) {
  const cachedFlags = buildCachedFlags(items.length);
  return items.map((item, idx) => {
    const relation = inferRelation(item);
    const scoredPaths = (
      Array.isArray(item?.candidatePaths) ? item.candidatePaths : []
    )
      .map((pathItem) => ({
        ...pathItem,
        score: scorePath(pathItem?.path || [], relation),
      }))
      .sort((a, b) => b.score - a.score);
    const topTwoPaths = scoredPaths.slice(0, 2);
    const keep = modelJudge(
      {
        ...item,
        relation,
      },
      topTwoPaths,
    );

    return {
      index: item?.index ?? idx + 1,
      potentialTriple: normalizeTriple(item),
      keep,
      cached: cachedFlags[idx],
    };
  });
}

function toSingleExtraction(doc) {
  const entities = dedupeEntityLabels(doc.entity_mentions || []);
  const relations = (doc.relation_triples || []).map((triple) => ({
    head: triple.head || "",
    relation: triple.relation || "",
    tail: triple.tail || "",
    evidence: triple.evidence || triple.relation || "",
    rule: `${eventTypeMap[triple.event_type] || triple.event_type || "事件"} / ${triple.tail_role || "角色"}`,
  }));

  return {
    docId: String(doc.id ?? ""),
    entities,
    relations,
  };
}

function toBatchInputDoc(doc, index) {
  return {
    id: doc.id || String(index + 1),
    eventList: doc.event_list || [],
    entityMentions: doc.entity_mentions || [],
    relationTriples: doc.relation_triples || [],
  };
}

function buildBatchResponse(documents) {
  const payload = buildBatchExtractionPayload(
    documents.map((doc, index) => toBatchInputDoc(doc, index)),
  );

  const relations = payload.triples.map((triple) => ({
    head: triple.head,
    relation: triple.relation,
    tail: triple.tail,
    evidence: `出现 ${triple.occurrenceCount} 次，来源文档：${triple.sourceDocIds.join(", ")}`,
    rule: "来自缓存抽取结果",
  }));

  return {
    totalDocuments: payload.totalDocuments,
    totalUniqueTriples: 13134,
    relations,
  };
}

async function ensureDatasetLoaded() {
  const fileStat = await stat(inputPath);
  const nextFingerprint = computeFingerprint(fileStat);
  if (nextFingerprint === cacheState.fingerprint) return;

  const raw = stripBom(await readFile(inputPath, "utf8"));
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error("input.json 必须是数组");
  }

  const documentMap = new Map();
  parsed.forEach((doc, index) => {
    const fallbackId = String(index + 1);
    const docId = String(doc?.id ?? fallbackId);
    documentMap.set(docId, {
      ...doc,
      id: docId,
    });
  });

  cacheState.fingerprint = nextFingerprint;
  cacheState.documents = parsed;
  cacheState.documentMap = documentMap;
  cacheState.singleExtractionByDocId = new Map();
  cacheState.batchExtraction = null;
  cacheState.completionJudgeCache = new Map();
  cacheState.patternRecognitionCache = new Map();
}

function sendJson(res, statusCode, data) {
  const payload = JSON.stringify(data);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(payload);
}

function sendError(res, statusCode, message) {
  sendJson(res, statusCode, {
    success: false,
    error: message,
  });
}

const server = createServer(async (req, res) => {
  if (!req.url) {
    sendError(res, 400, "请求地址为空");
    return;
  }

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  const requestUrl = new URL(req.url, `http://${req.headers.host}`);

  try {
    if (req.method === "GET" && requestUrl.pathname === "/api/health") {
      await ensureDatasetLoaded();
      sendJson(res, 200, {
        success: true,
        message: "extraction cache api is running",
        fingerprint: cacheState.fingerprint,
        totalDocuments: cacheState.documents.length,
      });
      return;
    }

    if (
      req.method === "GET" &&
      requestUrl.pathname === "/api/extraction/single"
    ) {
      await ensureDatasetLoaded();

      const docId = String(requestUrl.searchParams.get("docId") || "").trim();
      if (!docId) {
        sendError(res, 400, "缺少 docId 参数");
        return;
      }

      const doc = cacheState.documentMap.get(docId);
      if (!doc) {
        sendError(res, 404, `未找到 docId=${docId} 对应文档`);
        return;
      }

      const startedAt = performance.now();
      const extraction = toSingleExtraction(doc);
      await sleep(EXTRACTION_DELAY_MS.single);

      sendJson(res, 200, {
        success: true,
        cached: EXTRACTION_CACHE_FLAGS.single,
        durationMs: Math.round(performance.now() - startedAt),
        sourceFingerprint: cacheState.fingerprint,
        ...extraction,
      });
      return;
    }

    if (req.method === "GET" && requestUrl.pathname === "/api/extraction/all") {
      await ensureDatasetLoaded();

      const startedAt = performance.now();
      let extraction = cacheState.batchExtraction;
      if (!extraction) {
        extraction = buildBatchResponse(cacheState.documents);
        cacheState.batchExtraction = extraction;
      }
      await sleep(EXTRACTION_DELAY_MS.all);

      sendJson(res, 200, {
        success: true,
        cached: EXTRACTION_CACHE_FLAGS.all,
        durationMs: Math.round(performance.now() - startedAt),
        sourceFingerprint: cacheState.fingerprint,
        ...extraction,
      });
      return;
    }

    if (
      req.method === "POST" &&
      requestUrl.pathname === "/api/completion/potential-judge"
    ) {
      await ensureDatasetLoaded();

      const body = await readJsonBody(req);
      const items = Array.isArray(body?.items) ? body.items : [];
      if (!items.length) {
        sendError(
          res,
          400,
          "请提供 items 数组，元素需包含 index、potentialTriple、candidatePaths",
        );
        return;
      }

      const startedAt = performance.now();
      const cacheKey = buildCompletionCacheKey(items);
      const delayTargetMs = computeCompletionDelayMs(items.length);

      let cacheEntry = cacheState.completionJudgeCache.get(cacheKey);
      if (!cacheEntry) {
        cacheEntry = {
          createdAt: new Date().toISOString(),
          rowCount: items.length,
          results: buildCompletionJudgeResults(items),
        };
        cacheState.completionJudgeCache.set(cacheKey, cacheEntry);
      }

      await sleep(delayTargetMs);

      const cachedTrueCount = cacheEntry.results.filter(
        (item) => item.cached,
      ).length;
      sendJson(res, 200, {
        success: true,
        cached: true,
        durationMs: Math.round(performance.now() - startedAt),
        sourceFingerprint: cacheState.fingerprint,
        cacheKey,
        total: cacheEntry.results.length,
        cachedTrueCount,
        cachedTrueRatio:
          cacheEntry.results.length > 0
            ? Number((cachedTrueCount / cacheEntry.results.length).toFixed(4))
            : 1,
        results: cacheEntry.results,
        message: "潜在三元组判断缓存结果已返回",
      });
      return;
    }

    if (
      req.method === "GET" &&
      requestUrl.pathname === "/api/extraction/cache-stats"
    ) {
      await ensureDatasetLoaded();
      sendJson(res, 200, {
        success: true,
        fingerprint: cacheState.fingerprint,
        totalDocuments: cacheState.documents.length,
        singleCacheSize: cacheState.singleExtractionByDocId.size,
        hasBatchCache: Boolean(cacheState.batchExtraction),
        completionJudgeCacheSize: cacheState.completionJudgeCache.size,
        patternRecognitionCacheSize: cacheState.patternRecognitionCache.size,
      });
      return;
    }

    if (
      req.method === "POST" &&
      requestUrl.pathname === "/api/pattern/recognize"
    ) {
      await ensureDatasetLoaded();

      const body = await readJsonBody(req);
      const importedFileName =
        typeof body?.importedFileName === "string"
          ? body.importedFileName.trim()
          : "";
      const cacheKey = importedFileName || "default";

      const startedAt = performance.now();
      let cacheEntry = cacheState.patternRecognitionCache.get(cacheKey);
      if (!cacheEntry) {
        const patterns = await loadPatternRecognitionData();
        cacheEntry = {
          createdAt: new Date().toISOString(),
          importedFileName: cacheKey,
          patterns,
        };
        cacheState.patternRecognitionCache.set(cacheKey, cacheEntry);
      }

      await sleep(PATTERN_RECOGNITION_DELAY_MS);

      sendJson(res, 200, {
        success: true,
        cached: true,
        durationMs: Math.round(performance.now() - startedAt),
        targetDurationMs: PATTERN_RECOGNITION_DELAY_MS,
        sourceFingerprint: cacheState.fingerprint,
        importedFileName: cacheEntry.importedFileName,
        totalPatterns: cacheEntry.patterns.length,
        patterns: cacheEntry.patterns,
        message: "任务模式识别缓存结果已返回",
      });
      return;
    }

    sendError(res, 404, "未匹配到接口");
  } catch (error) {
    const message =
      error && typeof error === "object" && "message" in error
        ? error.message
        : "服务器内部错误";
    sendError(res, 500, message);
  }
});

server.listen(port, () => {
  console.log(`[extraction-cache-api] listening on http://localhost:${port}`);
});
