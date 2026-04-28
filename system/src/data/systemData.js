import rawDocuments from "../../input.json";

const keywordLexicon = [
  "\u7f8e\u56fd",
  "\u4fc4\u7f57\u65af",
  "\u5317\u7ea6",
  "\u6b27\u6d32",
  "\u5bfc\u5f39",
  "\u90e8\u7f72",
  "\u8bd5\u9a8c",
  "\u5c55\u51fa",
  "\u9632\u5fa1\u7cfb\u7edf",
  "\u88c5\u5907",
  "\u4efb\u52a1",
  "\u76ee\u6807",
];

const taskStages = [
  "\u77e5\u8bc6\u62bd\u53d6",
  "\u56fe\u8c31\u6784\u5efa",
  "\u56fe\u8c31\u8865\u5168",
  "\u6a21\u5f0f\u8bc6\u522b",
];

const eventTypeMap = {
  Deploy: "\u90e8\u7f72\u4e8b\u4ef6",
  Exhibit: "\u5c55\u51fa\u4e8b\u4ef6",
  Experiment: "\u8bd5\u9a8c\u4e8b\u4ef6",
  Support: "\u652f\u6301\u4e8b\u4ef6",
  Attack: "\u6253\u51fb\u4e8b\u4ef6",
  Transport: "\u8fd0\u8f93\u4e8b\u4ef6",
};

function clampText(text, size = 140) {
  if (!text) return "";
  if (text.length <= size) return text;
  return `${text.slice(0, size)}...`;
}

function detectStageTags(text) {
  const tags = [];
  if (/\u5bfc\u5f39|\u90e8\u7f72|\u53d1\u5c04|\u8bd5\u9a8c|\u5c55\u51fa/.test(text)) {
    tags.push(taskStages[0]);
  }
  if (/\u7f8e\u56fd|\u4fc4\u7f57\u65af|\u5317\u7ea6|\u516c\u53f8|\u5e73\u53f0|\u7cfb\u7edf/.test(text)) {
    tags.push(taskStages[1]);
  }
  if (/\u9884\u8ba1|\u53ef\u80fd|\u5b8c\u6210|\u63d0\u5347|\u8ba1\u5212/.test(text)) {
    tags.push(taskStages[2]);
  }
  if (/\u5a01\u80c1|\u5b89\u5168|\u6a21\u5f0f|\u4f53\u7cfb|\u80fd\u529b/.test(text)) {
    tags.push(taskStages[3]);
  }
  return tags.length ? tags : [taskStages[0]];
}

function countKeywordHits(items, keywords) {
  return keywords
    .map((keyword) => ({
      keyword,
      count: items.reduce((total, item) => total + (item.text.includes(keyword) ? 1 : 0), 0),
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);
}

function dedupeEntities(mentions = []) {
  const seen = new Set();
  return mentions.filter((mention) => {
    const key = `${mention.text}-${mention.role}-${mention.start}-${mention.end}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function dedupeEntityLabels(mentions = []) {
  const seen = new Set();
  return mentions.filter((mention) => {
    const key = `${mention.text}-${mention.role}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

const documents = rawDocuments.map((item, index) => {
  const mentions = dedupeEntities(item.entity_mentions || []);
  return {
    id: item.id || String(index + 1),
    originalText: item.text || "",
    markedText: item.marked_text || item.text || "",
    preview: clampText(item.text || "", 108),
    markedPreview: clampText(item.marked_text || item.text || "", 116),
    length: (item.text || "").length,
    tags: detectStageTags(item.text || ""),
    entities: dedupeEntityLabels(mentions),
    entityCount: mentions.length,
    eventList: item.event_list || [],
    relationTriples: item.relation_triples || [],
  };
});

const extractionCases = documents.map((item) => ({
  id: item.id,
  title: `\u6837\u672c\u6587\u6863 #${item.id}`,
  text: item.originalText,
  markedText: item.markedText,
  entities: item.entities,
  relations: item.relationTriples.map((triple) => ({
    head: triple.head,
    relation: triple.relation,
    tail: triple.tail,
    evidence: triple.evidence || triple.relation,
    rule: `${eventTypeMap[triple.event_type] || triple.event_type || "\u4e8b\u4ef6"} / ${triple.tail_role || "\u89d2\u8272"}`,
  })),
}));

const totalLength = documents.reduce((sum, item) => sum + item.length, 0);
const maxDocument = documents.reduce(
  (best, item) => (!best || item.length > best.length ? item : best),
  null,
);
const topKeywords = countKeywordHits(rawDocuments, keywordLexicon);
const highDensityDocuments = documents.filter((item) => item.length >= 220).length;
const taggedDocuments = documents.filter((item) => item.entityCount > 0).length;

const chapterMapping = [
  {
    key: "extract",
    title: "\u6587\u6863\u7ea7\u77e5\u8bc6\u62bd\u53d6",
    thesisAnchor: "\u7b2c\u4e00\u7ae0",
    summary:
      "\u4f9d\u636e test.json \u4e2d event_list \u63d0\u4f9b\u7684\u5b9e\u4f53\u4e0e\u4e8b\u4ef6\u8bba\u5143\u4fe1\u606f\uff0c\u5728\u539f\u6587\u4e2d\u5b9a\u4f4d\u771f\u5b9e\u5b9e\u4f53\u8fb9\u754c\u3002",
  },
  {
    key: "graph",
    title: "\u4efb\u52a1\u77e5\u8bc6\u56fe\u8c31\u6784\u5efa",
    thesisAnchor: "\u7b2c\u4e8c\u7ae0",
    summary:
      "\u5c06 event_list \u4e2d\u7684 Subject\u3001Equipment\u3001Militaryforce\u3001Location \u7b49\u8bba\u5143\u7ec4\u7ec7\u4e3a\u7ed3\u6784\u5316\u8282\u70b9\u4e0e\u8fde\u8fb9\u3002",
  },
  {
    key: "completion",
    title: "\u77e5\u8bc6\u56fe\u8c31\u8865\u5168",
    thesisAnchor: "\u7b2c\u4e09\u7ae0",
    summary:
      "\u57fa\u4e8e\u4e8b\u4ef6\u89e6\u53d1\u8bcd\u3001\u8bba\u5143\u89d2\u8272\u548c\u6587\u6863\u8bc1\u636e\u7ee7\u7eed\u6269\u5c55\u5019\u9009\u5173\u7cfb\u3002",
  },
  {
    key: "pattern",
    title: "\u6a21\u5f0f\u8bc6\u522b",
    thesisAnchor: "\u7b2c\u56db\u7ae0",
    summary:
      "\u5728\u771f\u5b9e\u4e8b\u4ef6\u89e6\u53d1\u548c\u8bba\u5143\u5173\u7cfb\u7684\u57fa\u7840\u4e0a\uff0c\u8bc6\u522b\u90e8\u7f72\u3001\u8bd5\u9a8c\u3001\u5c55\u51fa\u7b49\u6a21\u5f0f\u3002",
  },
];

const overviewMetrics = [
  {
    label: "\u6587\u6863\u6837\u672c\u603b\u6570",
    value: documents.length.toLocaleString("zh-CN"),
    note: "\u76f4\u63a5\u8bfb\u53d6 input.json",
  },
  {
    label: "\u5e73\u5747\u6587\u672c\u957f\u5ea6",
    value: `${Math.round(totalLength / Math.max(documents.length, 1))}`,
    note: "\u6309\u5b57\u7b26\u7edf\u8ba1",
  },
  {
    label: "\u5df2\u6807\u8bb0\u6587\u6863\u6570",
    value: `${taggedDocuments}`,
    note: "\u5b9e\u4f53\u8fb9\u754c\u5df2\u5199\u56de input.json",
  },
  {
    label: "\u9ad8\u4fe1\u606f\u5bc6\u5ea6\u6587\u6863",
    value: `${highDensityDocuments}`,
    note: maxDocument ? `\u6700\u957f\u6587\u6863 #${maxDocument.id}` : "\u6682\u65e0\u6570\u636e",
  },
];

const graphNodes = [
  { id: "\u7f8e\u56fd", type: "\u56fd\u5bb6", weight: 92 },
  { id: "\u4fc4\u7f57\u65af", type: "\u56fd\u5bb6", weight: 88 },
  { id: "\u5317\u7ea6", type: "\u7ec4\u7ec7", weight: 72 },
  { id: "\u6c11\u5175III", type: "\u88c5\u5907", weight: 66 },
  { id: "\u5bfc\u5f39\u9632\u5fa1\u7cfb\u7edf", type: "\u7cfb\u7edf", weight: 84 },
  { id: "\u874e\u5b50\u81ea\u52a8\u8feb\u51fb\u70ae\u6b66\u5668\u5e73\u53f0", type: "\u88c5\u5907", weight: 58 },
  { id: "\u9ad8\u673a\u52a8\u591a\u7528\u9014\u8f66", type: "\u5e73\u53f0", weight: 51 },
  { id: "\u6b27\u6d32", type: "\u533a\u57df", weight: 49 },
];

const graphLinks = [
  { source: "\u7f8e\u56fd", target: "\u6c11\u5175III", label: "\u90e8\u7f72" },
  { source: "\u7f8e\u56fd", target: "\u5bfc\u5f39\u9632\u5fa1\u7cfb\u7edf", label: "\u5efa\u8bbe" },
  { source: "\u4fc4\u7f57\u65af", target: "\u5bfc\u5f39\u9632\u5fa1\u7cfb\u7edf", label: "\u5a01\u80c1\u8bc4\u4f30" },
  { source: "\u5317\u7ea6", target: "\u6b27\u6d32", label: "\u90e8\u7f72\u533a\u57df" },
  { source: "\u874e\u5b50\u81ea\u52a8\u8feb\u51fb\u70ae\u6b66\u5668\u5e73\u53f0", target: "\u9ad8\u673a\u52a8\u591a\u7528\u9014\u8f66", label: "\u5e73\u53f0\u9002\u914d" },
  { source: "\u7f8e\u56fd", target: "\u6b27\u6d32", label: "\u884c\u52a8\u5f71\u54cd" },
];

const completionCandidates = [
  {
    triple: "\u7f8e\u56fd - \u90e8\u7f72 - B-2",
    score: 0.93,
    basis: "\u4e8b\u4ef6\u8bba\u5143",
    path: "\u7f8e\u56fd -> Deploy -> B-2",
    explanation: "\u57fa\u4e8e event_list \u4e2d\u7684 Subject \u4e0e Militaryforce \u771f\u5b9e\u8bba\u5143\u751f\u6210\u3002",
  },
  {
    triple: "\u7f8e\u56fd - \u90e8\u7f72 - \u5bfc\u5f39\u9632\u5fa1\u7cfb\u7edf",
    score: 0.89,
    basis: "\u89e6\u53d1\u8bcd\u89c4\u5219",
    path: "\u7f8e\u56fd -> Deploy -> \u5bfc\u5f39\u9632\u5fa1\u7cfb\u7edf",
    explanation: "\u5728\u6d4b\u8bd5\u6570\u636e\u4e2d\u7531 Deploy \u4e8b\u4ef6\u548c Militaryforce \u8bba\u5143\u5171\u540c\u786e\u8ba4\u3002",
  },
  {
    triple: "\u5357\u975e\u6cf0\u52d2\u65af\u516c\u53f8 - \u5c55\u51fa - \u874e\u5b50\u81ea\u52a8\u8feb\u51fb\u70ae\u6b66\u5668\u5e73\u53f0",
    score: 0.87,
    basis: "\u4e8b\u4ef6\u8bc1\u636e",
    path: "\u5357\u975e\u6cf0\u52d2\u65af\u516c\u53f8 -> Exhibit -> \u874e\u5b50\u81ea\u52a8\u8feb\u51fb\u70ae\u6b66\u5668\u5e73\u53f0",
    explanation: "\u6765\u81ea Exhibit \u4e8b\u4ef6\u7684\u771f\u5b9e Subject \u4e0e Equipment \u8bba\u5143\u3002",
  },
  {
    triple: "\u4e8b\u4ef6\u89e6\u53d1\u8bcd - \u7ea6\u675f - \u5173\u7cfb\u4e09\u5143\u7ec4",
    score: 0.81,
    basis: "\u89c4\u5219\u7ea6\u675f",
    path: "trigger -> arguments -> triples",
    explanation: "\u5c06 event_list \u7684\u89e6\u53d1\u8bcd\u548c\u8bba\u5143\u89d2\u8272\u76f4\u63a5\u6620\u5c04\u4e3a\u7ed3\u6784\u5316\u4e09\u5143\u7ec4\u3002",
  },
];

const patternSignals = [
  {
    name: "\u90e8\u7f72\u6d3b\u52a8\u6a21\u5f0f",
    confidence: "0.91",
    description: "\u591a\u7bc7\u6587\u6863\u7684 event_list \u4e2d\u51fa\u73b0 Deploy \u4e8b\u4ef6\uff0c\u53ef\u7528\u4e8e\u7edf\u8ba1\u90e8\u7f72\u884c\u4e3a\u3002",
    evidence: `${rawDocuments.filter((item) => (item.event_list || []).some((event) => event.event_type === "Deploy")).length} \u7bc7\u6587\u6863\u89e6\u53d1`,
  },
  {
    name: "\u5c55\u793a\u4e0e\u8bd5\u9a8c\u6a21\u5f0f",
    confidence: "0.86",
    description: "\u4ece Exhibit \u4e0e Experiment \u4e8b\u4ef6\u4e2d\u53ef\u4ee5\u63d0\u70bc\u88c5\u5907\u5c55\u793a\u4e0e\u8bd5\u9a8c\u4f53\u7cfb\u3002",
    evidence: `${rawDocuments.filter((item) => (item.event_list || []).some((event) => event.event_type === "Exhibit" || event.event_type === "Experiment")).length} \u7bc7\u6587\u6863\u89e6\u53d1`,
  },
  {
    name: "\u771f\u5b9e\u8bba\u5143\u6807\u6ce8\u6a21\u5f0f",
    confidence: "0.79",
    description: "\u5229\u7528 event_list \u8bba\u5143\u504f\u79fb\u5bf9\u6587\u6863\u8fdb\u884c <e>...</e> \u7ea7\u5b9e\u4f53\u8fb9\u754c\u6807\u8bb0\u3002",
    evidence: `${rawDocuments.filter((item) => (item.entity_mentions || []).length > 0).length} \u7bc7\u6587\u6863\u89e6\u53d1`,
  },
];

const architectureSteps = [
  {
    step: "01",
    title: "\u6570\u636e\u5bf9\u9f50\u4e0e\u56de\u5199",
    detail: "\u5c06 test.json \u4e2d event_list \u6309 id \u5bf9\u9f50\u5230 input.json \u5bf9\u5e94\u6587\u6863\u3002",
  },
  {
    step: "02",
    title: "\u5b9e\u4f53\u8fb9\u754c\u6807\u8bb0",
    detail: "\u4f9d\u636e arguments.offset \u5728\u539f\u6587\u4e2d\u52a0\u5165 <e> \u548c <\\e> \u6807\u8bc6\u3002",
  },
  {
    step: "03",
    title: "\u5019\u9009\u5b9e\u4f53\u6c47\u805a",
    detail: "\u5c06 Subject\u3001Equipment\u3001Militaryforce\u3001Location \u7b49\u8bba\u5143\u805a\u5408\u4e3a\u5019\u9009\u5b9e\u4f53\u5217\u8868\u3002",
  },
  {
    step: "04",
    title: "\u4e8b\u4ef6\u4e09\u5143\u7ec4\u751f\u6210",
    detail: "\u7528 trigger.text \u4f5c\u4e3a relation\uff0c\u5c06 Subject \u4e0e\u5176\u4ed6\u8bba\u5143\u7ec4\u5408\u6210\u4e09\u5143\u7ec4\u3002",
  },
  {
    step: "05",
    title: "\u524d\u7aef\u5206\u9875\u4e0e\u5c55\u793a",
    detail: "\u5728\u6587\u6863\u6837\u672c\u6c60\u4e2d\u5206\u9875\u5c55\u793a\uff0c\u5e76\u5728\u62bd\u53d6\u7ed3\u679c\u533a\u5448\u73b0\u771f\u5b9e\u6807\u8bb0\u4e0e\u4e09\u5143\u7ec4\u3002",
  },
];

export {
  architectureSteps,
  chapterMapping,
  completionCandidates,
  documents,
  extractionCases,
  graphLinks,
  graphNodes,
  overviewMetrics,
  patternSignals,
  taskStages,
  topKeywords,
};
