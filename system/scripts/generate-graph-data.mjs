import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const sourcePath = path.join(rootDir, "src", "data", "allDocumentTriples.json");
const outputPath = path.join(rootDir, "src", "data", "taskKnowledgeGraph.json");

const ROLE_COLORS = {
  Subject: "#5B8FF9",
  Location: "#61DDAA",
  Date: "#F6BD16",
  Equipment: "#7262FD",
  Militaryforce: "#F08BB4",
  Object: "#78D3F8",
  Participant: "#FF9D4D",
  Unknown: "#BFBFBF",
};

const ROLE_ORDER = [
  "Subject",
  "Location",
  "Date",
  "Equipment",
  "Militaryforce",
  "Object",
  "Participant",
  "Unknown",
];

const ROLE_LABELS = {
  Subject: "Subject",
  Location: "Location",
  Date: "Date",
  Equipment: "Equipment",
  Militaryforce: "Militaryforce",
  Object: "Object",
  Participant: "Participant",
  Unknown: "Unknown",
};

const MIN_NODE_DEGREE = 20;
const MIN_NODE_MENTIONS = 30;
const MIN_EDGE_WEIGHT = 3;
const MAX_VISIBLE_NODES = 120;
const MAX_VISIBLE_EDGES = 120;
const GRAPH_WIDTH = 1800;
const GRAPH_HEIGHT = 980;

function normalizeRole(role = "") {
  const mapping = {
    Subject: "Subject",
    Country: "Subject",
    Organization: "Subject",
    Location: "Location",
    Area: "Location",
    Date: "Date",
    Equipment: "Equipment",
    Materials: "Equipment",
    Militaryforce: "Militaryforce",
    Object: "Object",
    Content: "Object",
    Result: "Object",
    Quantity: "Object",
    Participant: "Participant",
  };

  return mapping[role] || "Unknown";
}

function addRoleCount(roleMap, role) {
  roleMap.set(role, (roleMap.get(role) || 0) + 1);
}

function pickDominantRole(roleMap) {
  if (!roleMap.size) return "Unknown";
  return Array.from(roleMap.entries()).sort((left, right) => {
    if (right[1] !== left[1]) return right[1] - left[1];
    return ROLE_ORDER.indexOf(left[0]) - ROLE_ORDER.indexOf(right[0]);
  })[0][0];
}

function polarToCartesian(cx, cy, radius, angle) {
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  };
}

function buildNodeTitle(node) {
  return [
    node.label,
    `Role: ${node.role}`,
    `Doc freq: ${node.docFreq}`,
    `Mentions: ${node.mentions}`,
    `Doc IDs: ${node.docIds.join("|")}`,
  ].join("\n");
}

function buildEdgeTitle(edge) {
  return [
    `Relation: ${edge.label}`,
    `Weight: ${edge.weight}`,
    `Head: ${edge.fromLabel}`,
    `Tail: ${edge.toLabel}`,
    `Doc IDs: ${edge.docIds.join("|")}`,
  ].join("\n");
}

const raw = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const triples = raw.triples || [];
const nodeMap = new Map();

triples.forEach((triple) => {
  const headRole = normalizeRole(triple.headRole);
  const tailRole = normalizeRole(triple.tailRole);

  if (!nodeMap.has(triple.head)) {
    nodeMap.set(triple.head, {
      id: `ENTITY::${triple.head}`,
      label: triple.head,
      roleMap: new Map(),
      mentions: 0,
      degree: 0,
      docIds: new Set(),
    });
  }

  if (!nodeMap.has(triple.tail)) {
    nodeMap.set(triple.tail, {
      id: `ENTITY::${triple.tail}`,
      label: triple.tail,
      roleMap: new Map(),
      mentions: 0,
      degree: 0,
      docIds: new Set(),
    });
  }

  const headNode = nodeMap.get(triple.head);
  const tailNode = nodeMap.get(triple.tail);

  addRoleCount(headNode.roleMap, headRole);
  addRoleCount(tailNode.roleMap, tailRole);

  headNode.mentions += triple.occurrenceCount || 1;
  tailNode.mentions += triple.occurrenceCount || 1;
  headNode.degree += 1;
  tailNode.degree += 1;

  (triple.sourceDocIds || []).forEach((docId) => {
    headNode.docIds.add(docId);
    tailNode.docIds.add(docId);
  });
});

const allNodes = Array.from(nodeMap.values())
  .map((node) => {
    const role = pickDominantRole(node.roleMap);
    const docIds = Array.from(node.docIds).sort((left, right) => Number(left) - Number(right));
    const docFreq = docIds.length;

    return {
      id: node.id,
      entityId: node.label,
      label: node.label,
      role,
      color: ROLE_COLORS[role] || ROLE_COLORS.Unknown,
      size: 20 + Math.min(Math.log(node.mentions + 1) * 4.8, 20),
      mentions: node.mentions,
      degree: node.degree,
      docFreq,
      docIds,
    };
  })
  .sort((left, right) => {
    if (right.degree !== left.degree) return right.degree - left.degree;
    if (right.mentions !== left.mentions) return right.mentions - left.mentions;
    return left.label.localeCompare(right.label, "zh-CN");
  });

const visibleNodes = allNodes
  .filter(
    (node) => node.degree >= MIN_NODE_DEGREE && node.mentions >= MIN_NODE_MENTIONS,
  )
  .slice(0, MAX_VISIBLE_NODES)
  .map((node) => ({ ...node }));
const visibleNodeIds = new Set(visibleNodes.map((node) => node.entityId));

const visibleEdges = triples
  .filter(
    (triple) =>
      (triple.occurrenceCount || 1) >= MIN_EDGE_WEIGHT &&
      visibleNodeIds.has(triple.head) && visibleNodeIds.has(triple.tail),
  )
  .sort((left, right) => right.occurrenceCount - left.occurrenceCount)
  .slice(0, MAX_VISIBLE_EDGES)
  .map((triple, index) => ({
    id: `EDGE::${index + 1}`,
    from: `ENTITY::${triple.head}`,
    to: `ENTITY::${triple.tail}`,
    fromLabel: triple.head,
    toLabel: triple.tail,
    label: triple.relation,
    weight: triple.occurrenceCount || 1,
    width: 1 + Math.min(Math.log((triple.occurrenceCount || 1) + 1) * 0.75, 2.6),
    docIds: triple.sourceDocIds || [],
  }));

const groupedNodes = ROLE_ORDER.reduce((groups, role) => {
  groups[role] = visibleNodes.filter((node) => node.role === role);
  return groups;
}, {});

const centerX = GRAPH_WIDTH / 2;
const centerY = GRAPH_HEIGHT / 2;
const sectorAngle = (Math.PI * 2) / ROLE_ORDER.length;

ROLE_ORDER.forEach((role, roleIndex) => {
  const nodes = groupedNodes[role];
  if (!nodes.length) return;

  nodes.forEach((node, nodeIndex) => {
    const ratio = nodes.length === 1 ? 0.5 : nodeIndex / (nodes.length - 1);
    const ring = 140 + (nodeIndex % 6) * 72 + Math.min(node.degree * 2.4, 42);
    const angle =
      -Math.PI / 2 +
      roleIndex * sectorAngle +
      sectorAngle * 0.14 +
      ratio * sectorAngle * 0.72;
    const point = polarToCartesian(centerX, centerY, ring, angle);

    node.x = Number(point.x.toFixed(2));
    node.y = Number(point.y.toFixed(2));
    node.title = buildNodeTitle(node);
  });
});

const nodeTitleMap = new Map(visibleNodes.map((node) => [node.id, node.label]));
const edges = visibleEdges.map((edge) => ({
  ...edge,
  title: buildEdgeTitle(edge),
}));

const payload = {
  generatedAt: new Date().toISOString(),
  source: "src/data/allDocumentTriples.json",
  stats: {
    totalNodes: allNodes.length,
    totalEdges: triples.length,
    visibleNodes: visibleNodes.length,
    visibleEdges: edges.length,
  },
  legend: ROLE_ORDER.map((role) => ({
    role,
    label: ROLE_LABELS[role],
    color: ROLE_COLORS[role],
  })),
  nodes: visibleNodes.map((node) => ({
    ...node,
    title: buildNodeTitle(node),
  })),
  edges: edges.map((edge) => ({
    ...edge,
    fromLabel: nodeTitleMap.get(edge.from) || edge.fromLabel,
    toLabel: nodeTitleMap.get(edge.to) || edge.toLabel,
  })),
};

await fs.writeFile(outputPath, JSON.stringify(payload, null, 2), "utf8");

console.log(
  `Generated graph data with ${payload.stats.visibleNodes} visible nodes and ${payload.stats.visibleEdges} visible edges.`,
);
