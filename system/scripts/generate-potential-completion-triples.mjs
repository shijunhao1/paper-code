import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TARGET_COUNT = 2000;
const MAX_PATH_HOPS = 3;
const PATH_LIMIT = 5;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const sourcePath = path.join(rootDir, "src", "data", "allDocumentTriples.json");
const outputPath = path.join(
  rootDir,
  "src",
  "data",
  "potentialCompletionTriples.json",
);

function tripleKey(head, relation, tail) {
  return `${head}||${relation}||${tail}`;
}

function pickAlternative(entities, original, seed) {
  const candidates = entities.filter((entity) => entity !== original);
  if (!candidates.length) {
    return null;
  }

  return candidates[seed % candidates.length];
}

function buildAdjacency(triples) {
  const edgeMap = new Map();
  for (const triple of triples) {
    const key = tripleKey(triple.head, triple.relation, triple.tail);
    const existing = edgeMap.get(key);
    const occurrence = Number(triple.occurrenceCount) || 1;

    if (!existing) {
      edgeMap.set(key, {
        from: triple.head,
        relation: triple.relation,
        to: triple.tail,
        occurrenceCount: occurrence,
      });
      continue;
    }

    existing.occurrenceCount += occurrence;
  }

  const adjacency = new Map();
  for (const edge of edgeMap.values()) {
    if (!adjacency.has(edge.from)) {
      adjacency.set(edge.from, []);
    }

    adjacency.get(edge.from).push(edge);
  }

  return { adjacency, edges: Array.from(edgeMap.values()) };
}

function findCandidatePaths(adjacency, start, target, maxHops = 3, limit = 5) {
  if (!start || !target || start === target) {
    return [];
  }

  const results = [];
  const visitedNodes = new Set([start]);

  function dfs(node, path) {
    if (results.length >= limit || path.length >= maxHops) {
      return;
    }

    const outgoing = adjacency.get(node) || [];
    for (const edge of outgoing) {
      if (results.length >= limit) {
        break;
      }

      if (visitedNodes.has(edge.to)) {
        continue;
      }

      const nextPath = [...path, edge];
      if (edge.to === target) {
        results.push(nextPath);
        continue;
      }

      visitedNodes.add(edge.to);
      dfs(edge.to, nextPath);
      visitedNodes.delete(edge.to);
    }
  }

  dfs(start, []);
  return results;
}

function formatPath(path) {
  const parts = [path[0].from];
  for (const edge of path) {
    parts.push(`-(${edge.relation})->`);
    parts.push(edge.to);
  }

  return parts.join(" ");
}

function randomPick(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function isPathExactlySameAsTriple(path, triple) {
  if (!Array.isArray(path) || path.length !== 1) {
    return false;
  }

  const edge = path[0];
  return (
    edge.from === triple.head &&
    edge.relation === triple.relation &&
    edge.to === triple.tail
  );
}

function buildRandomFallbackPath(allEdges, triple) {
  const fallbackHops = Math.floor(Math.random() * 3) + 1;
  const path = [];

  if (!allEdges.length) {
    const safeRelation =
      triple.relation === "related_to" ? "related_to_fallback" : "related_to";
    path.push({
      from: triple.head,
      relation: safeRelation,
      to: triple.tail,
      occurrenceCount: 1,
    });
    return path;
  }

  const firstBase = randomPick(allEdges);
  path.push({
    from: triple.head,
    relation: firstBase.relation,
    to: firstBase.to,
    occurrenceCount: firstBase.occurrenceCount,
  });

  let currentNode = firstBase.to;
  for (let i = 1; i < fallbackHops; i += 1) {
    const base = randomPick(allEdges);
    const isLast = i === fallbackHops - 1;
    path.push({
      from: currentNode,
      relation: base.relation,
      to: isLast ? triple.tail : base.to,
      occurrenceCount: base.occurrenceCount,
    });
    currentNode = isLast ? triple.tail : base.to;
  }

  if (fallbackHops === 1) {
    path[0].to = triple.tail;
  }

  return path;
}

function buildNonDuplicateFallbackPath(allEdges, triple) {
  const maxAttempts = 40;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const candidate = buildRandomFallbackPath(allEdges, triple);
    if (!isPathExactlySameAsTriple(candidate, triple)) {
      return candidate;
    }
  }

  return [
    {
      from: triple.head,
      relation:
        triple.relation === "fallback_bridge"
          ? "fallback_bridge_2"
          : "fallback_bridge",
      to: triple.tail,
      occurrenceCount: 1,
    },
  ];
}

const raw = await fs.readFile(sourcePath, "utf8");
const payload = JSON.parse(raw);
const triples = Array.isArray(payload.triples) ? payload.triples : [];

const roleEntityMap = new Map();
for (const triple of triples) {
  if (triple.headRole && triple.head) {
    if (!roleEntityMap.has(triple.headRole)) {
      roleEntityMap.set(triple.headRole, new Set());
    }

    roleEntityMap.get(triple.headRole).add(triple.head);
  }

  if (triple.tailRole && triple.tail) {
    if (!roleEntityMap.has(triple.tailRole)) {
      roleEntityMap.set(triple.tailRole, new Set());
    }

    roleEntityMap.get(triple.tailRole).add(triple.tail);
  }
}

const roleEntities = new Map(
  Array.from(roleEntityMap.entries()).map(([role, entities]) => [
    role,
    Array.from(entities).sort((a, b) => a.localeCompare(b, "zh-CN")),
  ]),
);
const { adjacency, edges: allEdges } = buildAdjacency(triples);

const existingTriples = new Set(
  triples.map((item) => tripleKey(item.head, item.relation, item.tail)),
);
const generatedKeys = new Set();
const potentialTriples = [];

let round = 0;
while (potentialTriples.length < TARGET_COUNT && round < 30) {
  let addedInRound = 0;

  for (let index = 0; index < triples.length; index += 1) {
    if (potentialTriples.length >= TARGET_COUNT) {
      break;
    }

    const triple = triples[index];
    const headCandidates = roleEntities.get(triple.headRole) || [];
    const tailCandidates = roleEntities.get(triple.tailRole) || [];

    const replacementHead = pickAlternative(
      headCandidates,
      triple.head,
      index + round * 5,
    );
    if (replacementHead) {
      const key = tripleKey(replacementHead, triple.relation, triple.tail);
      if (!existingTriples.has(key) && !generatedKeys.has(key)) {
        potentialTriples.push({
          id: potentialTriples.length + 1,
          head: replacementHead,
          headRole: triple.headRole,
          relation: triple.relation,
          tail: triple.tail,
          tailRole: triple.tailRole,
          generationMethod: "replace_head_same_role",
          sourceTripleIndex: index,
        });
        generatedKeys.add(key);
        addedInRound += 1;
      }
    }

    if (potentialTriples.length >= TARGET_COUNT) {
      break;
    }

    const replacementTail = pickAlternative(
      tailCandidates,
      triple.tail,
      index * 3 + round * 7,
    );
    if (replacementTail) {
      const key = tripleKey(triple.head, triple.relation, replacementTail);
      if (!existingTriples.has(key) && !generatedKeys.has(key)) {
        potentialTriples.push({
          id: potentialTriples.length + 1,
          head: triple.head,
          headRole: triple.headRole,
          relation: triple.relation,
          tail: replacementTail,
          tailRole: triple.tailRole,
          generationMethod: "replace_tail_same_role",
          sourceTripleIndex: index,
        });
        generatedKeys.add(key);
        addedInRound += 1;
      }
    }
  }

  if (!addedInRound) {
    break;
  }

  round += 1;
}

const potentialTriplesWithPaths = potentialTriples.map((item) => {
  const discoveredPaths = findCandidatePaths(
    adjacency,
    item.head,
    item.tail,
    MAX_PATH_HOPS,
    PATH_LIMIT,
  );

  const candidatePaths = discoveredPaths.map((path, index) => ({
    id: `${item.id}-${index + 1}`,
    hops: path.length,
    path,
    pathText: formatPath(path),
    source: "graph_search",
  }));

  if (candidatePaths.length) {
    return {
      ...item,
      candidatePaths,
    };
  }

  const fallbackPath = buildNonDuplicateFallbackPath(allEdges, item);
  return {
    ...item,
    candidatePaths: [
      {
        id: `${item.id}-fallback-1`,
        hops: fallbackPath.length,
        path: fallbackPath,
        pathText: formatPath(fallbackPath),
        source: "random_fallback",
      },
    ],
  };
});

const output = {
  generatedAt: new Date().toISOString(),
  sourceFile: "src/data/allDocumentTriples.json",
  targetCount: TARGET_COUNT,
  totalGenerated: potentialTriples.length,
  pathRule:
    "For each potential triple, candidate paths are precomputed by graph search up to 3 hops. If none exists, one random fallback path is generated.",
  generationRule:
    "Potential triples are generated by replacing head or tail entities with entities of the same role type.",
  potentialTriples: potentialTriplesWithPaths,
};

await fs.writeFile(outputPath, JSON.stringify(output, null, 2), "utf8");

console.log(`Generated ${potentialTriples.length} potential triples.`);
