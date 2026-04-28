function upsertRoleCount(roleMap, text, role) {
  if (!text || !role) return;
  const current = roleMap.get(text) || new Map();
  current.set(role, (current.get(role) || 0) + 1);
  roleMap.set(text, current);
}

function buildRoleLookup(doc = {}) {
  const roleMap = new Map();

  (doc.eventList || []).forEach((eventItem) => {
    (eventItem.arguments || []).forEach((argument) => {
      upsertRoleCount(roleMap, argument.text, argument.role);
    });
  });

  (doc.entityMentions || []).forEach((mention) => {
    upsertRoleCount(roleMap, mention.text, mention.role);
  });

  return roleMap;
}

function resolveEntityRole(roleLookup, text, fallbackRole = "") {
  if (fallbackRole) return fallbackRole;
  const roleStats = roleLookup.get(text);
  if (!roleStats) return "";

  return Array.from(roleStats.entries()).sort((left, right) => {
    if (right[1] !== left[1]) return right[1] - left[1];
    return left[0].localeCompare(right[0], "zh-CN");
  })[0][0];
}

function normalizeTriple(doc, triple, index) {
  const roleLookup = buildRoleLookup(doc);

  return {
    id: `${doc.id}-${index + 1}`,
    docId: doc.id,
    head: triple.head || "",
    headRole: resolveEntityRole(roleLookup, triple.head, triple.head_role || ""),
    relation: triple.relation || "",
    tail: triple.tail || "",
    tailRole: resolveEntityRole(roleLookup, triple.tail, triple.tail_role || ""),
    eventType: triple.event_type || "",
    evidence: triple.evidence || "",
  };
}

function buildAggregateKey(triple) {
  return [triple.head, triple.relation, triple.tail].join("::");
}

export function buildBatchExtractionPayload(documentItems = []) {
  const documentTriples = documentItems.map((doc) => {
    const triples = (doc.relationTriples || []).map((triple, index) =>
      normalizeTriple(doc, triple, index),
    );

    return {
      docId: doc.id,
      tripleCount: triples.length,
      triples,
    };
  });

  const aggregateMap = new Map();

  documentTriples.forEach((doc) => {
    doc.triples.forEach((triple) => {
      const key = buildAggregateKey(triple);
      if (!aggregateMap.has(key)) {
        aggregateMap.set(key, {
          head: triple.head,
          headRole: triple.headRole,
          relation: triple.relation,
          tail: triple.tail,
          tailRole: triple.tailRole,
          occurrenceCount: 0,
          sourceDocIds: [],
        });
      }

      const current = aggregateMap.get(key);
      current.occurrenceCount += 1;

      if (!current.sourceDocIds.includes(triple.docId)) {
        current.sourceDocIds.push(triple.docId);
      }
    });
  });

  const triples = Array.from(aggregateMap.values()).sort((left, right) => {
    if (right.occurrenceCount !== left.occurrenceCount) {
      return right.occurrenceCount - left.occurrenceCount;
    }
    return `${left.head}${left.relation}${left.tail}`.localeCompare(
      `${right.head}${right.relation}${right.tail}`,
      "zh-CN",
    );
  });

  const totalRawTriples = documentTriples.reduce(
    (sum, doc) => sum + doc.tripleCount,
    0,
  );

  return {
    generatedAt: new Date().toISOString(),
    totalDocuments: documentItems.length,
    documentsWithTriples: documentTriples.filter((doc) => doc.tripleCount > 0)
      .length,
    totalRawTriples,
    totalUniqueTriples: triples.length,
    triples,
    documentTriples,
  };
}

export function downloadBatchExtractionPayload(
  payload,
  fileName = "allDocumentTriples.json",
) {
  if (typeof window === "undefined" || typeof document === "undefined") return;

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
}
