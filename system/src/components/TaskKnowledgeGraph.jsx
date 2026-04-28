import { useEffect, useMemo, useRef } from "react";
import { DataSet, Network } from "vis-network/standalone/esm/vis-network";
import "vis-network/styles/vis-network.css";
import allDocumentTriples from "../data/allDocumentTriples.json";
import globalGraphData from "../data/globalGraphData.json";

const ROLE_CONFIG = {
  Subject: { label: "主体", color: "#5B8FF9" },
  Location: { label: "地点", color: "#61DDAA" },
  Equipment: { label: "装备", color: "#7262FD" },
  Date: { label: "时间", color: "#F6BD16" },
  Militaryforce: { label: "军力", color: "#F08BB4" },
  Area: { label: "区域", color: "#8B95A7" },
  Content: { label: "内容", color: "#36B37E" },
  Object: { label: "对象", color: "#78D3F8" },
  Materials: { label: "材料", color: "#FF9D4D" },
  Participant: { label: "参与方", color: "#9270CA" },
  Unknown: { label: "其他", color: "#BFBFBF" },
};

const ROLE_ORDER = [
  "Militaryforce",
  "Subject",
  "Location",
  "Equipment",
  "Object",
  "Materials",
  "Participant",
  "Unknown",
];

const HIDDEN_ROLES = new Set(["Date"]);

function parseRoleFromTitle(title = "") {
  const matched = title.match(/Entity role:\s*([^<\n]+)/i);
  return matched?.[1] || "Unknown";
}

function normalizeRole(role = "Unknown") {
  if (role === "Area" || role === "Content") {
    return "Location";
  }

  return role;
}

function getRoleColor(role) {
  return ROLE_CONFIG[role]?.color || ROLE_CONFIG.Unknown.color;
}

function getRoleLabel(role) {
  return ROLE_CONFIG[role]?.label || role;
}

function getNodeDisplayName(node) {
  if (typeof node?.label === "string" && node.label.trim()) {
    return node.label.trim();
  }

  if (typeof node?.id === "string") {
    const matched = node.id.match(/::(.+)$/);
    if (matched?.[1]) {
      return matched[1].trim();
    }
    return node.id.trim();
  }

  return "";
}

function buildFilteredGraphData() {
  const rawNodes = globalGraphData.nodesData || [];
  const rawEdges = globalGraphData.edgesData || [];

  const nodes = rawNodes
    .filter(
      (node) =>
        !HIDDEN_ROLES.has(normalizeRole(parseRoleFromTitle(node.title))),
    )
    .map((node) => ({
      ...node,
      title: getNodeDisplayName(node),
      color: getRoleColor(normalizeRole(parseRoleFromTitle(node.title))),
      value: node.size || 20,
      size: Math.max(18, Math.min((node.size || 20) * 1.15, 42)),
      shape: "dot",
      borderWidth: 1,
      borderWidthSelected: 2,
      font: {
        size: 18,
        color: "#111827",
        face: "Microsoft YaHei",
      },
    }));

  const visibleNodeIds = new Set(nodes.map((node) => node.id));

  const edges = rawEdges
    .filter(
      (edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to),
    )
    .map((edge) => ({
      ...edge,
      label: edge.label || edge.title || "",
      arrows: {
        to: {
          enabled: false,
        },
      },
      smooth: false,
      width: Math.max(1.6, Math.min(edge.width || 1.8, 3.6)),
      color: {
        color: "#a6b4c8",
        highlight: "#2563eb",
        hover: "#2563eb",
      },
      font: {
        size: 16,
        align: "middle",
        strokeWidth: 0,
        background: "rgba(255,255,255,0.9)",
        color: "#334155",
        face: "Microsoft YaHei",
      },
    }));

  return { nodes, edges };
}

function buildStatsFromTriples() {
  const nodeSet = new Set();
  let edgeCount = 0;

  allDocumentTriples.triples.forEach((triple) => {
    const rawHeadRole = triple.headRole || "Unknown";
    const rawTailRole = triple.tailRole || "Unknown";
    if (triple.head) {
      nodeSet.add(`${rawHeadRole}::${triple.head}`);
    }

    if (triple.tail) {
      nodeSet.add(`${rawTailRole}::${triple.tail}`);
    }

    edgeCount += 1;
  });

  return {
    nodeCount: nodeSet.size,
    edgeCount,
  };
}

function buildLegendFromTriples() {
  const roleSet = new Set();

  allDocumentTriples.triples.forEach((triple) => {
    const headRole = normalizeRole(triple.headRole || "Unknown");
    const tailRole = normalizeRole(triple.tailRole || "Unknown");

    if (!HIDDEN_ROLES.has(headRole)) {
      roleSet.add(headRole);
    }

    if (!HIDDEN_ROLES.has(tailRole)) {
      roleSet.add(tailRole);
    }
  });

  return ROLE_ORDER.filter((role) => roleSet.has(role)).map((role) => ({
    role,
    label: getRoleLabel(role),
    color: getRoleColor(role),
  }));
}

export default function TaskKnowledgeGraph() {
  const containerRef = useRef(null);
  const networkRef = useRef(null);
  const { nodes, edges } = useMemo(() => buildFilteredGraphData(), []);
  const { nodeCount, edgeCount } = useMemo(() => buildStatsFromTriples(), []);
  const legend = useMemo(() => buildLegendFromTriples(), []);

  useEffect(() => {
    if (!containerRef.current) return undefined;

    const data = {
      nodes: new DataSet(nodes),
      edges: new DataSet(edges),
    };

    const options = {
      autoResize: true,
      interaction: {
        hover: true,
        navigationButtons: true,
        keyboard: true,
        dragNodes: true,
        dragView: true,
        zoomView: true,
      },
      layout: {
        improvedLayout: true,
      },
      physics: {
        enabled: true,
        stabilization: {
          enabled: true,
          iterations: 300,
          fit: true,
        },
        barnesHut: {
          gravitationalConstant: -3200,
          centralGravity: 0.2,
          springLength: 180,
          springConstant: 0.04,
          damping: 0.09,
          avoidOverlap: 1,
        },
        minVelocity: 0.75,
      },
      nodes: {
        shape: "dot",
        borderWidth: 1,
        borderWidthSelected: 2,
        shadow: {
          enabled: true,
          color: "rgba(37, 99, 235, 0.12)",
          size: 12,
          x: 0,
          y: 4,
        },
        font: {
          size: 16,
          color: "#111827",
          face: "Microsoft YaHei",
        },
      },
      edges: {
        smooth: false,
        arrows: {
          to: {
            enabled: false,
          },
        },
        color: {
          color: "#a6b4c8",
          highlight: "#2563eb",
          hover: "#2563eb",
        },
        width: 1.8,
        selectionWidth: 2.4,
        hoverWidth: 2.4,
        font: {
          size: 16,
          align: "middle",
          strokeWidth: 0,
          background: "rgba(255,255,255,0.9)",
          color: "#334155",
          face: "Microsoft YaHei",
        },
      },
    };

    const network = new Network(containerRef.current, data, options);
    networkRef.current = network;

    network.once("stabilizationIterationsDone", () => {
      network.moveTo({
        scale: 1.1,
        animation: {
          duration: 500,
          easingFunction: "easeInOutQuad",
        },
      });
    });

    return () => {
      network.destroy();
      networkRef.current = null;
    };
  }, [nodes, edges]);

  return (
    <article className="panel wide graph-overview-panel graph-reference-panel">
      <div className="graph-reference-header">
        <div>
          <div className="graph-reference-title">任务模式图谱</div>
          <div className="graph-reference-meta">
            {`节点 ${nodeCount} | 边 ${edgeCount}`}
          </div>
        </div>
      </div>

      <div className="graph-reference-legend">
        {legend.map((item) => (
          <span className="graph-reference-legend-item" key={item.role}>
            <span
              className="graph-reference-legend-dot"
              style={{ backgroundColor: item.color }}
            />
            {item.label}
          </span>
        ))}
      </div>

      <div className="graph-reference-shell">
        <div
          ref={containerRef}
          className="graph-vis-container"
          aria-label="任务知识图谱"
        />
      </div>
    </article>
  );
}
