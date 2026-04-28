import { useMemo, useState } from "react";
import potentialCompletionTriples from "../data/potentialCompletionTriples.json";
import "./KnowledgeCompletionModule.css";

const PAGE_SIZE = 20;
const SAMPLE_MIN = 900;
const SAMPLE_MAX = 1200;
const MAX_PATH_HOPS = 3;

const COMPLETION_API_BASE = (
  import.meta.env.VITE_EXTRACTION_API_BASE || "http://localhost:8787"
).replace(/\/$/, "");

function tripleText(triple) {
  return `${triple.head} - ${triple.relation} - ${triple.tail}`;
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function shuffle(items) {
  const copied = [...items];
  for (let i = copied.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copied[i], copied[j]] = [copied[j], copied[i]];
  }

  return copied;
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function scorePath(path, targetRelation) {
  const hops = Math.max(1, path.length);
  const hopScore = Math.max(0, (MAX_PATH_HOPS + 1 - hops) / MAX_PATH_HOPS);
  const relationMatchCount = path.filter(
    (edge) => edge.relation === targetRelation,
  ).length;
  const relationScore = relationMatchCount / hops;
  const supportScore =
    path.reduce(
      (sum, edge) => sum + Math.min(20, Number(edge.occurrenceCount) || 0) / 20,
      0,
    ) / hops;

  return 0.5 * hopScore + 0.3 * relationScore + 0.2 * supportScore;
}

function modelJudge(row, topTwoPaths) {
  if (!topTwoPaths.length) {
    return false;
  }

  const bestScore = topTwoPaths[0]?.score || 0;
  const avgScore =
    topTwoPaths.reduce((sum, item) => sum + item.score, 0) / topTwoPaths.length;
  const graphEvidenceRatio =
    topTwoPaths.filter((item) => item.source === "graph_search").length /
    topTwoPaths.length;
  const pathCountBonus = (row.candidatePaths?.length || 0) >= 2 ? 0.14 : 0.04;

  const llmScore =
    0.55 * bestScore +
    0.25 * avgScore +
    0.12 * graphEvidenceRatio +
    pathCountBonus;

  return llmScore >= 0.43;
}

async function requestCompletionJudgeCache(rows) {
  const items = rows.map((row, index) => ({
    index: row?.id ?? index + 1,
    potentialTriple: {
      head: row?.head || "",
      relation: row?.relation || "",
      tail: row?.tail || "",
    },
    candidatePaths: Array.isArray(row?.candidatePaths)
      ? row.candidatePaths
      : [],
  }));

  const response = await fetch(
    `${COMPLETION_API_BASE}/api/completion/potential-judge`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ items }),
    },
  );

  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok || !payload?.success) {
    throw new Error(payload?.error || `补全接口请求失败（${response.status}）`);
  }

  return payload;
}

export default function KnowledgeCompletionModule() {
  const rows = useMemo(() => {
    const allRows = potentialCompletionTriples.potentialTriples || [];
    if (!allRows.length) {
      return [];
    }

    const sampleMax = Math.min(SAMPLE_MAX, allRows.length);
    const sampleMin = Math.min(SAMPLE_MIN, sampleMax);
    const sampleSize = randomInt(sampleMin, sampleMax);

    const sampled = shuffle(allRows).slice(0, sampleSize);
    sampled.sort((a, b) => {
      const aGt2 = (a.candidatePaths?.length || 0) > 2 ? 1 : 0;
      const bGt2 = (b.candidatePaths?.length || 0) > 2 ? 1 : 0;
      return bGt2 - aGt2;
    });

    return sampled;
  }, []);

  const [page, setPage] = useState(1);
  const [decisionMap, setDecisionMap] = useState({});
  const [manualKeepMap, setManualKeepMap] = useState({});
  const [processState, setProcessState] = useState({
    status: "idle",
    progress: 0,
    stage: "等待开始",
  });
  const [graphUpdateState, setGraphUpdateState] = useState({
    status: "idle",
    progress: 0,
    addedCount: 0,
    stage: "尚未更新任务模式图谱",
  });

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageStartIndex = (safePage - 1) * PAGE_SIZE;
  const visibleRows = rows.slice(pageStartIndex, pageStartIndex + PAGE_SIZE);

  const globalTopTwoPaths = useMemo(() => {
    const allPaths = Object.entries(decisionMap).flatMap(([rowId, decision]) =>
      (decision.scoredPaths || []).map((pathItem) => ({
        ...pathItem,
        rowId,
      })),
    );

    return allPaths.sort((a, b) => b.score - a.score).slice(0, 2);
  }, [decisionMap]);

  const keepSummary = useMemo(() => {
    const decisions = Object.values(decisionMap);
    if (!decisions.length) {
      return { keep: 0, drop: 0 };
    }

    let keep = 0;
    let drop = 0;
    Object.entries(decisionMap).forEach(([rowId, decision]) => {
      const manual = manualKeepMap[rowId];
      const finalKeep = typeof manual === "boolean" ? manual : decision.keep;
      if (finalKeep) {
        keep += 1;
      } else {
        drop += 1;
      }
    });

    return { keep, drop };
  }, [decisionMap, manualKeepMap]);

  const canUpdateGraph =
    processState.status === "done" && Object.keys(decisionMap).length > 0;

  const handleJudgeAll = async () => {
    if (processState.status === "running") {
      return;
    }

    const scoredMap = {};
    const trimmedMap = {};
    const finalMap = {};

    const completionApiPromise = requestCompletionJudgeCache(rows).catch(
      (error) => {
        console.warn(error);
        return null;
      },
    );

    setManualKeepMap({});
    setGraphUpdateState({
      status: "idle",
      progress: 0,
      addedCount: 0,
      stage: "尚未更新任务模式图谱",
    });
    setProcessState({
      status: "running",
      progress: 0,
      stage: "路径评分中",
    });

    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      const scoredPaths = (row.candidatePaths || [])
        .map((pathItem) => ({
          ...pathItem,
          score: scorePath(pathItem.path || [], row.relation),
        }))
        .sort((a, b) => b.score - a.score);
      scoredMap[String(row.id)] = scoredPaths;

      if (i % 24 === 0 || i === rows.length - 1) {
        const progress = Math.min(60, Math.round(((i + 1) / rows.length) * 60));
        setProcessState({
          status: "running",
          progress,
          stage: "路径评分中",
        });
        await sleep(100);
      }
    }

    setProcessState({
      status: "running",
      progress: 60,
      stage: "裁剪每个三元组Top2路径",
    });

    for (let i = 0; i < rows.length; i += 1) {
      const scoredPaths = scoredMap[String(rows[i].id)] || [];
      trimmedMap[String(rows[i].id)] = scoredPaths.slice(0, 2);

      if (i % 30 === 0 || i === rows.length - 1) {
        const progress = 60 + Math.round(((i + 1) / rows.length) * 25);
        setProcessState({
          status: "running",
          progress: Math.min(85, progress),
          stage: "裁剪每个三元组Top2路径",
        });
        await sleep(20);
      }
    }

    setProcessState({
      status: "running",
      progress: 85,
      stage: "大模型判断保留与否",
    });

    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      const key = String(row.id);
      const topTwoPaths = trimmedMap[key] || [];
      finalMap[key] = {
        scoredPaths: scoredMap[key] || [],
        topTwoPaths,
        keep: modelJudge(row, topTwoPaths),
      };

      if (i % 30 === 0 || i === rows.length - 1) {
        const progress = 85 + Math.round(((i + 1) / rows.length) * 15);
        setProcessState({
          status: "running",
          progress: Math.min(100, progress),
          stage: "大模型判断保留与否",
        });
        await sleep(500);
      }
    }

    await completionApiPromise;

    setDecisionMap(finalMap);
    setProcessState({
      status: "done",
      progress: 100,
      stage: "潜在三元组判断完成",
    });
  };

  const handleUpdateGraph = async () => {
    if (!canUpdateGraph || graphUpdateState.status === "running") {
      return;
    }

    const addedCount = keepSummary.keep;
    setGraphUpdateState({
      status: "running",
      progress: 0,
      addedCount: 0,
      stage: "正在写入任务模式图谱",
    });

    const steps = 36;
    for (let i = 1; i <= steps; i += 1) {
      const progress = Math.round((i / steps) * 100);
      const currentAdded = Math.round((i / steps) * addedCount);
      setGraphUpdateState({
        status: "running",
        progress,
        addedCount: currentAdded,
        stage: "正在写入任务模式图谱",
      });
      await sleep(40);
    }

    setGraphUpdateState({
      status: "done",
      progress: 100,
      addedCount,
      stage: "任务模式图谱更新完成",
    });
  };

  return (
    <section className="completion-module-wrap">
      <article className="panel wide completion-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Potential Triple Table</p>
            <h3>潜在三元组表格</h3>
          </div>
          <div className="completion-meta">
            <span>{`抽样展示: ${rows.length} 条`}</span>
            <span>{`当前页: ${safePage}/${totalPages}`}</span>
          </div>
        </div>

        <div className="judge-action-row">
          <button
            type="button"
            className="judge-all-btn"
            onClick={handleJudgeAll}
            disabled={processState.status === "running"}
          >
            {processState.status === "running"
              ? "潜在三元组判断中..."
              : "潜在三元组判断"}
          </button>
          <button
            type="button"
            className="judge-all-btn update-kg-btn"
            onClick={handleUpdateGraph}
            disabled={!canUpdateGraph || graphUpdateState.status === "running"}
          >
            {graphUpdateState.status === "running"
              ? "更新任务模式图谱中..."
              : "更新任务模式图谱"}
          </button>
          <span className="process-stage-label">{processState.stage}</span>
        </div>

        <div className="process-track">
          <div
            className="process-fill"
            style={{ width: `${processState.progress}%` }}
          />
        </div>

        <div className="process-stages">
          <span
            className={
              processState.progress >= 1
                ? "process-stage active"
                : "process-stage"
            }
          >
            1. 路径评分
          </span>
          <span
            className={
              processState.progress >= 60
                ? "process-stage active"
                : "process-stage"
            }
          >
            2. 裁剪Top2路径
          </span>
          <span
            className={
              processState.progress >= 85
                ? "process-stage active"
                : "process-stage"
            }
          >
            3. 大模型判断
          </span>
        </div>

        <div className="kg-update-block">
          <div className="kg-update-head">
            <strong>{graphUpdateState.stage}</strong>
            <span>{`新增三元组数量: ${graphUpdateState.addedCount}`}</span>
          </div>
          <div className="process-track kg-update-track">
            <div
              className="process-fill kg-update-fill"
              style={{ width: `${graphUpdateState.progress}%` }}
            />
          </div>
        </div>

        <div className="completion-table-shell">
          <table className="completion-table">
            <thead>
              <tr>
                <th>序号</th>
                <th>潜在三元组</th>
                <th>候选路径（最多3跳）</th>
                <th>是否保留</th>
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => {
                const rowKey = String(row.id);
                const decision = decisionMap[rowKey];
                const displayPaths = decision?.topTwoPaths?.length
                  ? decision.topTwoPaths
                  : row.candidatePaths || [];
                const manualKeep = manualKeepMap[rowKey];
                const finalKeep =
                  typeof manualKeep === "boolean"
                    ? manualKeep
                    : Boolean(decision?.keep);

                return (
                  <tr key={`${row.id}-${pageStartIndex + index}`}>
                    <td>
                      <div className="cell-scroll">
                        {pageStartIndex + index + 1}
                      </div>
                    </td>
                    <td>
                      <div className="cell-scroll triple-cell">
                        <strong>{tripleText(row)}</strong>
                      </div>
                    </td>
                    <td>
                      <div className="cell-scroll path-cell">
                        <div className="path-list">
                          {displayPaths.map((pathItem) => (
                            <div
                              className={
                                decision?.topTwoPaths?.some(
                                  (topPath) => topPath.id === pathItem.id,
                                )
                                  ? "path-row top2"
                                  : "path-row"
                              }
                              key={pathItem.id}
                            >
                              <span>{pathItem.pathText}</span>
                              <em>{`${pathItem.hops} 跳`}</em>
                            </div>
                          ))}
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="cell-scroll">
                        <label className="keep-checkbox">
                          <input
                            type="checkbox"
                            checked={finalKeep}
                            disabled={!decision}
                            onChange={(event) =>
                              setManualKeepMap((previous) => ({
                                ...previous,
                                [rowKey]: event.target.checked,
                              }))
                            }
                          />
                          <span>
                            {decision
                              ? finalKeep
                                ? "保留"
                                : "不保留"
                              : "待判断"}
                          </span>
                        </label>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="completion-pagination">
          <button
            type="button"
            onClick={() => setPage((current) => Math.max(1, current - 1))}
            disabled={safePage === 1}
          >
            上一页
          </button>
          <span>{`第 ${safePage} / ${totalPages} 页`}</span>
          <button
            type="button"
            onClick={() =>
              setPage((current) => Math.min(totalPages, current + 1))
            }
            disabled={safePage === totalPages}
          >
            下一页
          </button>
        </div>
      </article>

      {/* <article className="panel completion-top2-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Process Summary</p>
            <h3>流程结果摘要</h3>
          </div>
        </div>

        <div className="summary-grid">
          <div className="summary-card">
            <strong>{keepSummary.keep}</strong>
            <span>保留三元组</span>
          </div>
          <div className="summary-card">
            <strong>{keepSummary.drop}</strong>
            <span>裁剪三元组</span>
          </div>
        </div>

        {globalTopTwoPaths.length ? (
          <div className="top-path-list">
            {globalTopTwoPaths.map((item, idx) => (
              <div className="top-path-card" key={`${item.rowId}-${item.id}`}>
                <strong>{`#${idx + 1} 全局Top路径`}</strong>
                <p>{item.pathText}</p>
                <small>{`行ID: ${item.rowId}`}</small>
              </div>
            ))}
          </div>
        ) : (
          <p className="top-path-empty">点击“潜在三元组判断”后展示结果。</p>
        )}
      </article> */}
    </section>
  );
}
