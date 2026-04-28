import {
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
} from "react";
import { documents, extractionCases, taskStages } from "./data/systemData";
import allDocumentTriples from "./data/allDocumentTriples.json";
import TaskKnowledgeGraph from "./components/TaskKnowledgeGraph";
import KnowledgeCompletionModule from "./components/KnowledgeCompletionModule";
import PatternRecognitionModule from "./components/PatternRecognitionModule";
import { Alert } from "antd";

const UI = {
  appTitle: "\u4efb\u52a1\u77e5\u8bc6\u5efa\u6a21\u7cfb\u7edf",
  appCopy:
    "\u9762\u5411\u77e5\u8bc6\u62bd\u53d6\u3001\u56fe\u8c31\u6784\u5efa\u3001\u56fe\u8c31\u8865\u5168\u4e0e\u6a21\u5f0f\u8bc6\u522b\u4e00\u4f53\u5316\u754c\u9762\u3002",
  systemOverview: "\u7cfb\u7edf\u603b\u89c8",
  extraction: "\u77e5\u8bc6\u62bd\u53d6",
  graph: "\u77e5\u8bc6\u56fe\u8c31",
  completion: "\u56fe\u8c31\u8865\u5168",
  pattern: "\u6a21\u5f0f\u8bc6\u522b",
  workflow: "\u6838\u5fc3\u6d41\u7a0b",
  overviewArch: "\u7cfb\u7edf\u6d41\u7a0b\u67b6\u6784",
  keywordTitle: "\u9ad8\u9891\u4e3b\u9898\u8bcd",
  chapterTitle: "\u8bba\u6587\u6a21\u5757\u6620\u5c04",
  docPool: "\u6587\u6863\u6837\u672c\u6c60",
  searchPlaceholder:
    "\u641c\u7d22\u6587\u6863\u7f16\u53f7\u3001\u539f\u6587\u3001\u6807\u7b7e\u6216\u5b9e\u4f53",
  pageInfo: "\u9875\u7801\u4fe1\u606f",
  pageSizeUnit: "\u6761/\u9875",
  jumpTo: "\u8df3\u8f6c\u5230",
  extractPanel: "\u77e5\u8bc6\u62bd\u53d6\u5206\u6790\u533a",
  extractButton: "\u77e5\u8bc6\u62bd\u53d6",
  extractAllButton: "\u5168\u90e8\u6587\u6863\u62bd\u53d6",
  extracting: "\u62bd\u53d6\u8fdb\u884c\u4e2d...",
  extractingAll: "\u5168\u90e8\u6587\u6863\u62bd\u53d6\u4e2d...",
  savedToJson:
    "\u62bd\u53d6\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230 JSON \u6587\u4ef6",
  originalText: "\u539f\u59cb\u6587\u672c",
  progressTitle: "\u77e5\u8bc6\u62bd\u53d6\u8fdb\u5ea6",
  entityCandidates: "\u5019\u9009\u5b9e\u4f53",
  markedText: "\u8fb9\u754c\u6807\u8bb0\u6587\u672c",
  relationTriples:
    "\u89c4\u5219\u7ea6\u675f\u4e0b\u7684\u5173\u7cfb\u4e09\u5143\u7ec4",
  fromJson: "\u6765\u81ea JSON \u6837\u672c",
  docLabel: "\u6587\u6863",
  charUnit: "\u5b57",
  itemUnit: "\u9879",
  rowUnit: "\u6761",
  graphOverview: "\u4efb\u52a1\u77e5\u8bc6\u56fe\u8c31\u6982\u89c8",
  relationSchema: "\u5173\u7cfb\u6a21\u5f0f",
  modeling: "\u5efa\u6a21\u601d\u8def",
  completionTitle: "\u56fe\u8c31\u8865\u5168\u5019\u9009",
  reasoning: "\u63a8\u7406\u8bf4\u660e",
  patternTitle: "\u6a21\u5f0f\u8bc6\u522b\u7ed3\u679c",
  output: "\u8f93\u51fa\u8bf4\u660e",
  nodeWeight: "\u8282\u70b9\u6743\u91cd",
  pagePrefix: "\u7b2c",
  pageSuffix: "\u9875",
  pageOf: "\u5171",
  matchedDocs: "\u6761\u5339\u914d\u6587\u6863",
  entityRoleSep: " | ",
};

const loadingHints = [
  "\u6b63\u5728\u5b9a\u4f4d\u5b9e\u4f53\u8fb9\u754c\u6807\u8bb0...",
  "\u6b63\u5728\u7b5b\u9009\u5019\u9009\u5b9e\u4f53...",
  "\u6b63\u5728\u5e94\u7528\u89c4\u5219\u7ea6\u675f...",
  "\u6b63\u5728\u751f\u6210\u5173\u7cfb\u4e09\u5143\u7ec4...",
];

const sectionItems = [
  { key: "overview", label: "系统总览" },
  { key: "extract", label: "任务模式知识抽取" },
  { key: "graph", label: "任务模式图谱展示" },
  { key: "completion", label: "任务模式图谱补全" },
  { key: "pattern", label: "任务模式识别" },
];
const graphDependentSectionKeys = ["graph", "completion", "pattern"];

const EXTRACTION_API_BASE = (
  import.meta.env.VITE_EXTRACTION_API_BASE || "http://localhost:8787"
).replace(/\/$/, "");

async function requestExtractionApi(path) {
  const response = await fetch(`${EXTRACTION_API_BASE}${path}`);
  let payload = null;

  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok || !payload?.success) {
    throw new Error(payload?.error || `请求失败（${response.status}）`);
  }

  return payload;
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

const initialExtractionState = {
  status: "idle",
  mode: "single",
  progress: 0,
  duration: 0,
  entities: [],
  relations: [],
};

const overviewArchitectureSteps = [
  { step: "01", title: "实体边界标记" },
  { step: "02", title: "筛选候选实体" },
  { step: "03", title: "应用规则约束生成任务模式三元组" },
  { step: "04", title: "任务模式图谱展示" },
  { step: "05", title: "构建潜在三元组的候选路径" },
  { step: "06", title: "判断潜在三元组是否应该保留" },
  { step: "07", title: "使用保留的潜在三元组更新任务模式图谱" },
  { step: "08", title: "上传预定义任务模式" },
  { step: "09", title: "进行任务模式识别并显示识别结果" },
];

const overviewKeywordLexicon = [
  "美国",
  "俄罗斯",
  "北约",
  "欧洲",
  "导弹",
  "部署",
  "试验",
  "展示",
  "防御系统",
  "装备",
  "任务",
  "目标",
  "海军",
  "空军",
  "雷达",
  "无人机",
];

const eventTypeKeywordMap = {
  Deploy: "部署",
  Exhibit: "展示",
  Experiment: "试验",
  Support: "支持",
  Attack: "打击",
  Transport: "运输",
  Obtain: "获取",
};

function buildOverviewStats(records) {
  const safeRecords = Array.isArray(records) ? records : [];
  const parsedRecords = safeRecords
    .filter((item) => item && typeof item === "object")
    .map((item, index) => {
      const text = typeof item.text === "string" ? item.text : "";
      const eventList = Array.isArray(item.event_list) ? item.event_list : [];
      return {
        id: item.id ?? String(index + 1),
        text,
        eventList,
      };
    });

  const docCount = parsedRecords.length;
  const totalLength = parsedRecords.reduce(
    (sum, item) => sum + item.text.length,
    0,
  );
  const avgLength = Math.round(totalLength / Math.max(docCount, 1));
  const taggedCount = parsedRecords.filter(
    (item) => item.eventList.length > 0,
  ).length;
  const highInfoCount = parsedRecords.filter(
    (item) => item.text.length >= 220,
  ).length;

  const keywordCounter = new Map();
  overviewKeywordLexicon.forEach((keyword) => keywordCounter.set(keyword, 0));
  parsedRecords.forEach((item) => {
    overviewKeywordLexicon.forEach((keyword) => {
      if (item.text.includes(keyword)) {
        keywordCounter.set(keyword, (keywordCounter.get(keyword) || 0) + 1);
      }
    });
    item.eventList.forEach((event) => {
      const mapped = eventTypeKeywordMap[event?.event_type];
      if (!mapped) return;
      keywordCounter.set(mapped, (keywordCounter.get(mapped) || 0) + 1);
    });
  });

  const keywords = Array.from(keywordCounter.entries())
    .map(([keyword, count]) => ({ keyword, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  return {
    metrics: [
      {
        label: "文档样本总数",
        value: docCount.toLocaleString("zh-CN"),
        note: "来自导入文本数据",
      },
      {
        label: "平均文本长度",
        value: String(avgLength),
        note: "按字符数统计",
      },
      {
        label: "高信息密度文档",
        value: String(highInfoCount),
        note: "文本长度 >= 220",
      },
    ],
    keywords,
  };
}

function buildPagination(currentPage, totalPages) {
  // 总页数少时全部显示（从7改为5）
  if (totalPages <= 5) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }

  // 前段（减少两个按钮：只显示3个页码）
  if (currentPage <= 3) {
    return [1, 2, 3, "...", totalPages];
  }

  // 后段
  if (currentPage >= totalPages - 2) {
    return [1, "...", totalPages - 2, totalPages - 1, totalPages];
  }

  // 中间（只保留当前页）
  return [1, "...", currentPage, "...", totalPages];
}

function App() {
  const [activeSection, setActiveSection] = useState("overview");
  const [overviewStats, setOverviewStats] = useState({
    metrics: [],
    keywords: [],
  });
  const [overviewUploadStatus, setOverviewUploadStatus] =
    useState("请先导入文本数据");
  const [overviewImported, setOverviewImported] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState(documents[0]?.id ?? "");
  const [searchText, setSearchText] = useState("");
  const itemsPerPage = 5;
  const [currentPage, setCurrentPage] = useState(1);
  const [jumpPageInput, setJumpPageInput] = useState("");
  const [extractionState, setExtractionState] = useState(
    initialExtractionState,
  );
  const [batchSummary, setBatchSummary] = useState("");
  const [hasTriggeredBatchExtraction, setHasTriggeredBatchExtraction] =
    useState(false);
  const deferredSearchText = useDeferredValue(searchText);
  const timersRef = useRef([]);
  const overviewFileInputRef = useRef(null);
  const [uploadErrorVisible, setUploadErrorVisible] = useState(false);
  const [extractError, setExtractError] = useState("");
  const extractionRequestRef = useRef(0);

  const clearExtractionTimers = () => {
    timersRef.current.forEach((timerId) => {
      window.clearTimeout(timerId);
    });
    timersRef.current = [];
  };

  const filteredDocuments = documents.filter((item) => {
    const query = deferredSearchText.trim();
    if (!query) return true;
    return (
      item.id.includes(query) ||
      item.originalText.includes(query) ||
      item.markedText.includes(query) ||
      item.entities.some((entity) => entity.text.includes(query))
    );
  });

  const totalPages = Math.max(
    1,
    Math.ceil(filteredDocuments.length / itemsPerPage),
  );
  const visibleDocuments = filteredDocuments.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage,
  );
  const paginationItems = buildPagination(currentPage, totalPages);

  const selectedDocument =
    documents.find((item) => item.id === selectedDocId) ||
    filteredDocuments[0] ||
    documents[0];
  const selectedCase =
    extractionCases.find((item) => item.id === selectedDocument?.id) ||
    extractionCases[0];

  useEffect(() => {
    setCurrentPage(1);
  }, [deferredSearchText]);

  useEffect(() => {
    if (!filteredDocuments.length) return;
    if (!filteredDocuments.some((item) => item.id === selectedDocId)) {
      setSelectedDocId(filteredDocuments[0].id);
    }
  }, [filteredDocuments, selectedDocId]);

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  useEffect(() => {
    clearExtractionTimers();
    extractionRequestRef.current += 1;
    setExtractionState(initialExtractionState);
    setBatchSummary("");
    setExtractError("");
    return clearExtractionTimers;
  }, [selectedDocId]);

  useEffect(() => clearExtractionTimers, []);

  useEffect(() => {
    if (activeSection === "overview") return;
    if (!overviewImported) {
      setActiveSection("overview");
      return;
    }
    if (
      !hasTriggeredBatchExtraction &&
      graphDependentSectionKeys.includes(activeSection)
    ) {
      setActiveSection("extract");
    }
  }, [activeSection, hasTriggeredBatchExtraction, overviewImported]);

  const updateLoadingProgress = (mode, ceiling = 95) => {
    const intervalId = window.setInterval(() => {
      setExtractionState((previous) => {
        if (previous.status !== "loading" || previous.mode !== mode) {
          return previous;
        }

        const gap = ceiling - previous.progress;
        const step = Math.max(1, Math.ceil(gap * 0.12));

        return {
          ...previous,
          progress: Math.min(ceiling, previous.progress + step),
        };
      });
    }, 180);

    timersRef.current = [intervalId];
    return intervalId;
  };

  const handleExtract = async () => {
    if (!selectedDocument?.id || !selectedCase) return;

    clearExtractionTimers();
    setBatchSummary("");
    setExtractError("");

    const requestId = extractionRequestRef.current + 1;
    extractionRequestRef.current = requestId;
    const startedAt = performance.now();

    setExtractionState({
      status: "loading",
      mode: "single",
      progress: 4,
      duration: 0,
      entities: [],
      relations: [],
    });

    const intervalId = updateLoadingProgress("single", 96);

    try {
      await Promise.all([
        requestExtractionApi(
          `/api/extraction/single?docId=${encodeURIComponent(String(selectedDocument.id))}`,
        ),
        delay(5000),
      ]);

      if (requestId !== extractionRequestRef.current) return;

      window.clearInterval(intervalId);

      const duration = Math.max(1, Math.round(performance.now() - startedAt));

      startTransition(() => {
        setExtractionState({
          status: "done",
          mode: "single",
          progress: 100,
          duration,
          entities: selectedCase.entities,
          relations: selectedCase.relations,
        });
      });
    } catch (error) {
      if (requestId !== extractionRequestRef.current) return;

      window.clearInterval(intervalId);
      setExtractionState({ ...initialExtractionState, mode: "single" });
      setExtractError(
        error instanceof Error ? error.message : "单个文档抽取失败",
      );
    } finally {
      if (requestId === extractionRequestRef.current) {
        timersRef.current = [];
      }
    }
  };

  const handleExtractAll = async () => {
    clearExtractionTimers();
    setHasTriggeredBatchExtraction(false);
    setBatchSummary("");
    setExtractError("");

    const requestId = extractionRequestRef.current + 1;
    extractionRequestRef.current = requestId;
    const startedAt = performance.now();

    setExtractionState({
      status: "loading",
      mode: "batch",
      progress: 2,
      duration: 0,
      entities: [],
      relations: [],
    });

    const intervalId = updateLoadingProgress("batch", 95);

    try {
      await Promise.all([
        requestExtractionApi(`/api/extraction/all`),
        delay(1000),
      ]);

      if (requestId !== extractionRequestRef.current) return;

      window.clearInterval(intervalId);

      const duration = Math.max(1, Math.round(performance.now() - startedAt));

      startTransition(() => {
        setExtractionState({
          status: "done",
          mode: "batch",
          progress: 100,
          duration,
          entities: [],
          relations: allDocumentTriples.triples.map((triple) => ({
            head: triple.head,
            relation: triple.relation,
            tail: triple.tail,
            evidence: `出现 ${triple.occurrenceCount} 次，来源文档：${triple.sourceDocIds.join(", ")}`,
            rule: UI.savedToJson,
          })),
        });
        setBatchSummary(
          `${UI.savedToJson}：${allDocumentTriples.totalUniqueTriples} 条唯一三元组，覆盖 ${allDocumentTriples.documentsWithTriples} 篇文档。`,
        );
        setHasTriggeredBatchExtraction(true);
      });
    } catch (error) {
      if (requestId !== extractionRequestRef.current) return;

      window.clearInterval(intervalId);
      setExtractionState({ ...initialExtractionState, mode: "batch" });
      setHasTriggeredBatchExtraction(false);
      setExtractError(
        error instanceof Error ? error.message : "全部文档抽取失败",
      );
    } finally {
      if (requestId === extractionRequestRef.current) {
        timersRef.current = [];
      }
    }
  };

  const hintIndex = Math.min(
    loadingHints.length - 1,
    Math.floor((extractionState.progress / 100) * loadingHints.length),
  );

  const handleChooseOverviewFile = () => {
    overviewFileInputRef.current?.click();
  };

  const handleOverviewFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const content = await file.text();
      const parsed = JSON.parse(content);

      if (!Array.isArray(parsed)) {
        throw new Error("invalid format");
      }

      const requiredFields = ["id", "text"];

      const isValid = parsed.every((item) => {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
          return false;
        }

        const hasRequiredFields = requiredFields.every((field) =>
          Object.prototype.hasOwnProperty.call(item, field),
        );

        if (!hasRequiredFields) {
          return false;
        }

        if (typeof item.id !== "string" && typeof item.id !== "number") {
          return false;
        }

        if (typeof item.text !== "string") {
          return false;
        }

        return true;
      });

      if (!isValid) {
        throw new Error("invalid format");
      }

      const records = parsed;
      const nextStats = buildOverviewStats(records);
      setOverviewStats(nextStats);
      setOverviewImported(true);
      setHasTriggeredBatchExtraction(false);
      setUploadErrorVisible(false);
      setOverviewUploadStatus(
        `已导入 ${file.name}，共 ${records.length.toLocaleString("zh-CN")} 篇文档`,
      );
    } catch (error) {
      setOverviewImported(false);
      setHasTriggeredBatchExtraction(false);
      setOverviewStats({ metrics: [], keywords: [] });
      setUploadErrorVisible(true);
      setOverviewUploadStatus("数据导入失败，请重新选择正确格式的文件");
    } finally {
      event.target.value = "";
    }
  };

  const isSectionEnabled = (sectionKey) => {
    if (sectionKey === "overview") return true;
    if (sectionKey === "extract") return overviewImported;
    if (graphDependentSectionKeys.includes(sectionKey)) {
      return overviewImported && hasTriggeredBatchExtraction;
    }
    return true;
  };

  const getSectionDisabledReason = (sectionKey) => {
    if (!overviewImported && sectionKey !== "overview") {
      return "请先在系统总览页面导入文本数据";
    }
    if (
      graphDependentSectionKeys.includes(sectionKey) &&
      !hasTriggeredBatchExtraction
    ) {
      return "请先在知识抽取页面完成“全部文档抽取”";
    }
    return undefined;
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-card">
          <h1>任务知识构建与模式识别系统</h1>
          <p className="brand-copy">
            面向知识抽取、图谱构建、图谱补全与模式识别的一体化界面
          </p>
        </div>

        <nav className="nav-list">
          {sectionItems.map((item) => {
            const enabled = isSectionEnabled(item.key);
            return (
              <button
                key={item.key}
                className={
                  item.key === activeSection ? "nav-item active" : "nav-item"
                }
                onClick={() => setActiveSection(item.key)}
                type="button"
                disabled={!enabled}
                title={enabled ? undefined : getSectionDisabledReason(item.key)}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="stage-card">
          <p className="card-title">{UI.workflow}</p>
          {taskStages.map((stage, index) => (
            <div className="stage-row" key={stage}>
              <span>{`0${index + 1}`}</span>
              <strong>{stage}</strong>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-panel">
        {activeSection === "overview" && (
          <section className="content-grid graph-single-layout">
            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Data Import</p>
                  <h3>文本数据导入</h3>
                </div>
                <div className="extract-actions">
                  <input
                    ref={overviewFileInputRef}
                    type="file"
                    accept=".json,application/json"
                    className="overview-hidden-input"
                    onChange={handleOverviewFileChange}
                  />
                  <button
                    className="extract-btn secondary"
                    type="button"
                    onClick={handleChooseOverviewFile}
                  >
                    选择文本数据
                  </button>
                </div>
              </div>
              <p className="overview-upload-status">{overviewUploadStatus}</p>
              {uploadErrorVisible && (
                <Alert
                  style={{ marginTop: 12 }}
                  message="请上传正确格式的数据"
                  type="error"
                  showIcon
                  closable
                  onClose={() => setUploadErrorVisible(false)}
                />
              )}
            </article>
          </section>
        )}

        {activeSection === "overview" && overviewImported && (
          <section className="metrics-grid">
            {overviewStats.metrics.map((item) => (
              <article className="metric-card" key={item.label}>
                <p>{item.label}</p>
                <strong>{item.value}</strong>
              </article>
            ))}
          </section>
        )}

        {activeSection === "overview" && overviewImported && (
          <section className="content-grid">
            <article className="panel wide">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Architecture</p>
                  <h3>{UI.overviewArch}</h3>
                </div>
              </div>
              <div className="timeline">
                {overviewArchitectureSteps.map((item) => (
                  <div className="timeline-item" key={item.step}>
                    <span>{item.step}</span>
                    <div>
                      <strong>{item.title}</strong>
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Keywords</p>
                  <h3>{UI.keywordTitle}</h3>
                </div>
              </div>
              <div className="keyword-list">
                {overviewStats.keywords.map((item) => (
                  <div className="keyword-row" key={item.keyword}>
                    <span>{item.keyword}</span>
                    <strong>{item.count}</strong>
                  </div>
                ))}
              </div>
            </article>
          </section>
        )}

        {activeSection === "extract" && (
          <section
            className={
              extractionState.status === "done" &&
              extractionState.mode === "batch"
                ? "content-grid extract-layout batch-layout"
                : "content-grid extract-layout"
            }
          >
            <article className="panel extract-doc-panel">
              <div className="panel-header">
                <div>
                  <h3>文档池</h3>
                </div>
                <input
                  className="search-input"
                  placeholder={UI.searchPlaceholder}
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                />
              </div>

              <div className="document-list doc-pool-list">
                {visibleDocuments.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={
                      item.id === selectedDocument?.id
                        ? "doc-card active"
                        : "doc-card"
                    }
                    onClick={() => setSelectedDocId(item.id)}
                  >
                    <div className="doc-card-header">
                      <strong>{`${UI.docLabel} #${item.id}`}</strong>
                      <span>{`${item.length} ${UI.charUnit}`}</span>
                    </div>
                    <div className="marked-snippet">
                      <p>{item.markedPreview}</p>
                    </div>
                  </button>
                ))}
              </div>

              <div className="pagination-bar">
                <button
                  type="button"
                  className="page-arrow"
                  onClick={() =>
                    setCurrentPage((page) => Math.max(1, page - 1))
                  }
                  disabled={currentPage === 1}
                >
                  {"<"}
                </button>

                <div className="page-number-list">
                  {paginationItems.map((item, index) =>
                    item === "..." ? (
                      <span className="page-ellipsis" key={`ellipsis-${index}`}>
                        ...
                      </span>
                    ) : (
                      <button
                        type="button"
                        key={item}
                        className={
                          item === currentPage
                            ? "page-number active"
                            : "page-number"
                        }
                        onClick={() => setCurrentPage(item)}
                      >
                        {item}
                      </button>
                    ),
                  )}
                </div>

                <button
                  type="button"
                  className="page-arrow"
                  onClick={() =>
                    setCurrentPage((page) => Math.min(totalPages, page + 1))
                  }
                  disabled={currentPage === totalPages}
                >
                  {">"}
                </button>

                <label className="page-jump">
                  <span>{UI.jumpTo}</span>
                  <input
                    value={jumpPageInput}
                    onChange={(event) =>
                      setJumpPageInput(event.target.value.replace(/\D/g, ""))
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        const targetPage = Number(jumpPageInput);
                        if (targetPage >= 1 && targetPage <= totalPages) {
                          setCurrentPage(targetPage);
                        }
                      }
                    }}
                  />
                  <span>{UI.pageSuffix}</span>
                </label>
              </div>
            </article>

            <article
              className={
                extractionState.status === "done" &&
                extractionState.mode === "batch"
                  ? "panel wide batch-result-panel"
                  : "panel wide"
              }
            >
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Extraction</p>
                  <h3>
                    {extractionState.mode === "batch"
                      ? UI.extractPanel
                      : `${UI.extractPanel} - ${UI.docLabel} #${selectedDocument?.id}`}
                  </h3>
                </div>
                <div className="extract-actions">
                  <button
                    className="extract-btn secondary"
                    type="button"
                    onClick={handleExtractAll}
                  >
                    {extractionState.status === "loading" &&
                    extractionState.mode === "batch"
                      ? UI.extractingAll
                      : UI.extractAllButton}
                  </button>
                  <button
                    className="extract-btn"
                    type="button"
                    onClick={handleExtract}
                  >
                    {extractionState.status === "loading" &&
                    extractionState.mode === "single"
                      ? UI.extracting
                      : "抽取单个文档"}
                  </button>
                </div>
              </div>

              {extractError && (
                <Alert
                  style={{ marginBottom: 12 }}
                  message={extractError}
                  type="error"
                  showIcon
                  closable
                  onClose={() => setExtractError("")}
                />
              )}

              <div
                className={
                  extractionState.status === "done" &&
                  extractionState.mode === "batch"
                    ? "analysis-grid batch-analysis-grid"
                    : "analysis-grid"
                }
              >
                {extractionState.mode !== "batch" && (
                  <div className="sub-panel span-2">
                    <div className="sub-panel-head">
                      <h4>{UI.originalText}</h4>
                      <span>{`${selectedDocument?.length ?? 0} ${UI.charUnit}`}</span>
                    </div>
                    <p className="text-block">
                      {selectedDocument?.originalText}
                    </p>
                  </div>
                )}

                {extractionState.status === "loading" && (
                  <div className="sub-panel span-2 loading-panel">
                    <div className="sub-panel-head">
                      <h4>{UI.progressTitle}</h4>
                      <span>{`${extractionState.progress}%`}</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: `${extractionState.progress}%` }}
                      />
                    </div>
                    <p className="loading-hint">{loadingHints[hintIndex]}</p>
                    <div className="loading-steps">
                      {loadingHints.map((hint, index) => (
                        <div
                          className={
                            index <= hintIndex
                              ? "loading-step active"
                              : "loading-step"
                          }
                          key={hint}
                        >
                          {hint}
                        </div>
                      ))}
                    </div>
                    {extractionState.mode === "batch" ? (
                      <p className="batch-export-note">
                        {`${allDocumentTriples.totalDocuments} 篇文档正在统一抽取，预计生成全量关系三元组文件。`}
                      </p>
                    ) : null}
                  </div>
                )}

                {extractionState.status === "done" &&
                  extractionState.mode === "single" && (
                    <>
                      <div className="sub-panel">
                        <div className="sub-panel-head candidate-entity-head">
                          <h4>{UI.entityCandidates}</h4>
                          <span>{`${extractionState.entities.length} ${UI.itemUnit}`}</span>
                        </div>
                        <div className="chip-wrap equal-height-scroll">
                          {extractionState.entities.map((entity) => (
                            <span
                              className="chip"
                              key={`${entity.role}-${entity.text}`}
                            >
                              {`${entity.text}${UI.entityRoleSep}${entity.role}`}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="sub-panel">
                        <div className="sub-panel-head">
                          <h4>{UI.markedText}</h4>
                        </div>
                        <p className="text-block marked-text-block equal-height-scroll">
                          {selectedCase?.markedText}
                        </p>
                      </div>

                      <div className="sub-panel span-2">
                        <div className="sub-panel-head">
                          <h4>{UI.relationTriples}</h4>
                          <span>{`${extractionState.relations.length} ${UI.rowUnit}`}</span>
                        </div>
                        <div className="relation-table fixed-height-scroll">
                          {extractionState.relations.map((item) => (
                            <div
                              className="relation-row"
                              key={`${item.head}-${item.relation}-${item.tail}`}
                            >
                              <div>
                                <strong>{`${item.head} - ${item.relation} - ${item.tail}`}</strong>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </>
                  )}

                {extractionState.status === "done" &&
                  extractionState.mode === "batch" && (
                    <div className="sub-panel span-2 batch-relation-panel">
                      <div className="sub-panel-head">
                        <h4>{UI.relationTriples}</h4>
                        <span>{`${extractionState.relations.length} ${UI.rowUnit}`}</span>
                      </div>
                      <div className="relation-table fixed-height-scroll">
                        {extractionState.relations.map((item) => (
                          <div
                            className="relation-row"
                            key={`${item.head}-${item.relation}-${item.tail}`}
                          >
                            <div>
                              <strong>{`${item.head} - ${item.relation} - ${item.tail}`}</strong>
                              <p>{item.evidence}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </article>
          </section>
        )}

        {activeSection === "graph" && (
          <section className="content-grid graph-single-layout">
            <TaskKnowledgeGraph />
          </section>
        )}

        {activeSection === "completion" && (
          <section className="content-grid graph-single-layout">
            <KnowledgeCompletionModule />
          </section>
        )}

        {activeSection === "pattern" && (
          <section className="content-grid graph-single-layout">
            <PatternRecognitionModule />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
