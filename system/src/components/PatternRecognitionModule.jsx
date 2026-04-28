import { useEffect, useMemo, useRef, useState } from "react";
import { Alert } from "antd";
import "./PatternRecognitionModule.css";

const htmlAssetMap = import.meta.glob("../model/部署构建模式/*.html", {
  eager: true,
  query: "?url",
  import: "default",
});

const deploymentBuildRawAssetMap = {
  ...import.meta.glob("../model/部署构建文件/**/*", {
    eager: true,
    query: "?raw",
    import: "default",
  }),
  ...import.meta.glob("../model/部署构建/**/*", {
    eager: true,
    query: "?raw",
    import: "default",
  }),
};
const MATCHED_SUBGRAPHS_PER_PAGE = 5;
const PATTERN_API_BASE = (
  import.meta.env.VITE_EXTRACTION_API_BASE || "http://localhost:8787"
).replace(/\/$/, "");

function sortByFileName(a, b) {
  const aNum = Number.parseInt(a.baseName, 10);
  const bNum = Number.parseInt(b.baseName, 10);
  const bothNumeric = !Number.isNaN(aNum) && !Number.isNaN(bNum);
  if (bothNumeric) {
    return aNum - bNum;
  }
  return a.baseName.localeCompare(b.baseName, "zh-CN");
}

const graphTextRepairPairs = [
  ["?/b>", "</b>"],
  ["?br>", "<br>"],
  ["俄罗?", "俄罗斯"],
  ["伊拉?", "伊拉克"],
  ["阿富?", "阿富汗"],
  ["奥地?", "奥地利"],
  ["美国?", "美国号"],
  ["黄蜂?", "黄蜂号"],
  ["新奥尔良?", "新奥尔良号"],
  ["追踪者系?", "追踪者系统"],
  ["魔爪军用机器?", "魔爪军用机器人"],
  ["魔爪IV机器?", "魔爪IV机器人"],
  ["零备?", "零备件"],
  ["克拉斯诺雅茨?", "克拉斯诺雅茨克"],
  ["阿尔泰地?", "阿尔泰地区"],
  ["摩尔曼斯?", "摩尔曼斯克"],
  ["伊尔库茨?", "伊尔库茨克"],
  ["加里宁格?", "加里宁格勒"],
  ["印度-太平?", "印度-太平洋"],
  ["印度-太平洋地?", "印度-太平洋地区"],
  ["编队航行和作?", "编队航行和作战"],
  ["2010?", "2010年"],
  ["2013?", "2013年"],
  ["2018?", "2018年"],
  ["2?5", "2月5"],
  ["1?", "1日"],
  ["?00%", "100%"],
  ["?019", "2019"],
  ["?013", "2013"],
  ["去?2月", "去年12月"],
  ["朢新", "最新"],
  ["朢先进", "最先进"],
  ["朢尖端", "最尖端"],
  ["朢为首要", "最为首要"],
  ["朢为", "最为"],
  ["朢", "最"],
  ["丢种", "第一种"],
  ["箢易", "简易"],
  ["抢术", "技术"],
  ["进?010", "进展2010"],
  ["UAS ?", "UAS。"],
];

function sanitizeGraphHtml(rawHtml) {
  if (typeof rawHtml !== "string" || !rawHtml) {
    return "";
  }

  let next = rawHtml;
  next = next.replace(
    /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g,
    "",
  );
  graphTextRepairPairs.forEach(([from, to]) => {
    next = next.replaceAll(from, to);
  });

  next = next.replace(
    /([\u4e00-\u9fffA-Za-z0-9])\?([\u4e00-\u9fffA-Za-z0-9])/g,
    "$1$2",
  );
  next = next.replace(
    /([\u4e00-\u9fffA-Za-z0-9])\?(?=(<br>|<|\"|,|，|。|$))/g,
    "$1",
  );

  // Keep only the graph canvas in embedded previews.
  next = next.replace(
    /<\/style>/i,
    `
    .page { padding: 0 !important; }
    .panel {
      border: none !important;
      border-radius: 0 !important;
      box-shadow: none !important;
    }
    .header, .title, .meta, .legend { display: none !important; }
    #mynetwork {
      height: 100vh !important;
      min-height: 100vh !important;
    }
    .vis-navigation, .vis-button { display: none !important; }
  </style>`,
  );
  next = next.replace(
    /navigationButtons\s*:\s*true/g,
    "navigationButtons: false",
  );

  return next;
}

function extractNodeCountFromHtml(html) {
  if (typeof html !== "string" || !html) {
    return 0;
  }

  const match = html.match(/const\s+nodesData\s*=\s*(\[[\s\S]*?\]);/);
  if (!match) {
    return 0;
  }

  try {
    const parsed = JSON.parse(match[1]);
    return Array.isArray(parsed) ? parsed.length : 0;
  } catch (error) {
    return 0;
  }
}

function hasValidPatternJsonShape(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return false;
  }

  const nodes = parsed.nodes ?? parsed.nodesData;
  const edges = parsed.edges ?? parsed.edgesData;
  return Array.isArray(nodes) && Array.isArray(edges);
}

async function requestPatternRecognitionCache(importedFileName) {
  const response = await fetch(`${PATTERN_API_BASE}/api/pattern/recognize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      importedFileName: importedFileName || "",
    }),
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok || !payload?.success) {
    throw new Error(
      payload?.error || `任务模式识别接口请求失败（${response.status}）`,
    );
  }

  return payload;
}

const ZIP_UTF8_FLAG = 0x0800;
const crc32Table = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let c = i;
    for (let j = 0; j < 8; j += 1) {
      c = (c & 1) === 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function getCrc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = (crc >>> 8) ^ crc32Table[(crc ^ bytes[i]) & 0xff];
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function getDosTime(date) {
  const hours = date.getHours() & 0x1f;
  const minutes = date.getMinutes() & 0x3f;
  const seconds = Math.floor(date.getSeconds() / 2) & 0x1f;
  return (hours << 11) | (minutes << 5) | seconds;
}

function getDosDate(date) {
  const year = Math.max(1980, date.getFullYear());
  const month = (date.getMonth() + 1) & 0x0f;
  const day = date.getDate() & 0x1f;
  return ((year - 1980) << 9) | (month << 5) | day;
}

function createZipBlob(files) {
  const zipParts = [];
  const centralDirectoryParts = [];
  const now = new Date();
  const dosTime = getDosTime(now);
  const dosDate = getDosDate(now);
  const encoder = new TextEncoder();
  let currentOffset = 0;

  files.forEach((file) => {
    const fileName = file.name.replace(/\\/g, "/");
    const fileNameBytes = encoder.encode(fileName);
    const data = file.data;
    const crc32 = getCrc32(data);

    const localHeader = new Uint8Array(30 + fileNameBytes.length);
    const localView = new DataView(localHeader.buffer);
    localView.setUint32(0, 0x04034b50, true);
    localView.setUint16(4, 20, true);
    localView.setUint16(6, ZIP_UTF8_FLAG, true);
    localView.setUint16(8, 0, true);
    localView.setUint16(10, dosTime, true);
    localView.setUint16(12, dosDate, true);
    localView.setUint32(14, crc32, true);
    localView.setUint32(18, data.length, true);
    localView.setUint32(22, data.length, true);
    localView.setUint16(26, fileNameBytes.length, true);
    localView.setUint16(28, 0, true);
    localHeader.set(fileNameBytes, 30);

    const centralHeader = new Uint8Array(46 + fileNameBytes.length);
    const centralView = new DataView(centralHeader.buffer);
    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, ZIP_UTF8_FLAG, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, dosTime, true);
    centralView.setUint16(14, dosDate, true);
    centralView.setUint32(16, crc32, true);
    centralView.setUint32(20, data.length, true);
    centralView.setUint32(24, data.length, true);
    centralView.setUint16(28, fileNameBytes.length, true);
    centralView.setUint16(30, 0, true);
    centralView.setUint16(32, 0, true);
    centralView.setUint16(34, 0, true);
    centralView.setUint16(36, 0, true);
    centralView.setUint32(38, 0, true);
    centralView.setUint32(42, currentOffset, true);
    centralHeader.set(fileNameBytes, 46);

    zipParts.push(localHeader, data);
    centralDirectoryParts.push(centralHeader);
    currentOffset += localHeader.length + data.length;
  });

  const centralDirectorySize = centralDirectoryParts.reduce(
    (total, part) => total + part.length,
    0,
  );
  const endOfCentralDirectory = new Uint8Array(22);
  const endView = new DataView(endOfCentralDirectory.buffer);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, files.length, true);
  endView.setUint16(10, files.length, true);
  endView.setUint32(12, centralDirectorySize, true);
  endView.setUint32(16, currentOffset, true);
  endView.setUint16(20, 0, true);

  return new Blob(
    [...zipParts, ...centralDirectoryParts, endOfCentralDirectory],
    { type: "application/zip" },
  );
}

function getTimestamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  const year = now.getFullYear();
  const month = pad(now.getMonth() + 1);
  const day = pad(now.getDate());
  const hours = pad(now.getHours());
  const minutes = pad(now.getMinutes());
  const seconds = pad(now.getSeconds());
  return `${year}${month}${day}${hours}${minutes}${seconds}`;
}

export default function PatternRecognitionModule() {
  const fileInputRef = useRef(null);
  const recognizeIntervalRef = useRef(null);
  const recognizeTimeoutRef = useRef(null);

  const [importStatus, setImportStatus] = useState("尚未导入预定义任务模式");
  const [importedFileName, setImportedFileName] = useState("");
  const [importSucceed, setImportSucceed] = useState(false);
  const [uploadErrorVisible, setUploadErrorVisible] = useState(false);

  const [recognized, setRecognized] = useState(false);
  const [recognizeStatus, setRecognizeStatus] = useState("待识别");
  const [recognizeProgress, setRecognizeProgress] = useState(0);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [matchedSubgraphs, setMatchedSubgraphs] = useState([]);
  const [matchedPage, setMatchedPage] = useState(1);
  const [sanitizedHtmlByPath, setSanitizedHtmlByPath] = useState({});
  const [isDownloadingBuildZip, setIsDownloadingBuildZip] = useState(false);
  const [downloadStatus, setDownloadStatus] =
    useState("识别完成后可下载部署构建压缩包");

  const allPatternHtmlFiles = useMemo(
    () =>
      Object.entries(htmlAssetMap)
        .map(([path, assetUrl]) => {
          const fileName = path.split("/").pop() || "";
          const baseName = fileName.replace(/\.html$/i, "");
          return {
            path,
            fileName,
            baseName,
            url: String(assetUrl),
          };
        })
        .sort(sortByFileName),
    [],
  );

  const preview43 = useMemo(
    () => allPatternHtmlFiles.find((item) => item.baseName === "43") || null,
    [allPatternHtmlFiles],
  );

  const deploymentBuildFiles = useMemo(
    () =>
      Object.entries(deploymentBuildRawAssetMap)
        .map(([path, assetUrl]) => {
          const fromNamedFolder = path.startsWith("../model/部署构建文件/");
          const rootPrefix = fromNamedFolder
            ? "../model/部署构建文件/"
            : "../model/部署构建/";
          const relativePath = path.startsWith(rootPrefix)
            ? path.slice(rootPrefix.length)
            : path;
          return {
            path,
            relativePath,
            fromNamedFolder,
            rawContent: String(assetUrl),
          };
        })
        .filter((item) => item.relativePath && item.rawContent.length > 0)
        .sort((a, b) => {
          if (a.fromNamedFolder !== b.fromNamedFolder) {
            return a.fromNamedFolder ? -1 : 1;
          }
          return a.relativePath.localeCompare(b.relativePath, "zh-CN");
        })
        .reduce((acc, item) => {
          if (!acc.some((entry) => entry.relativePath === item.relativePath)) {
            acc.push(item);
          }
          return acc;
        }, []),
    [deploymentBuildRawAssetMap],
  );

  const getFrameSourceProps = (item) => {
    const srcDoc = sanitizedHtmlByPath[item.path];
    if (srcDoc) {
      return { srcDoc };
    }
    return { src: item.url };
  };

  const rankedPatternHtmlFiles = useMemo(
    () =>
      allPatternHtmlFiles
        .map((item) => ({
          ...item,
          nodeCount: extractNodeCountFromHtml(sanitizedHtmlByPath[item.path]),
        }))
        .sort((a, b) => {
          if (b.nodeCount !== a.nodeCount) {
            return b.nodeCount - a.nodeCount;
          }
          return sortByFileName(a, b);
        }),
    [allPatternHtmlFiles, sanitizedHtmlByPath],
  );

  const matchedTotalPages = Math.max(
    1,
    Math.ceil(matchedSubgraphs.length / MATCHED_SUBGRAPHS_PER_PAGE),
  );
  const pagedMatchedSubgraphs = useMemo(() => {
    const start = (matchedPage - 1) * MATCHED_SUBGRAPHS_PER_PAGE;
    return matchedSubgraphs.slice(start, start + MATCHED_SUBGRAPHS_PER_PAGE);
  }, [matchedPage, matchedSubgraphs]);

  useEffect(() => {
    let isDisposed = false;

    const loadAndSanitizeHtml = async () => {
      const entries = await Promise.all(
        allPatternHtmlFiles.map(async (item) => {
          try {
            const response = await fetch(item.url);
            if (!response.ok) {
              return [item.path, ""];
            }
            const rawHtml = await response.text();
            return [item.path, sanitizeGraphHtml(rawHtml)];
          } catch (error) {
            return [item.path, ""];
          }
        }),
      );

      if (!isDisposed) {
        setSanitizedHtmlByPath(Object.fromEntries(entries));
      }
    };

    loadAndSanitizeHtml();
    return () => {
      isDisposed = true;
    };
  }, [allPatternHtmlFiles]);

  const clearRecognizeTimers = () => {
    if (recognizeIntervalRef.current) {
      window.clearInterval(recognizeIntervalRef.current);
      recognizeIntervalRef.current = null;
    }
    if (recognizeTimeoutRef.current) {
      window.clearTimeout(recognizeTimeoutRef.current);
      recognizeTimeoutRef.current = null;
    }
  };

  useEffect(() => clearRecognizeTimers, []);

  const handleChooseFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const content = await file.text();
      const parsed = JSON.parse(content);
      if (!hasValidPatternJsonShape(parsed)) {
        throw new Error("INVALID_PATTERN_JSON");
      }

      clearRecognizeTimers();
      setImportedFileName(file.name);
      setImportSucceed(true);
      setUploadErrorVisible(false);
      setImportStatus("预定义任务模式导入成功");
      setRecognized(false);
      setMatchedSubgraphs([]);
      setMatchedPage(1);
      setRecognizeProgress(0);
      setIsRecognizing(false);
      setIsDownloadingBuildZip(false);
      setRecognizeStatus("待识别");
      setDownloadStatus("识别完成后可下载部署构建压缩包");
    } catch (error) {
      clearRecognizeTimers();
      setImportedFileName(file.name);
      setImportSucceed(false);
      setUploadErrorVisible(true);
      const uploadErrorMessage =
        error instanceof Error && error.message === "INVALID_PATTERN_JSON"
          ? "导入失败：缺少节点或边信息，请重新上传"
          : "导入失败：文件不是合法 JSON，请重新选择";
      setImportStatus(uploadErrorMessage);
      setRecognized(false);
      setMatchedSubgraphs([]);
      setMatchedPage(1);
      setRecognizeProgress(0);
      setIsRecognizing(false);
      setIsDownloadingBuildZip(false);
      setRecognizeStatus("待识别");
      setDownloadStatus("识别完成后可下载部署构建压缩包");
    } finally {
      event.target.value = "";
    }
  };

  const handleRecognize = () => {
    if (!importSucceed) {
      setRecognizeStatus("请先导入预定义任务模式 JSON 文件");
      setRecognized(false);
      return;
    }
    if (isRecognizing) {
      return;
    }

    clearRecognizeTimers();
    setIsRecognizing(true);
    setRecognized(false);
    setMatchedSubgraphs([]);
    setMatchedPage(1);
    setRecognizeProgress(0);
    setIsDownloadingBuildZip(false);
    setRecognizeStatus("模式识别中...");
    setDownloadStatus("识别完成后可下载部署构建压缩包");
    requestPatternRecognitionCache(importedFileName).catch((error) => {
      console.warn(error);
    });

    recognizeIntervalRef.current = window.setInterval(() => {
      setRecognizeProgress((prev) => {
        const next = Math.min(94, prev + Math.floor(Math.random() * 12) + 4);
        return next;
      });
    }, 100);

    recognizeTimeoutRef.current = window.setTimeout(() => {
      clearRecognizeTimers();

      const selected = rankedPatternHtmlFiles
        .filter((item) => item.baseName !== "43")
        .map((item, index) => ({
          ...item,
          displayName: `子图 ${index + 1}`,
        }));

      setMatchedSubgraphs(selected);
      setMatchedPage(1);
      setRecognizeProgress(100);
      setRecognized(true);
      setIsRecognizing(false);
      setRecognizeStatus(`识别完成：匹配到${selected.length}个相同模式子图`);
      setDownloadStatus(
        deploymentBuildFiles.length
          ? "识别完成，可下载部署构建压缩包"
          : "识别完成，但未找到可下载的部署构建文件",
      );
    }, 1600);
  };

  const handleDownloadBuildZip = async () => {
    if (!recognized || isRecognizing || isDownloadingBuildZip) {
      return;
    }
    if (!deploymentBuildFiles.length) {
      setDownloadStatus("下载失败：未找到部署构建目录文件");
      return;
    }

    setIsDownloadingBuildZip(true);
    setDownloadStatus("正在打包部署构建文件...");

    try {
      const encoder = new TextEncoder();
      const filesForZip = deploymentBuildFiles.map((item) => ({
        name: `部署构建/${item.relativePath}`,
        data: encoder.encode(item.rawContent),
      }));

      const zipBlob = createZipBlob(filesForZip);
      const fileName = `部署构建_${getTimestamp()}.zip`;
      const downloadUrl = URL.createObjectURL(zipBlob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(downloadUrl);
      setDownloadStatus(`下载已开始：${fileName}`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "未知错误，请稍后重试";
      setDownloadStatus(`下载失败：${message}`);
    } finally {
      setIsDownloadingBuildZip(false);
    }
  };

  return (
    <section className="pattern-recognition-wrap">
      <article className="panel wide pattern-recognition-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Pattern Recognition</p>
            <h3>任务模式识别</h3>
          </div>
        </div>

        <div className="pattern-main-grid">
          <div className="pattern-left-steps">
            <div className="pattern-step-block">
              <div className="pattern-step-head">
                <strong>步骤 1：导入预定义任务模式</strong>
              </div>
              <div className="pattern-import-row">
                <input
                  ref={fileInputRef}
                  className="pattern-hidden-input"
                  type="file"
                  accept=".json,application/json"
                  onChange={handleFileChange}
                />
                <button
                  type="button"
                  className="pattern-action-btn"
                  onClick={handleChooseFile}
                >
                  上传 JSON 文件
                </button>
                <span className="pattern-status">{importStatus}</span>
              </div>
              <p className="pattern-file-name">
                {importedFileName
                  ? `已选择文件：${importedFileName}`
                  : "未选择文件"}
              </p>
              {uploadErrorVisible ? (
                <Alert
                  style={{ marginTop: 12 }}
                  message="请上传正确格式的数据"
                  type="error"
                  showIcon
                  closable
                  onClose={() => setUploadErrorVisible(false)}
                />
              ) : null}
            </div>

            <div className="pattern-step-block">
              <div className="pattern-step-head">
                <strong>步骤 2：导入模式子图展示</strong>
              </div>
              {importSucceed && preview43 ? (
                <div className="imported-subgraph-card">
                  <div className="matched-subgraph-head">
                    <span>导入模式子图</span>
                  </div>
                  <iframe
                    className="matched-subgraph-frame"
                    title="imported-pattern-subgraph-43"
                    {...getFrameSourceProps(preview43)}
                  />
                </div>
              ) : (
                <div className="pattern-placeholder-box">
                  请先导入预定义任务模式
                </div>
              )}
            </div>

            <div className="pattern-step-block">
              <div className="pattern-step-head">
                <strong>步骤 3：任务模式识别</strong>
              </div>
              <div className="pattern-recognize-row">
                <button
                  type="button"
                  className="pattern-action-btn recognize-btn"
                  onClick={handleRecognize}
                  disabled={isRecognizing}
                >
                  {isRecognizing ? "识别中..." : "任务模式识别"}
                </button>
                <span className="pattern-status">{recognizeStatus}</span>
              </div>
              <div className="pattern-progress-track">
                <div
                  className="pattern-progress-fill"
                  style={{ width: `${recognizeProgress}%` }}
                />
              </div>
            </div>
          </div>

          <div className="pattern-right-result">
            <div className="pattern-result-shell">
              <div className="pattern-result-head">
                <h4>识别结果</h4>
                <button
                  type="button"
                  className="pattern-action-btn download-btn"
                  onClick={handleDownloadBuildZip}
                  disabled={
                    !recognized || isRecognizing || isDownloadingBuildZip
                  }
                >
                  {isDownloadingBuildZip ? "打包中..." : "下载所有模式识别结果"}
                </button>
              </div>
              {!recognized ? (
                <p className="pattern-result-hint">
                  点击“任务模式识别”后，右侧展示匹配子图结果。
                </p>
              ) : (
                <div className="pattern-result-grid single-column">
                  <div className="pattern-result-card">
                    <strong>部署构建模式子图</strong>
                    <div className="matched-subgraph-pagination">
                      <button
                        type="button"
                        className="pattern-action-btn pagination-btn"
                        onClick={() =>
                          setMatchedPage((prev) => Math.max(1, prev - 1))
                        }
                        disabled={matchedPage <= 1}
                      >
                        上一页
                      </button>
                      <span className="matched-page-indicator">
                        当前页 {matchedPage} / {matchedTotalPages} · 总共{" "}
                        {matchedSubgraphs.length}
                      </span>
                      <button
                        type="button"
                        className="pattern-action-btn pagination-btn"
                        onClick={() =>
                          setMatchedPage((prev) =>
                            Math.min(matchedTotalPages, prev + 1),
                          )
                        }
                        disabled={matchedPage >= matchedTotalPages}
                      >
                        下一页
                      </button>
                    </div>
                    <div className="matched-subgraph-list">
                      {pagedMatchedSubgraphs.map((item) => (
                        <div className="matched-subgraph-card" key={item.path}>
                          <div className="matched-subgraph-head">
                            <span>{item.displayName}</span>
                          </div>
                          <iframe
                            className="matched-subgraph-frame"
                            title={`matched-subgraph-${item.displayName}`}
                            {...getFrameSourceProps(item)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </article>
    </section>
  );
}
