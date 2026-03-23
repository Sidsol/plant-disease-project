import { useEffect, useState } from "react";
import { fetchHistory } from "../api/client";
import type { HistoryItem, HistoryResponse } from "../types";

interface Props {
  refreshTrigger: number;
}

function confidenceBadgeClass(confidence: number, healthy: boolean): string {
  if (healthy) return "match-badge healthy";
  if (confidence >= 90) return "match-badge high";
  return "match-badge medium";
}

export default function HistoryPanel({ refreshTrigger }: Props) {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const limit = 6;

  useEffect(() => {
    setLoading(true);
    fetchHistory(page, limit)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [page, refreshTrigger]);

  return (
    <>
      {/* Editorial Header */}
      <div className="history-header-section">
        <div className="history-header-row">
          <h2 className="history-heading">History</h2>
          {data && (
            <span className="history-count">{data.total} Records</span>
          )}
        </div>
        <div className="history-underline" />
      </div>

      {/* Content */}
      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          <p>Loading history&hellip;</p>
        </div>
      )}

      {data && data.items.length === 0 && (
        <div className="chat-welcome">
          <p className="chat-welcome-title">No scans yet</p>
          <p className="chat-welcome-sub">
            Upload and classify a plant leaf image to see your scan history here.
          </p>
        </div>
      )}

      {data && data.items.length > 0 && (
        <>
          <div className="history-list">
            {data.items.map((item: HistoryItem) => (
              <div key={item.id} className="history-item">
                <div className="history-item-inner">
                  {item.thumbnail && (
                    <div className="history-thumb">
                      <img
                        src={`data:image/jpeg;base64,${item.thumbnail}`}
                        alt={item.class_name}
                      />
                      <div className="history-thumb-overlay" />
                    </div>
                  )}
                  <div className="history-item-info">
                    <div>
                      <p className="history-plant">{item.plant}</p>
                      <h3 className="history-condition">
                        {item.healthy ? "Healthy Specimen" : item.condition}
                      </h3>
                    </div>
                    <div className="history-meta">
                      <span className={confidenceBadgeClass(item.confidence, item.healthy)}>
                        {item.confidence.toFixed(0)}% Match
                      </span>
                      <span className="history-date">
                        {new Date(item.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {data.pages > 1 && (
            <div className="pagination">
              <button
                className="page-btn"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                <span className="material-symbols-outlined">chevron_left</span>
              </button>
              {Array.from({ length: Math.min(data.pages, 5) }, (_, i) => i + 1).map((p) => (
                <button
                  key={p}
                  className={`page-btn ${p === page ? "active" : ""}`}
                  onClick={() => setPage(p)}
                >
                  {p}
                </button>
              ))}
              {data.pages > 5 && <span style={{ margin: "0 0.25rem", color: "var(--outline)" }}>&hellip;</span>}
              <button
                className="page-btn"
                disabled={page >= data.pages}
                onClick={() => setPage((p) => p + 1)}
              >
                <span className="material-symbols-outlined">chevron_right</span>
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}
