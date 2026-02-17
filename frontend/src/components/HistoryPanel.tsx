import { useEffect, useState } from "react";
import { fetchHistory } from "../api/client";
import type { HistoryItem, HistoryResponse } from "../types";

interface Props {
  refreshTrigger: number;
}

export default function HistoryPanel({ refreshTrigger }: Props) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const limit = 6;

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    fetchHistory(page, limit)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [open, page, refreshTrigger]);

  return (
    <div className="history-panel">
      <button
        className="btn btn-history"
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle scan history"
      >
        &#128203; {open ? "Hide History" : "Scan History"}
      </button>

      {open && (
        <div className="history-content">
          {loading && <p className="history-loading">Loading&hellip;</p>}
          {data && data.items.length === 0 && (
            <p className="history-empty">No scans yet.</p>
          )}
          {data && data.items.length > 0 && (
            <>
              <div className="history-grid">
                {data.items.map((item: HistoryItem) => (
                  <div
                    key={item.id}
                    className={`history-card ${item.healthy ? "healthy" : "disease"}`}
                  >
                    {item.thumbnail && (
                      <img
                        src={`data:image/jpeg;base64,${item.thumbnail}`}
                        alt={item.class_name}
                        className="history-thumb"
                      />
                    )}
                    <div className="history-info">
                      <strong>{item.plant}</strong>
                      <span className={item.healthy ? "badge-healthy" : "badge-disease"}>
                        {item.condition}
                      </span>
                      <span className="history-conf">
                        {item.confidence.toFixed(2)}%
                      </span>
                      <span className="history-date">
                        {new Date(item.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="history-pagination">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => p - 1)}
                >
                  &laquo; Prev
                </button>
                <span>
                  Page {data.page} of {data.pages}
                </span>
                <button
                  disabled={page >= data.pages}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next &raquo;
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
