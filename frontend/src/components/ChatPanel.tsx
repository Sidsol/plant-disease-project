import { useState, useRef, useEffect, useCallback } from "react";
import { sendChatMessage, fetchChatStatus, fetchChatHistory } from "../api/client";
import type { ChatMessage, DiagnosisResponse, TreatmentResponse } from "../types";
import TreatmentTips from "./TreatmentTips";

interface Props {
  diagnosis: DiagnosisResponse | null;
  treatment?: TreatmentResponse | null;
}
const SUGGESTED_QUESTIONS_DISEASE = [
  "How do I treat this organically?",
  "What causes this condition?",
  "Can it spread to other plants?",
];

const SUGGESTED_QUESTIONS_HEALTHY = [
  "How do I keep my plant healthy?",
  "What common diseases should I watch for?",
  "What's the best watering schedule?",
];

export default function ChatPanel({ diagnosis, treatment }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [ollamaAvailable, setOllamaAvailable] = useState<boolean | null>(null);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const prevScanIdRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    fetchChatStatus()
      .then((s) => setOllamaAvailable(s.available))
      .catch(() => setOllamaAvailable(false));
  }, []);

  useEffect(() => {
    const currentScanId = diagnosis?.scan_id;
    if (currentScanId && currentScanId !== prevScanIdRef.current) {
      prevScanIdRef.current = currentScanId;
      setMessages([]);
      setSessionId(undefined);
      fetchChatHistory(currentScanId).then((msgs) => {
        if (msgs.length > 0) setMessages(msgs);
      });
    }
  }, [diagnosis?.scan_id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(
    async (text: string) => {
      if (!text.trim() || streaming) return;
      const userMsg: ChatMessage = { role: "user", content: text.trim() };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setStreaming(true);

      const assistantMsg: ChatMessage = { role: "assistant", content: "" };
      setMessages((prev) => [...prev, assistantMsg]);

      try {
        await sendChatMessage(
          {
            message: text.trim(),
            scan_id: diagnosis?.scan_id,
            session_id: sessionId,
            history: messages,
          },
          (token) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last.role === "assistant") {
                updated[updated.length - 1] = { ...last, content: last.content + token };
              }
              return updated;
            });
          },
          (sid) => { if (sid) setSessionId(sid); },
          (error) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last.role === "assistant") {
                updated[updated.length - 1] = { ...last, content: `Error: ${error}` };
              }
              return updated;
            });
          }
        );
      } catch (e) {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = {
              ...last,
              content: `Error: ${e instanceof Error ? e.message : "Failed to get response"}`,
            };
          }
          return updated;
        });
      } finally {
        setStreaming(false);
      }
    },
    [streaming, diagnosis?.scan_id, sessionId, messages]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      send(input);
    }
  };

  const top = diagnosis?.prediction;
  const isHealthy = top?.healthy;
  const suggestions = isHealthy ? SUGGESTED_QUESTIONS_HEALTHY : SUGGESTED_QUESTIONS_DISEASE;

  return (
    <div className="chat-page">
      {/* Diagnosis summary card */}
      {diagnosis && top && (
        <div className="diag-summary">
          <div className="diag-summary-decor" />
          <div className="diag-summary-content">
            {diagnosis.attention_map && (
              <div className="diag-thumb">
                <img
                  src={`data:image/jpeg;base64,${diagnosis.attention_map}`}
                  alt="Scan thumbnail"
                />
              </div>
            )}
            <div>
              <p className={`diag-summary-label ${isHealthy ? "healthy" : ""}`}>
                {isHealthy ? "Healthy Specimen" : "Diagnosis Detected"}
              </p>
              <h2 className="diag-summary-title">
                {isHealthy ? `${top.plant} — Healthy` : `${top.condition} Detected`}
              </h2>
              <p className="diag-summary-desc">
                {isHealthy
                  ? "Your plant appears healthy. Ask about care tips below."
                  : `${top.plant} — ${diagnosis.confidence_percentage}% confidence`}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Treatment tips (inline, from mockup) */}
      {treatment && (
        <TreatmentTips tips={treatment.tips} healthy={treatment.healthy} />
      )}

      {/* Ollama status */}
      {ollamaAvailable === false && (
        <div className="ollama-status error">
          LLM unavailable — ensure Ollama is running (<code>ollama serve</code>)
        </div>
      )}

      {/* Chat header */}
      <div className="chat-diag-banner">
        <div className="flex">
          <div className="chat-diag-icon">
            <span className="material-symbols-outlined">forum</span>
          </div>
          <span className="chat-diag-title">Plant Care Assistant</span>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <>
            {/* Initial assistant greeting */}
            <div className="chat-msg assistant">
              <div className="chat-msg-bubble">
                {diagnosis && top
                  ? `Hello! I've analyzed your diagnosis. How can I help you manage this ${isHealthy ? "healthy plant" : top.condition} today?`
                  : "Hello! Upload and classify a plant leaf first, or ask me any plant care question."}
              </div>
              <span className="chat-msg-meta">Assistant &middot; Now</span>
            </div>
            {/* Suggested questions */}
            <div className="chat-suggestions">
              {suggestions.map((q, i) => (
                <button
                  key={i}
                  className="chat-suggestion-btn"
                  onClick={() => send(q)}
                  disabled={streaming || ollamaAvailable === false}
                >
                  {q}
                </button>
              ))}
            </div>
          </>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            <div className="chat-msg-bubble">
              {msg.content || (streaming && i === messages.length - 1 ? (
                <span className="chat-typing">Thinking&hellip;</span>
              ) : null)}
            </div>
            {msg.role === "assistant" && (
              <span className="chat-msg-meta">
                Assistant {streaming && i === messages.length - 1 ? "· typing..." : ""}
              </span>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <input
          ref={inputRef}
          className="chat-input"
          type="text"
          placeholder={ollamaAvailable === false ? "LLM unavailable" : "Ask your botanical expert..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={streaming || ollamaAvailable === false}
        />
        <button
          className="chat-send-btn"
          onClick={() => send(input)}
          disabled={!input.trim() || streaming || ollamaAvailable === false}
          aria-label="Send message"
        >
          <span className="material-symbols-outlined">send</span>
        </button>
      </div>
    </div>
  );
}
