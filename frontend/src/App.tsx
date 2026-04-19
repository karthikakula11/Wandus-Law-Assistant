import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from "react";

import { AgentGraphPage } from "./AgentGraphPage";
import { EvalMetricsPage } from "./EvalMetricsPage";
import { UsageDashboardPage } from "./UsageDashboardPage";
import { graphNodeLabel } from "./graphLabels";
import {
  createEmptyThread,
  loadThreadsState,
  MAX_HISTORY_FOR_API,
  saveThreadsState,
  sortThreadsRecents,
  StoredMessage,
  threadTitleFromFirstUser,
  ThreadsState,
  WELCOME_CONTENT,
} from "./chatStorage";
import { fetchChatState, hasUserMessages, putChatState } from "./chatStateSync";
import { resolveMemoryUserId } from "./memoryUserId";
import { PintuMascot, type MascotMood } from "./PintuMascot";

const API = "/api";

type StreamEvent =
  | { event: "graph_step"; node?: string }
  | { event: "meta"; citations?: Citation[]; source?: string }
  | { event: "token"; text?: string }
  | { event: "done" }
  | { event: "error"; detail?: string };

async function consumeChatSse(
  body: object,
  onEvent: (ev: StreamEvent) => void
): Promise<void> {
  const r = await fetch(`${API}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  const reader = r.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (value) buf += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const line = block.trim();
      if (line.startsWith("data: ")) {
        try {
          onEvent(JSON.parse(line.slice(6)) as StreamEvent);
        } catch {
          /* ignore malformed chunk */
        }
      }
    }
    if (done) break;
  }
  const tail = buf.trim();
  if (tail.startsWith("data: ")) {
    try {
      onEvent(JSON.parse(tail.slice(6)) as StreamEvent);
    } catch {
      /* ignore */
    }
  }
}

type Citation = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  chunk_index: number;
  excerpt: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  source?: "documents" | "general";
};

type IndexedDocument = {
  id: string;
  title: string;
  created_at: string;
  chunk_count: number;
};

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content: WELCOME_CONTENT,
};

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Prior turns sent to the API for conversation memory (excludes current message). */
function buildHistoryPayload(msgs: ChatMessage[]): { role: "user" | "assistant"; content: string }[] {
  return msgs
    .filter(
      (m) =>
        m.id !== "welcome" &&
        (m.role === "user" || m.role === "assistant")
    )
    .slice(-MAX_HISTORY_FOR_API)
    .map((m) => ({ role: m.role, content: m.content }));
}

function storedToChatMessages(raw: StoredMessage[]): ChatMessage[] {
  return raw.map((m) => ({
    id: m.id,
    role: m.role,
    content: m.content,
    citations: m.citations as Citation[] | undefined,
    source: m.source,
  }));
}

export function App() {
  const [page, setPage] = useState<"chat" | "agent-graph" | "usage" | "eval">("chat");
  const [chatStore, setChatStore] = useState<ThreadsState>(() => loadThreadsState());

  const messages: ChatMessage[] = useMemo(() => {
    const t = chatStore.threads.find((x) => x.id === chatStore.activeThreadId);
    if (!t?.messages?.length) return [WELCOME];
    return storedToChatMessages(t.messages);
  }, [chatStore]);

  const setMessages = useCallback(
    (updater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
      setChatStore((prev) => {
        const tid = prev.activeThreadId;
        const thread = prev.threads.find((x) => x.id === tid);
        if (!thread) return prev;
        const current = storedToChatMessages(thread.messages);
        const next =
          typeof updater === "function"
            ? (updater as (p: ChatMessage[]) => ChatMessage[])(current)
            : updater;
        const stored: StoredMessage[] = next.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          citations: m.citations,
          source: m.source,
        }));
        let title = thread.title;
        if (title === "New chat") {
          const fu = next.find((m) => m.role === "user");
          if (fu) title = threadTitleFromFirstUser(fu.content);
        }
        const threads = prev.threads.map((th) =>
          th.id === tid
            ? { ...th, messages: stored, updatedAt: new Date().toISOString(), title }
            : th
        );
        return { ...prev, threads };
      });
    },
    []
  );
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [typing, setTyping] = useState(false);
  /** LangGraph node id while a streamed reply is in progress (SSE ``graph_step``). */
  const [graphStep, setGraphStep] = useState<string | null>(null);
  /** Ordered node ids for the in-flight run (append each ``graph_step``). */
  const [graphPathNodes, setGraphPathNodes] = useState<string[]>([]);
  /** Snapshot after the last finished chat (agentic path: plan → … → generate). */
  const [lastRunPath, setLastRunPath] = useState<string[] | null>(null);
  const graphPathRef = useRef<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [ingestStatus, setIngestStatus] = useState("");
  const [indexedDocs, setIndexedDocs] = useState<IndexedDocument[]>([]);
  const [indexedLoading, setIndexedLoading] = useState(false);
  const [indexedErr, setIndexedErr] = useState<string | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);

  const refreshIndexedDocuments = useCallback(async () => {
    setIndexedLoading(true);
    setIndexedErr(null);
    try {
      const r = await fetch(`${API}/documents`);
      if (!r.ok) throw new Error((await r.text()) || "Could not list documents");
      const j = (await r.json()) as { documents?: IndexedDocument[] };
      setIndexedDocs(j.documents ?? []);
    } catch (e) {
      setIndexedErr(e instanceof Error ? e.message : String(e));
      setIndexedDocs([]);
    } finally {
      setIndexedLoading(false);
    }
  }, []);

  const deleteIndexedDocument = useCallback(
    async (doc: IndexedDocument) => {
      if (
        !window.confirm(
          `Remove “${doc.title}” from search?\n\nThis deletes stored chunks and frees space. You can’t undo this.`
        )
      ) {
        return;
      }
      setErr(null);
      setDeletingDocId(doc.id);
      try {
        const r = await fetch(`${API}/documents/${doc.id}`, { method: "DELETE" });
        if (!r.ok) throw new Error((await r.text()) || "Could not remove file");
        setIngestStatus(`Removed “${doc.title}” from your library.`);
        await refreshIndexedDocuments();
      } catch (e) {
        setErr(String(e));
      } finally {
        setDeletingDocId(null);
      }
    },
    [refreshIndexedDocuments]
  );

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [mascotMood, setMascotMood] = useState<MascotMood>("idle");
  const mascotMoodTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** After first load from API (or failure); debounced server sync waits for this. */
  const [serverChatReady, setServerChatReady] = useState(false);
  /** Skip one debounced PUT after hydrating from server or right after uploading local snapshot. */
  const skipNextServerPutRef = useRef(false);

  const scheduleMascotIdle = useCallback(() => {
    if (mascotMoodTimerRef.current) clearTimeout(mascotMoodTimerRef.current);
    mascotMoodTimerRef.current = setTimeout(() => {
      setMascotMood("idle");
      mascotMoodTimerRef.current = null;
    }, 2800);
  }, []);

  const setMascotMoodTransient = useCallback(
    (m: MascotMood) => {
      if (mascotMoodTimerRef.current) {
        clearTimeout(mascotMoodTimerRef.current);
        mascotMoodTimerRef.current = null;
      }
      setMascotMood(m);
      if (m === "happy" || m === "sad") scheduleMascotIdle();
    },
    [scheduleMascotIdle]
  );

  useEffect(() => {
    return () => {
      if (mascotMoodTimerRef.current) clearTimeout(mascotMoodTimerRef.current);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing, busy]);

  useEffect(() => {
    if (!err) return;
    const t = window.setTimeout(() => setErr(null), 6000);
    return () => window.clearTimeout(t);
  }, [err]);

  /** Load Recents from Postgres (or push local-only history once) keyed by ``memory_user_id``. */
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const uid = await resolveMemoryUserId();
      try {
        const remote = await fetchChatState(uid);
        if (cancelled) return;
        if (remote && remote.v === 2 && Array.isArray(remote.threads) && remote.threads.length > 0) {
          setChatStore(remote);
          saveThreadsState(remote);
          skipNextServerPutRef.current = true;
        } else {
          const local = loadThreadsState();
          if (hasUserMessages(local)) {
            const ok = await putChatState(uid, local);
            if (ok) skipNextServerPutRef.current = true;
          }
        }
      } catch {
        /* offline or server error — keep local cache */
      } finally {
        if (!cancelled) setServerChatReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    saveThreadsState(chatStore);
  }, [chatStore]);

  useEffect(() => {
    if (!sidebarOpen) return;
    void refreshIndexedDocuments();
  }, [sidebarOpen, refreshIndexedDocuments]);

  /** Debounced sync of full sidebar state to the API (survives restarts). */
  useEffect(() => {
    if (!serverChatReady) return;
    if (skipNextServerPutRef.current) {
      skipNextServerPutRef.current = false;
      return;
    }
    const t = window.setTimeout(() => {
      void (async () => {
        const uid = await resolveMemoryUserId();
        await putChatState(uid, chatStore);
      })();
    }, 500);
    return () => window.clearTimeout(t);
  }, [chatStore, serverChatReady]);

  const recents = useMemo(
    () => sortThreadsRecents(chatStore.threads),
    [chatStore.threads]
  );

  function newChat() {
    if (busy) return;
    setChatStore((prev) => {
      const nt = createEmptyThread();
      const threads = [nt, ...prev.threads].slice(0, 50);
      return { v: 2, threads, activeThreadId: nt.id };
    });
    setErr(null);
  }

  function selectThread(threadId: string) {
    if (busy) return;
    setChatStore((prev) => {
      if (!prev.threads.some((t) => t.id === threadId)) return prev;
      return { ...prev, activeThreadId: threadId };
    });
  }

  function deleteThread(threadId: string, e: MouseEvent<HTMLButtonElement>) {
    e.stopPropagation();
    if (busy) return;
    if (!window.confirm("Delete this chat thread?")) return;
    setChatStore((prev) => {
      const threads = prev.threads.filter((t) => t.id !== threadId);
      if (threads.length === 0) {
        const nt = createEmptyThread();
        return { v: 2, threads: [nt], activeThreadId: nt.id };
      }
      let activeThreadId = prev.activeThreadId;
      if (activeThreadId === threadId) {
        const sorted = sortThreadsRecents(threads);
        activeThreadId = sorted[0].id;
      }
      return { v: 2, threads, activeThreadId };
    });
  }

  function clearCurrentChat() {
    setMessages([WELCOME]);
    setErr(null);
  }

  /** ChatGPT-style short title from the API after the first reply. */
  async function suggestThreadTitle(
    threadId: string,
    userMessage: string,
    assistantExcerpt: string,
    sourceDocumentTitles?: string[]
  ) {
    try {
      const payload: Record<string, unknown> = {
        user_message: userMessage.slice(0, 4000),
        assistant_message: assistantExcerpt.slice(0, 4000),
      };
      if (sourceDocumentTitles && sourceDocumentTitles.length > 0) {
        payload.source_document_titles = [...new Set(sourceDocumentTitles)].slice(0, 8);
      }
      const r = await fetch(`${API}/titles/suggest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) return;
      const data = (await r.json()) as { title?: string };
      const t = (data.title ?? "").trim();
      if (!t) return;
      setChatStore((prev) => ({
        ...prev,
        threads: prev.threads.map((th) =>
          th.id === threadId ? { ...th, title: t.slice(0, 80) } : th
        ),
      }));
    } catch {
      /* offline / ignore */
    }
  }

  async function ingestFile(file: File) {
    setErr(null);
    setBusy(true);
    setIngestStatus("");
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setErr("Only PDF files are supported. Please choose a .pdf file.");
      setBusy(false);
      return;
    }
    try {
      const fd = new FormData();
      // Use the real filename as title so retrieval scope matches what users type in chat.
      fd.append("title", file.name);
      fd.append("file", file);
      const r = await fetch(`${API}/ingest/file`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setIngestStatus(`File indexed: ${data.chunks_created} chunk(s).`);
      void refreshIndexedDocuments();
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: `File **${file.name}** is in! ${data.chunks_created} chunks indexed. Ask away!`,
        },
      ]);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function send() {
    const q = input.trim();
    if (!q || busy) return;

    const threadIdSnapshot = chatStore.activeThreadId;
    const isFirstUserTurn =
      messages.filter((m) => m.id !== "welcome" && m.role === "user").length ===
      0;

    setErr(null);
    setInput("");
    if (mascotMoodTimerRef.current) {
      clearTimeout(mascotMoodTimerRef.current);
      mascotMoodTimerRef.current = null;
    }
    setMascotMood("thinking");
    const historyPayload = buildHistoryPayload(messages);
    const userMsg: ChatMessage = { id: newId(), role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setTyping(true);
    setBusy(true);
    setGraphStep(null);
    graphPathRef.current = [];
    setGraphPathNodes([]);

    let assistantContent = "";
    let metaCitations: Citation[] | undefined;
    let metaSource: "documents" | "general" | undefined;
    let sawStreamError = false;

    try {
      await consumeChatSse(
        {
          question: q,
          top_k: 4,
          history: historyPayload.length > 0 ? historyPayload : undefined,
          memory_user_id: await resolveMemoryUserId(),
        },
        (ev) => {
          if (ev.event === "graph_step" && ev.node) {
            setGraphStep(ev.node);
            graphPathRef.current = [...graphPathRef.current, ev.node];
            setGraphPathNodes([...graphPathRef.current]);
          }
          if (ev.event === "meta") {
            metaCitations = ev.citations;
            metaSource = ev.source as "documents" | "general" | undefined;
          }
          if (ev.event === "token" && ev.text) assistantContent += ev.text;
          if (ev.event === "error") {
            sawStreamError = true;
            setErr(ev.detail ?? "Stream error");
            setMascotMoodTransient("sad");
          }
        }
      );
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: assistantContent,
          citations: metaCitations ?? [],
          source: metaSource,
        },
      ]);
      if (!sawStreamError) {
        if (assistantContent.trim()) {
          setMascotMoodTransient("happy");
        } else {
          setMascotMoodTransient("sad");
        }
      }
      if (isFirstUserTurn && assistantContent.trim()) {
        const docTitles = (metaCitations ?? [])
          .map((c) => c.document_title)
          .filter((t): t is string => Boolean(t && t.trim()));
        void suggestThreadTitle(threadIdSnapshot, q, assistantContent, docTitles);
      }
    } catch (e) {
      setErr(String(e));
      setMascotMoodTransient("sad");
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: "Oops — I couldn't reach the server. Is the API running on port 8000?",
        },
      ]);
    } finally {
      setLastRunPath(
        graphPathRef.current.length > 0 ? [...graphPathRef.current] : null
      );
      setTyping(false);
      setBusy(false);
      setGraphStep(null);
      textareaRef.current?.focus();
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  if (page === "agent-graph") {
    const pathForDisplay = busy ? graphPathNodes : lastRunPath ?? [];
    const visited = new Set(pathForDisplay);
    return (
      <AgentGraphPage
        onBack={() => setPage("chat")}
        activeStep={graphStep}
        pathNodes={pathForDisplay}
        pathStreaming={busy}
        visitedNodeIds={visited}
      />
    );
  }

  if (page === "usage") {
    return <UsageDashboardPage onBack={() => setPage("chat")} />;
  }

  if (page === "eval") {
    return <EvalMetricsPage onBack={() => setPage("chat")} />;
  }

  return (
    <div className="app-shell app-shell-with-nav">
      <nav className="chat-recents" aria-label="Recent chats">
        <button
          type="button"
          className="btn-new-chat"
          onClick={newChat}
          disabled={busy}
        >
          + New chat
        </button>
        <div className="recents-label">Recents</div>
        <ul className="recents-list">
          {recents.map((t) => (
            <li key={t.id}>
              <div
                className={`recents-row ${t.id === chatStore.activeThreadId ? "recents-row-active" : ""}`}
              >
                <button
                  type="button"
                  className="recents-item"
                  onClick={() => selectThread(t.id)}
                  disabled={busy}
                  title={t.title}
                >
                  {t.title}
                </button>
                <button
                  type="button"
                  className="recents-delete"
                  onClick={(e) => deleteThread(t.id, e)}
                  disabled={busy}
                  aria-label="Delete chat"
                  title="Delete"
                >
                  ×
                </button>
              </div>
            </li>
          ))}
        </ul>
      </nav>

      <div className="chat-main">
      <header className="app-header">
        <div className="brand">
          <PintuMascot mood={mascotMood} size="header" />
          <div className="brand-text">
            <h1>Pintu</h1>
            <p>Law assistant · gpt-4o-mini</p>
          </div>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn-ghost"
            onClick={clearCurrentChat}
            disabled={busy}
            title="Clear messages in this thread only"
          >
            Clear chat
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setPage("agent-graph")}
            disabled={busy}
            title="How the agentic RAG graph is structured"
          >
            Agent pipeline
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setPage("usage")}
            disabled={busy}
            title="Token usage and cost from Langfuse"
          >
            Usage
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setPage("eval")}
            disabled={busy}
            title="Retrieval confidence drift (KS test vs prior window)"
          >
            Drift
          </button>
          <button
            type="button"
            className="btn-ghost btn-ghost--with-icon"
            onClick={() => setSidebarOpen(true)}
            disabled={busy}
          >
            <svg
              className="btn-ghost-file-icon"
              width={18}
              height={18}
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden
            >
              <path
                d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points="14 2 14 8 20 8"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Upload file
          </button>
        </div>
      </header>

      <div className="messages-wrap">
        <div className="messages" role="log" aria-live="polite">
          {messages.map((m) => (
            <MessageRow key={m.id} msg={m} />
          ))}
          {typing && (
            <div className="msg-row assistant">
              <div className="mini-avatar mini-avatar-mascot" aria-hidden>
                <PintuMascot mood={mascotMood} size="inline" />
              </div>
              <div className="msg-body">
                <div className="bubble-meta">Pintu</div>
                {graphStep && (
                  <div className="graph-step-banner" role="status" aria-live="polite">
                    <span className="graph-step-dot" aria-hidden />
                    {graphNodeLabel(graphStep)}
                  </div>
                )}
                <div className="typing-bubble" aria-label="Pintu is typing">
                  <div className="typing-dots">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <div className="composer-inner">
            <textarea
              ref={textareaRef}
              rows={1}
              placeholder="Ask a legal question or about your uploaded laws…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={busy}
            />
            <button
              type="button"
              className="btn-send"
              onClick={() => void send()}
              disabled={busy || !input.trim()}
              aria-label="Send"
            >
              ↑
            </button>
          </div>
          <p className="disclaimer">
            Pintu answers law-related questions only; indexed text is used when relevant. Not legal advice.
          </p>
        </div>
      </div>

      {err && (
        <div className="toast-err" role="alert">
          {err}
        </div>
      )}
      </div>

      {sidebarOpen && (
        <>
          <div
            className="backdrop"
            role="presentation"
            onClick={() => setSidebarOpen(false)}
          />
          <aside className="sidebar sidebar--library" aria-label="Upload file">
            <button
              type="button"
              className="sidebar-close"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
            <div className="sidebar-hero">
              <div>
                <h2 className="sidebar-hero-title">Your library</h2>
                <p className="sidebar-hero-sub">
                  PDFs you add are searchable in chat. Mention a filename (e.g.{" "}
                  <code>contract.pdf</code>) to steer retrieval.
                </p>
              </div>
            </div>

            <section className="sidebar-panel" aria-labelledby="indexed-heading">
              <div className="sidebar-panel-head">
                <h3 id="indexed-heading" className="sidebar-panel-title">
                  Indexed files
                </h3>
                <button
                  type="button"
                  className="sidebar-refresh"
                  onClick={() => void refreshIndexedDocuments()}
                  disabled={busy || indexedLoading}
                  title="Refresh list"
                >
                  Refresh
                </button>
              </div>
              {indexedLoading && indexedDocs.length === 0 ? (
                <p className="sidebar-muted">Loading…</p>
              ) : indexedErr ? (
                <p className="sidebar-err" role="alert">
                  {indexedErr}
                </p>
              ) : indexedDocs.length === 0 ? (
                <p className="sidebar-muted">
                  No PDFs yet. Add one below — they will appear here after indexing.
                </p>
              ) : (
                <ul className="indexed-list">
                  {indexedDocs.map((d) => (
                    <li key={d.id} className="indexed-item">
                      <span className="indexed-item-icon" aria-hidden>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path
                            d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <polyline points="14 2 14 8 20 8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </span>
                      <div className="indexed-item-body">
                        <span className="indexed-item-title" title={d.title}>
                          {d.title}
                        </span>
                        <span className="indexed-item-meta">
                          {d.chunk_count} chunk{d.chunk_count === 1 ? "" : "s"} ·{" "}
                          {new Date(d.created_at).toLocaleString(undefined, {
                            dateStyle: "medium",
                            timeStyle: "short",
                          })}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="indexed-item-delete"
                        title="Remove from search"
                        aria-label={`Remove ${d.title} from index`}
                        disabled={deletingDocId !== null}
                        onClick={() => void deleteIndexedDocument(d)}
                      >
                        {deletingDocId === d.id ? "…" : "Remove"}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="sidebar-panel sidebar-panel--upload" aria-labelledby="upload-heading">
              <h3 id="upload-heading" className="sidebar-panel-title">
                Add PDF
              </h3>
              <label className="upload-dropzone">
                <input
                  id="sb-file"
                  type="file"
                  accept=".pdf,application/pdf"
                  disabled={busy}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    e.target.value = "";
                    if (f) void ingestFile(f);
                  }}
                />
                <span className="upload-dropzone-text">
                  <strong>Choose a PDF</strong>
                  <span>or drop it here (browser permitting)</span>
                </span>
              </label>
              {ingestStatus && <p className="sidebar-status">{ingestStatus}</p>}
            </section>
          </aside>
        </>
      )}
    </div>
  );
}

function MessageRow({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  const isWelcome = msg.id === "welcome";
  return (
    <div
      className={`msg-row ${isUser ? "user" : "assistant"} ${isWelcome ? "msg-row--welcome" : ""}`}
    >
      {isUser ? (
        <div className="mini-avatar" aria-hidden title="You">
          Y
        </div>
      ) : (
        <div className="mini-avatar mini-avatar-mascot" aria-hidden title="Pintu">
          <PintuMascot mood="idle" size="inline" />
        </div>
      )}
      <div className="msg-body">
        {!isUser && <div className="bubble-meta">Pintu</div>}
        {isUser && <div className="bubble-meta">You</div>}
        {!isUser && msg.source && (
          <span className={`source-pill ${msg.source}`}>
            {msg.source === "documents" ? "📚 From your uploads" : "💬 General answer"}
          </span>
        )}
        <div className={`bubble ${isWelcome ? "bubble--welcome" : ""}`}>
          <FormattedText text={msg.content} />
          {msg.citations && msg.citations.length > 0 && (
            <details className="cite-block">
              <summary>Sources ({msg.citations.length})</summary>
              <ol className="cite-list">
                {msg.citations.map((c) => (
                  <li key={c.chunk_id}>
                    <strong>{c.document_title}</strong> · chunk {c.chunk_index}
                    <div style={{ marginTop: "0.25rem", opacity: 0.9 }}>{c.excerpt}</div>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

/** ``**bold**`` and `` `inline code` `` for welcome message */
function FormattedText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code key={i} className="inline-welcome-code">
              {part.slice(1, -1)}
            </code>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
