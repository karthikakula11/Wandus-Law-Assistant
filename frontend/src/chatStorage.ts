export type StoredMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: unknown[];
  source?: "documents" | "general";
};

const STORAGE_KEY = "pintu-chat-v2";
const LEGACY_KEY = "pintu-chat-v1";

/** Must match App welcome copy (single source). */
export const WELCOME_CONTENT =
  "Hi! I'm **Pintu** — I only help with **law and legal study**, including questions about **your uploaded laws** from **Upload file** when they match your question.\n\n" +
  "`I can speed-read your PDFs — I can't represent you in court. Not legal advice.`";

/** Keep UI history bounded per thread (browser quota). */
const MAX_STORED_MESSAGES = 120;
/** Must match backend `MAX_HISTORY_MESSAGES` for what we send to the API */
export const MAX_HISTORY_FOR_API = 24;
const MAX_THREADS = 50;

export type ChatThread = {
  id: string;
  title: string;
  updatedAt: string;
  messages: StoredMessage[];
  activeDocumentId?: string | null;
};

export type ThreadsState = {
  v: 2;
  threads: ChatThread[];
  activeThreadId: string;
};

function trimMessages(messages: StoredMessage[]): StoredMessage[] {
  if (messages.length <= MAX_STORED_MESSAGES) return messages;
  const welcome = messages[0]?.id === "welcome" ? messages[0] : null;
  const rest = welcome ? messages.slice(1) : messages;
  const budget = MAX_STORED_MESSAGES - (welcome ? 1 : 0);
  const tail = rest.slice(-budget);
  return welcome ? [welcome, ...tail] : tail;
}

function welcomeStored(): StoredMessage {
  return { id: "welcome", role: "assistant", content: WELCOME_CONTENT };
}

/** Fallback title before LLM naming (first line, ~8 words, trimmed). */
export function threadTitleFromFirstUser(text: string): string {
  const raw = text.replace(/\s+/g, " ").trim();
  if (!raw) return "New chat";
  const firstLine = raw.split(/[.!?\n]/)[0]?.trim() || raw;
  const noQ = firstLine.replace(/\?+$/, "").trim();
  const words = noQ.split(/\s+/).filter(Boolean).slice(0, 9).join(" ");
  if (!words) return "New chat";
  return words.length > 56 ? `${words.slice(0, 54)}…` : words;
}

export function createEmptyThread(): ChatThread {
  const id =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `th-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  return {
    id,
    title: "New chat",
    updatedAt: new Date().toISOString(),
    messages: [welcomeStored()],
    activeDocumentId: null,
  };
}

function migrateLegacyV1(raw: string): ThreadsState | null {
  try {
    const data = JSON.parse(raw) as {
      messages?: StoredMessage[];
      activeDocumentId?: string | null;
    };
    if (!Array.isArray(data.messages) || data.messages.length === 0) {
      return null;
    }
    const cleaned = trimMessages(
      data.messages.filter((m) => m?.id && m?.role && m?.content)
    );
    const id =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `mig-${Date.now()}`;
    const fu = cleaned.find((m) => m.role === "user");
    const title = fu ? threadTitleFromFirstUser(fu.content) : "Imported chat";
    return {
      v: 2,
      threads: [
        {
          id,
          title,
          updatedAt: new Date().toISOString(),
          messages: cleaned,
          activeDocumentId:
            typeof data.activeDocumentId === "string" ? data.activeDocumentId : null,
        },
      ],
      activeThreadId: id,
    };
  } catch {
    return null;
  }
}

export function loadThreadsState(): ThreadsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw) as ThreadsState;
      if (data.v === 2 && Array.isArray(data.threads) && data.threads.length > 0) {
        const active =
          data.threads.some((t) => t.id === data.activeThreadId) && data.activeThreadId
            ? data.activeThreadId
            : data.threads[0].id;
        return {
          v: 2,
          threads: data.threads.map((t) => ({
            ...t,
            messages: trimMessages(
              (t.messages ?? []).filter((m) => m?.id && m?.role && m?.content)
            ),
          })),
          activeThreadId: active,
        };
      }
    }
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const migrated = migrateLegacyV1(legacy);
      if (migrated) {
        saveThreadsState(migrated);
        try {
          localStorage.removeItem(LEGACY_KEY);
        } catch {
          /* ignore */
        }
        return migrated;
      }
    }
  } catch {
    /* ignore */
  }
  const t = createEmptyThread();
  return { v: 2, threads: [t], activeThreadId: t.id };
}

export function saveThreadsState(state: ThreadsState): void {
  try {
    const threads = state.threads
      .slice(0, MAX_THREADS)
      .map((t) => ({
        ...t,
        messages: trimMessages(t.messages),
      }));
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ v: 2, threads, activeThreadId: state.activeThreadId })
    );
  } catch {
    try {
      const st = loadThreadsState();
      if (st.threads.length > 1) {
        const tail = st.threads.slice(-20);
        saveThreadsState({ ...st, threads: tail });
      }
    } catch {
      /* ignore */
    }
  }
}

export function sortThreadsRecents(threads: ChatThread[]): ChatThread[] {
  return [...threads].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );
}
