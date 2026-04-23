import type { ThreadsState } from "./chatStorage";
import { API } from "./apiBase";

/** True if any thread has at least one user message (worth syncing to the server). */
export function hasUserMessages(state: ThreadsState): boolean {
  return state.threads.some((t) =>
    (t.messages ?? []).some((m) => m.role === "user")
  );
}

export async function fetchChatState(uid: string): Promise<ThreadsState | null> {
  const r = await fetch(
    `${API}/users/chat-state?memory_user_id=${encodeURIComponent(uid)}`
  );
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as ThreadsState;
}

export async function putChatState(uid: string, state: ThreadsState): Promise<boolean> {
  const r = await fetch(
    `${API}/users/chat-state?memory_user_id=${encodeURIComponent(uid)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
    }
  );
  return r.ok;
}
