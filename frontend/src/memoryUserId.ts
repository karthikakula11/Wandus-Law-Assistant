/** Stable opaque id for server-side long-term memory (localStorage + optional DB primary user). */

const MEMORY_USER_KEY = "wandus-memory-user-id";
const LEGACY_MEMORY_USER_KEY = "pintu-memory-user-id";
const API = "/api";

const KEY_RE = /^[a-zA-Z0-9_-]{8,64}$/;

/**
 * Prefer existing localStorage id; else fetch seeded ``GET /users/primary`` and use
 * ``memory_user_key``; else generate a random id (anonymous browser).
 */
export async function resolveMemoryUserId(): Promise<string> {
  try {
    let existing = localStorage.getItem(MEMORY_USER_KEY);
    if (!existing || !KEY_RE.test(existing)) {
      const legacy = localStorage.getItem(LEGACY_MEMORY_USER_KEY);
      if (legacy && KEY_RE.test(legacy)) {
        try {
          localStorage.setItem(MEMORY_USER_KEY, legacy);
          localStorage.removeItem(LEGACY_MEMORY_USER_KEY);
        } catch {
          /* ignore */
        }
        existing = legacy;
      }
    }
    if (existing && KEY_RE.test(existing)) {
      return existing;
    }
  } catch {
    /* private mode */
  }

  try {
    const r = await fetch(`${API}/users/primary`);
    if (r.ok) {
      const u = (await r.json()) as { memory_user_key?: string };
      const k = u.memory_user_key?.trim();
      if (k && KEY_RE.test(k)) {
        try {
          localStorage.setItem(MEMORY_USER_KEY, k);
        } catch {
          /* ignore */
        }
        return k;
      }
    }
  } catch {
    /* offline / no API */
  }

  try {
    const nid = crypto.randomUUID().replace(/-/g, "");
    localStorage.setItem(MEMORY_USER_KEY, nid);
    return nid;
  } catch {
    const fb = `u${Date.now().toString(36)}${Math.random().toString(36).slice(2, 14)}`;
    return fb.length >= 8 ? fb.slice(0, 32) : `${fb}xxxxxxxx`.slice(0, 16);
  }
}
