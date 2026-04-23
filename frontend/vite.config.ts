import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** Shared by dev server and `vite preview` — preview does not inherit `server.proxy` by default. */
const apiProxy = {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    // Listen on all interfaces so http://localhost:5173 and http://127.0.0.1:5173 both hit Vite.
    host: true,
    port: 5173,
    strictPort: true,
    proxy: apiProxy,
  },
  preview: {
    host: true,
    port: 4173,
    strictPort: true,
    proxy: apiProxy,
  },
});
