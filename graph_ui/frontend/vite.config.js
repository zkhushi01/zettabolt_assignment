import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174, // distinct from webapp/frontend's 5173 so both can run at once
    proxy: {
      // Forwards to the graph_ui FastAPI backend (run separately:
      // uvicorn graph_ui.backend.main:app --port 8003) so the frontend can
      // call same-origin `/api/...` with no CORS setup needed in dev.
      // Port 8003, not 8002 or 8001 -- both got stuck with an orphaned
      // listening socket during development that no tool (netstat/taskkill/
      // Get-Process) could attach an owning process to or kill. If 8003
      // ever gets stuck the same way, bump to 8004 and update here, the
      // backend's CORS allow_origins, and graph_ui/README.md.
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
    },
  },
})
