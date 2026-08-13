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
      // uvicorn graph_ui.backend.main:app --port 8002) so the frontend can
      // call same-origin `/api/...` with no CORS setup needed in dev.
      // Port 8002, not 8001 -- 8001 got stuck with an orphaned listening
      // socket during development that no tool (netstat/taskkill/
      // Get-Process) could attach an owning process to or kill.
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
      },
    },
  },
})
