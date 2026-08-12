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
      // uvicorn graph_ui.backend.main:app --port 8001) so the frontend can
      // call same-origin `/api/...` with no CORS setup needed in dev.
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
