import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    // Tutto cio' che il browser chiede a /api/... viene inoltrato al server
    // Flask su :8000. Cosi' il browser parla con un'unica origine (Vite) e non
    // ci sono problemi di CORS.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
