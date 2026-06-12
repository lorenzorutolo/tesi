import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const CSV_REPO_PATH = path.resolve(__dirname, '..', 'risultati_benchmark.csv')
const CSV_URL = '/risultati_benchmark.csv'

// Serve il CSV direttamente dalla root del repo (sia in dev che in build),
// cosi' non serve duplicare il file in frontend/public.
function serveRepoCsv() {
  return {
    name: 'serve-csv-from-repo-root',
    configureServer(server) {
      server.middlewares.use(CSV_URL, (req, res) => {
        if (!fs.existsSync(CSV_REPO_PATH)) {
          res.statusCode = 404
          res.end(`CSV non trovato: ${CSV_REPO_PATH}`)
          return
        }
        res.setHeader('Content-Type', 'text/csv; charset=utf-8')
        res.setHeader('Cache-Control', 'no-store')
        fs.createReadStream(CSV_REPO_PATH).pipe(res)
      })
    },
    closeBundle() {
      if (fs.existsSync(CSV_REPO_PATH)) {
        const distDir = path.resolve(__dirname, 'dist')
        fs.mkdirSync(distDir, { recursive: true })
        fs.copyFileSync(CSV_REPO_PATH, path.join(distDir, 'risultati_benchmark.csv'))
        console.log('[serve-csv-from-repo-root] CSV copiato in dist/')
      } else {
        console.warn(`[serve-csv-from-repo-root] CSV non trovato: ${CSV_REPO_PATH}`)
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), serveRepoCsv()],
  server: {
    port: 5173,
    open: true,
  },
})
