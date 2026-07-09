import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { extname, join } from 'node:path'

const root = process.cwd()
const port = 5173
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.mjs': 'text/javascript' }

createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(req.url.split('?')[0])
    if (p === '/') p = '/index.html'
    const buf = await readFile(join(root, p))
    res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream' })
    res.end(buf)
  } catch {
    res.writeHead(404); res.end('not found')
  }
}).listen(port, () => console.log(`serving on http://localhost:${port}`))
