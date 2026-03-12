import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/runs': 'http://localhost:8000',
      '/review': 'http://localhost:8000',
      '/export': 'http://localhost:8000',
      '/providers': 'http://localhost:8000',
      '/prompts': 'http://localhost:8000',
    },
  },
})
