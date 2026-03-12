import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import ReviewStage from './pages/ReviewStage'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        {/* Navigation */}
        <nav className="sticky top-0 z-20 bg-gray-900 border-b border-gray-800 px-6 py-3">
          <div className="max-w-7xl mx-auto flex items-center gap-4">
            <div className="flex items-center gap-3 flex-shrink-0">
              <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-xs">
                AI
              </div>
              <span className="font-semibold">HITL AI Pipeline</span>
              <span className="text-xs text-gray-500 hidden sm:inline">
                MSc IT Dissertation — Shyam Kishor Pandit, UWS
              </span>
            </div>
          </div>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/review" element={<ReviewStage />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
