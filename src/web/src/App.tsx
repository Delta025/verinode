import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import VerifyPage from './components/VerifyPage'
import NodeList from './components/NodeList'
import GraphView from './components/GraphView'
import IngestForm from './components/IngestForm'
import QueryInterface from './components/QueryInterface'
import HypothesisTest from './components/HypothesisTest'
import AnomalyList from './components/AnomalyList'
import AnomalyReview from './components/AnomalyReview'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="verify" element={<VerifyPage />} />
          <Route path="nodes" element={<NodeList />} />
          <Route path="graph" element={<GraphView />} />
          <Route path="ingest" element={<IngestForm />} />
          <Route path="query" element={<QueryInterface />} />
          <Route path="hypothesis" element={<HypothesisTest />} />
          <Route path="anomalies" element={<AnomalyList />} />
          <Route path="review" element={<AnomalyReview />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
