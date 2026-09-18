import { useState } from 'react'
import { ModelList } from './components/ModelList'
import { LiveMetrics } from './components/LiveMetrics'
import { NewRunForm } from './components/NewRunForm'
import { ConfigPanel } from './components/ConfigPanel'
import { SpeedControl } from './components/SpeedControl'
import { ViewerPanel } from './components/ViewerPanel'

function App() {
  const [resumeModelId, setResumeModelId] = useState<string | null>(null)
  const [runKey, setRunKey] = useState(0) // bump to force children to re-fetch after a run starts

  return (
    <div>
      <h1>neuroarena dashboard</h1>
      <NewRunForm
        key={`${runKey}-${resumeModelId ?? 'new'}`} // remount: fresh form when entering/leaving resume mode
        resumeModelId={resumeModelId}
        onStarted={() => setRunKey((k) => k + 1)}
      />
      {resumeModelId !== null && (
        <button type="button" onClick={() => setResumeModelId(null)}>
          New run
        </button>
      )}
      <LiveMetrics />
      <ConfigPanel />
      <SpeedControl />
      <ViewerPanel />
      <h2>Saved models</h2>
      <ModelList key={runKey} onResume={setResumeModelId} />
    </div>
  )
}

export default App
