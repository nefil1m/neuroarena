import { useState } from 'react'
import { ModelList } from './components/ModelList'
import { NewRunForm } from './components/NewRunForm'

function App() {
  const [resumeModelId, setResumeModelId] = useState<string | null>(null)
  const [runKey, setRunKey] = useState(0) // bump to force children to re-fetch after a run starts

  return (
    <div>
      <h1>neuroarena dashboard</h1>
      <NewRunForm
        key={runKey}
        resumeModelId={resumeModelId}
        onStarted={() => setRunKey((k) => k + 1)}
      />
      <h2>Saved models</h2>
      <ModelList onResume={setResumeModelId} />
    </div>
  )
}

export default App
