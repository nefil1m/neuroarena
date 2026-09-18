import { useEffect, useState } from 'react'
import { listModels } from '../api'
import type { Model } from '../types'

export function ModelList({
  onResume,
}: {
  onResume: (modelId: string) => void
}) {
  const [models, setModels] = useState<Model[]>([])

  useEffect(() => {
    listModels().then(setModels).catch(console.error)
  }, [])

  if (models.length === 0) return <p>No saved models yet.</p>

  return (
    <ul>
      {models.map((m) => (
        <li key={m.model_id}>
          {m.model_id} ({m.backend}, created {m.created_at}){' '}
          <button onClick={() => onResume(m.model_id)}>Resume</button>
        </li>
      ))}
    </ul>
  )
}
