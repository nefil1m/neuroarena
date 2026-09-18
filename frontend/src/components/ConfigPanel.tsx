import { useEffect, useState } from 'react'
import { getPollInterval, setPollInterval, stopRun, updateConfig } from '../api'

export function ConfigPanel() {
  const [maxGenerationSteps, setMaxGenerationSteps] = useState('')
  const [maxGenerations, setMaxGenerations] = useState('')
  const [targetFitness, setTargetFitness] = useState('')
  const [pollDraft, setPollDraft] = useState('200')
  const [acceptedPollMs, setAcceptedPollMs] = useState(200)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    getPollInterval().then((r) => {
      setPollDraft(String(r.interval_ms))
      setAcceptedPollMs(r.interval_ms)
    })
  }, [])

  async function applyConfig(e: React.FormEvent) {
    e.preventDefault()
    setMessage(null)
    const partial: Record<string, number | null> = {}
    if (maxGenerationSteps !== '') partial.max_generation_steps = Number(maxGenerationSteps)
    if (maxGenerations !== '') partial.max_generations = Number(maxGenerations)
    if (targetFitness !== '') partial.target_fitness = Number(targetFitness)
    try {
      await updateConfig(partial)
      setMessage('Applied — takes effect at the next generation boundary.')
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  async function applyPollInterval(draft: string) {
    setPollDraft(draft)
    const value = Number(draft)
    if (draft === '' || !Number.isInteger(value) || value < 1) return
    try {
      await setPollInterval(value)
      setAcceptedPollMs(value)
    } catch (err) {
      setPollDraft(String(acceptedPollMs))
      setMessage(String(err))
    }
  }

  async function handleStop() {
    try {
      await stopRun()
      setMessage('Stop requested — the run will end after its current generation.')
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div>
      <h2>Controls</h2>
      <form onSubmit={applyConfig}>
        <label>
          Max generation steps
          <input
            type="number"
            min={1}
            value={maxGenerationSteps}
            onChange={(e) => setMaxGenerationSteps(e.target.value)}
          />
        </label>
        <label>
          Max generations
          <input
            type="number"
            min={1}
            value={maxGenerations}
            onChange={(e) => setMaxGenerations(e.target.value)}
          />
        </label>
        <label>
          Target fitness
          <input
            type="number"
            step="any"
            value={targetFitness}
            onChange={(e) => setTargetFitness(e.target.value)}
          />
        </label>
        <button type="submit">Apply</button>
      </form>
      <label>
        Poll interval (ms)
        <input
          type="number"
          min={1}
          value={pollDraft}
          onChange={(e) => applyPollInterval(e.target.value)}
        />
      </label>
      <button onClick={handleStop}>Stop run</button>
      {message && <p>{message}</p>}
    </div>
  )
}
