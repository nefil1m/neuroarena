import { useEffect, useState } from 'react'
import { getModelSettings, listTracks, startRun } from '../api'
import type { Track } from '../types'

export function NewRunForm({
  resumeModelId,
  onStarted,
}: {
  resumeModelId: string | null
  onStarted: () => void
}) {
  const [tracks, setTracks] = useState<Track[]>([])
  const [trackId, setTrackId] = useState('')
  const [populationSize, setPopulationSize] = useState<number | ''>('')
  const [maxGenerations, setMaxGenerations] = useState<number | ''>('')
  const [targetFitness, setTargetFitness] = useState<number | ''>('')
  const [error, setError] = useState<string | null>(null)
  const [currentGeneration, setCurrentGeneration] = useState<number | null>(null)

  useEffect(() => {
    listTracks().then(setTracks).catch(console.error)
  }, [])

  useEffect(() => {
    if (resumeModelId === null) {
      setCurrentGeneration(null)
      return
    }
    let cancelled = false
    getModelSettings(resumeModelId)
      .then((settings) => {
        if (cancelled) return
        if (typeof settings.population_size === 'number') setPopulationSize(settings.population_size)
        if (typeof settings.max_generations === 'number') setMaxGenerations(settings.max_generations)
        if (typeof settings.target_fitness === 'number') setTargetFitness(settings.target_fitness)
        if (typeof settings.track_id === 'string') setTrackId(settings.track_id)
        setCurrentGeneration(typeof settings.current_generation === 'number' ? settings.current_generation : null)
      })
      .catch((err) => {
        if (cancelled) return
        setError(`Could not load settings for ${resumeModelId}: ${err instanceof Error ? err.message : String(err)}`)
      })
    return () => {
      cancelled = true
    }
  }, [resumeModelId])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    try {
      await startRun({
        track_id: trackId,
        resume_model_id: resumeModelId,
        population_size: populationSize === '' ? null : populationSize,
        max_generations: maxGenerations === '' ? null : maxGenerations,
        target_fitness: targetFitness === '' ? null : targetFitness,
      })
      onStarted()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2>{resumeModelId ? `Resume ${resumeModelId}` : 'New run'}</h2>
      {currentGeneration !== null && (
        <p>
          Currently at generation {currentGeneration}.{' '}
          {maxGenerations !== '' && Number(maxGenerations) <= currentGeneration && (
            <strong>
              max_generations ({maxGenerations}) is at or below this — raise it, or the run
              will train exactly one more generation and stop immediately.
            </strong>
          )}
        </p>
      )}
      <label>
        Track
        <select value={trackId} onChange={(e) => setTrackId(e.target.value)} required>
          <option value="" disabled>
            select a track
          </option>
          {tracks.map((t) => (
            <option key={t.track_id} value={t.track_id}>
              {t.track_id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Population size
        <input
          type="number"
          min={1}
          value={populationSize}
          onChange={(e) => setPopulationSize(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
      <label>
        Max generations
        <input
          type="number"
          min={1}
          value={maxGenerations}
          onChange={(e) => setMaxGenerations(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
      <label>
        Target fitness
        <input
          type="number"
          step="any"
          value={targetFitness}
          onChange={(e) => setTargetFitness(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <button type="submit">{resumeModelId ? 'Resume' : 'Start'}</button>
    </form>
  )
}
