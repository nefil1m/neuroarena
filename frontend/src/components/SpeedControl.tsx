import { useEffect, useState } from 'react'
import { getSpeed, setSpeed } from '../api'

export function SpeedControl() {
  const [presets, setPresets] = useState<string[]>([])
  const [preset, setPresetState] = useState('max')
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getSpeed()
      .then((speed) => {
        if (cancelled) return
        setPresets(speed.presets)
        setPresetState(speed.preset)
      })
      .catch((err) => {
        if (!cancelled) setMessage(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function choose(next: string) {
    const previous = preset
    setPresetState(next)
    setMessage(null)
    try {
      await setSpeed(next)
    } catch (err) {
      setPresetState(previous)
      setMessage(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div>
      <h2>Speed</h2>
      <label>
        Simulation speed
        <select value={preset} onChange={(e) => choose(e.target.value)}>
          {presets.map((p) => (
            <option key={p} value={p}>
              {p === 'max' ? 'Max (unpaced)' : p}
            </option>
          ))}
        </select>
      </label>
      <p>1x is real time. Slower speeds pace training so it is watchable, and make it take longer.</p>
      {message && <p role="alert">{message}</p>}
    </div>
  )
}
