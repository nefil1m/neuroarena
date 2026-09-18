import { useEffect, useRef, useState } from 'react'

interface ProgressData {
  generation: number
  population_size: number
  active_genomes_remaining: number
  elapsed_steps: number
  step_ceiling: number
  best_fitness_so_far: number | null
}

interface GenerationData {
  progress_index: number
  best_fitness: number
  mean_fitness: number
  worst_fitness: number
  population_size: number
  champion_metrics: Record<string, number>
  sim_time: number
  wall_time: number
  schema_version: number
}

interface StatusData {
  state: 'idle' | 'running' | 'completed' | 'crashed' | 'stopped'
  detail: string | null
}

type Envelope =
  | { type: 'progress'; schema_version: number; data: ProgressData }
  | { type: 'generation'; schema_version: number; data: GenerationData }
  | { type: 'status'; schema_version: number; data: StatusData }

export function useDashboardSocket() {
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [generation, setGeneration] = useState<GenerationData | null>(null)
  const [status, setStatus] = useState<StatusData | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`)
    socketRef.current = socket

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data) as Envelope
      if (msg.type === 'progress') setProgress(msg.data)
      else if (msg.type === 'generation') setGeneration(msg.data)
      else if (msg.type === 'status') setStatus(msg.data)
    }

    return () => socket.close()
  }, [])

  return { progress, generation, status }
}
