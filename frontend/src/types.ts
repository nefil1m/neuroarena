export interface Track {
  track_id: string
  [key: string]: unknown
}

export interface Model {
  model_id: string
  backend: string
  created_at: string
}

export interface RunStatus {
  status: 'idle' | 'running' | 'completed' | 'crashed' | 'stopped'
  model_id: string | null
  generation: number | null
}

export interface StartRunRequest {
  track_id: string
  resume_model_id?: string | null
  population_size?: number | null
  max_generations?: number | null
  target_fitness?: number | null
}

export interface SpeedState {
  preset: string
  presets: string[]
}

export type CameraMode = 'fit' | 'follow_best' | 'follow_rank'

export interface ViewSettings {
  zoom: number
  camera_mode: CameraMode
  follow_rank: number
  follow_seq: number
}

export interface ViewerState {
  open: boolean
  settings: ViewSettings
}
