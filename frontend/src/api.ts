import type { Model, RunStatus, SpeedState, StartRunRequest, Track, ViewSettings, ViewerState } from './types'

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(detail.detail ?? `request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function listTracks(): Promise<Track[]> {
  return fetch('/api/tracks').then((r) => json<Track[]>(r))
}

export function listModels(): Promise<Model[]> {
  return fetch('/api/models').then((r) => json<Model[]>(r))
}

export function getModelSettings(modelId: string): Promise<Record<string, unknown>> {
  return fetch(`/api/models/${modelId}/settings`).then((r) => json<Record<string, unknown>>(r))
}

export function startRun(request: StartRunRequest): Promise<{ model_id: string; status: string }> {
  return fetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  }).then((r) => json<{ model_id: string; status: string }>(r))
}

export function getCurrentRun(): Promise<RunStatus> {
  return fetch('/api/runs/current').then((r) => json<RunStatus>(r))
}

export function stopRun(): Promise<{ status: string }> {
  return fetch('/api/runs/current/stop', { method: 'POST' }).then((r) => json(r))
}

export function updateConfig(
  partial: Record<string, number | null>,
): Promise<{ status: string }> {
  return fetch('/api/runs/current/config', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  }).then((r) => json(r))
}

export function getPollInterval(): Promise<{ interval_ms: number }> {
  return fetch('/api/poll-interval').then((r) => json(r))
}

export function setPollInterval(intervalMs: number): Promise<{ interval_ms: number }> {
  return fetch('/api/poll-interval', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ interval_ms: intervalMs }),
  }).then((r) => json(r))
}

export function getSpeed(): Promise<SpeedState> {
  return fetch('/api/speed').then((r) => json<SpeedState>(r))
}

export function setSpeed(preset: string): Promise<{ preset: string }> {
  return fetch('/api/speed', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset }),
  }).then((r) => json<{ preset: string }>(r))
}

export function getViewer(): Promise<ViewerState> {
  return fetch('/api/viewer').then((r) => json<ViewerState>(r))
}

export function openViewer(): Promise<ViewerState> {
  return fetch('/api/viewer/open', { method: 'POST' }).then((r) => json<ViewerState>(r))
}

export function closeViewer(): Promise<ViewerState> {
  return fetch('/api/viewer/close', { method: 'POST' }).then((r) => json<ViewerState>(r))
}

export function updateViewerSettings(
  partial: Record<string, number | string>,
): Promise<ViewSettings> {
  return fetch('/api/viewer/settings', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  }).then((r) => json<ViewSettings>(r))
}
