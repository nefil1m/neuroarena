import { useCallback, useEffect, useState } from 'react'
import { closeViewer, getViewer, openViewer, updateViewerSettings } from '../api'
import type { CameraMode, ViewerState } from '../types'

const POLL_MS = 1000 // notice a window closed by hand

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export function ViewerPanel() {
  const [viewer, setViewer] = useState<ViewerState | null>(null)
  const [zoomDraft, setZoomDraft] = useState<number | null>(null)
  const [rank, setRank] = useState('1')
  const [message, setMessage] = useState<string | null>(null)

  const refresh = useCallback(() => {
    getViewer()
      .then(setViewer)
      .catch((err) => setMessage(describe(err)))
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, POLL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  async function handleOpen() {
    setMessage(null)
    try {
      setViewer(await openViewer())
    } catch (err) {
      setMessage(describe(err))
    }
  }

  async function handleClose() {
    setMessage(null)
    try {
      setViewer(await closeViewer())
    } catch (err) {
      setMessage(describe(err))
    }
  }

  async function applySettings(partial: Record<string, number | string>) {
    setMessage(null)
    try {
      const settings = await updateViewerSettings(partial)
      setViewer((current) => (current === null ? current : { ...current, settings }))
    } catch (err) {
      setMessage(describe(err))
    }
  }

  // While a zoom change is pending, show the draft; otherwise the server-confirmed value.
  const zoomShown = zoomDraft ?? viewer?.settings.zoom ?? 1

  async function commitZoom(value: number) {
    if (zoomDraft === null) return // nothing was changed since the last commit
    await applySettings({ zoom: value })
    setZoomDraft(null)
  }

  const rankValue = Number(rank)
  const rankValid = Number.isInteger(rankValue) && rankValue >= 1

  return (
    <div>
      <h2>Game window</h2>
      <p>
        Opens the game on the machine running the backend, showing every car of the current
        generation. Status: {viewer === null ? '…' : viewer.open ? 'open' : 'closed'}
      </p>
      <button type="button" onClick={handleOpen} disabled={viewer === null || viewer.open}>
        Open game window
      </button>
      <button type="button" onClick={handleClose} disabled={viewer === null || !viewer.open}>
        Close game window
      </button>
      {viewer !== null && (
        <div>
          <label>
            Camera
            <select
              value={viewer.settings.camera_mode}
              onChange={(e) => applySettings({ camera_mode: e.target.value as CameraMode })}
            >
              <option value="fit">Whole track</option>
              <option value="follow_best">Follow the best car</option>
              <option value="follow_rank">Follow a chosen car</option>
            </select>
          </label>
          <label>
            Zoom ({zoomShown}×)
            <input
              type="range"
              min={0.25}
              max={4}
              step={0.25}
              value={zoomShown}
              onChange={(e) => setZoomDraft(Number(e.target.value))}
              onPointerUp={(e) => commitZoom(Number(e.currentTarget.value))}
              onKeyUp={(e) => commitZoom(Number(e.currentTarget.value))}
            />
          </label>
          <label>
            Follow rank
            <input
              type="number"
              min={1}
              step={1}
              value={rank}
              onChange={(e) => setRank(e.target.value)}
            />
          </label>
          <button
            type="button"
            disabled={!rankValid}
            onClick={() => applySettings({ camera_mode: 'follow_rank', follow_rank: rankValue })}
          >
            Follow this rank
          </button>
        </div>
      )}
      {message && <p role="alert">{message}</p>}
    </div>
  )
}
