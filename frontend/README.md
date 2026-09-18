# neuroarena dashboard frontend

React + TypeScript + Vite single-page UI for the neuroarena dashboard (Phase 7): model list with resume, new-run/resume form, live training metrics over WebSocket, live config editing, poll-interval knob and manual stop. It talks only to the backend started by `uv run neuroarena-dashboard` (default `127.0.0.1:8000`).

## Commands

Run from this directory.

```bash
npm install        # install dependencies
npm run dev        # dev server on http://localhost:5173
npm run build      # type-check (tsc -b) and produce a production build in dist/
npm run lint       # oxlint
npx tsc -b         # type-check only
```

## Backend proxy

The dev server proxies `/api` (REST) and `/ws` (WebSocket) to the backend, so start the backend first. See `vite.config.ts` for the target.
