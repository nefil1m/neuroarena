import { useDashboardSocket } from '../ws'

export function LiveMetrics() {
  const { progress, generation, status } = useDashboardSocket()
  const noRun = (!status || status.state === 'idle') && !progress && !generation

  return (
    <div>
      <h2>Live metrics</h2>
      {status && !noRun && <p>Status: {status.state}</p>}
      {progress && (
        <div>
          <h3>Generation {progress.generation} (in progress)</h3>
          <p>
            Active genomes: {progress.active_genomes_remaining} / {progress.population_size}
          </p>
          <p>
            Steps: {progress.elapsed_steps} / {progress.step_ceiling}
          </p>
          <p>Best fitness so far: {progress.best_fitness_so_far ?? '—'}</p>
        </div>
      )}
      {generation && (
        <div>
          <h3>Last completed: generation {generation.progress_index}</h3>
          <p>Best: {generation.best_fitness}</p>
          <p>Mean: {generation.mean_fitness}</p>
          <p>Worst: {generation.worst_fitness}</p>
        </div>
      )}
      {noRun && <p>No run active.</p>}
    </div>
  )
}
