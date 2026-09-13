from neuroarena.backends.neat.evaluation import StepBudget, evaluate_genome
from tests.interfaces.doubles import DummyEnvironment, DummyModel, DummyObjective


def test_evaluate_genome_runs_until_env_truncates() -> None:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = DummyObjective()
    budget = StepBudget(remaining=100)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    # DummyEnvironment truncates after 5 steps (see tests/interfaces/doubles.py).
    assert result.fitness == 5.0
    assert result.final_info == {"t": 5}
    assert budget.remaining == 95


def test_evaluate_genome_force_truncates_when_budget_runs_out_mid_episode() -> None:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = DummyObjective()
    budget = StepBudget(remaining=3)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    assert result.fitness == 3.0
    assert budget.remaining == 0


def test_evaluate_genome_scores_zero_when_budget_is_already_exhausted() -> None:
    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = DummyObjective()
    budget = StepBudget(remaining=0)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    assert result.fitness == 0.0
    assert result.final_info == {}


def test_evaluate_genome_stops_early_when_objective_says_stop() -> None:
    class StopsImmediately(DummyObjective):
        def should_stop(self) -> bool:
            return True

    env = DummyEnvironment()
    model = DummyModel(env.observation_space, env.action_space)
    objective = StopsImmediately()
    budget = StepBudget(remaining=100)
    result = evaluate_genome(model, env, objective, budget, seed=0)
    assert result.fitness == 1.0
    assert budget.remaining == 99
