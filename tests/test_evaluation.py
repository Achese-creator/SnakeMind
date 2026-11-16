import csv

import pytest

from snakemind.agents import RandomAgent, HamiltonianAgent
from snakemind.env import SnakeEnv
from snakemind.evaluation import (
    evaluate_agent,
    summarize_results,
    save_results_csv,
)


def make_env():
    return SnakeEnv(size=10, max_episode_steps=30)


def test_evaluation_runs_requested_number_of_episodes():
    results = evaluate_agent(
        RandomAgent(),
        make_env,
        seeds=[10, 20, 30],
    )

    assert len(results) == 3


def test_evaluation_records_required_metrics():
    results = evaluate_agent(
        RandomAgent(seed=42),
        make_env,
        seeds=[42],
    )

    result = results[0]

    expected_fields = {
        "seed",
        "score",
        "survival_steps",
        "episode_return",
        "final_length",
        "terminated",
        "truncated",
        "termination_reason",
    }

    assert expected_fields.issubset(result.keys())


def test_evaluation_is_reproducible():
    first = evaluate_agent(
        RandomAgent(),
        make_env,
        seeds=[42],
    )

    second = evaluate_agent(
        RandomAgent(),
        make_env,
        seeds=[42],
    )

    assert first == second


def test_empty_seed_list_is_rejected():
    with pytest.raises(ValueError):
        evaluate_agent(RandomAgent(), make_env, seeds=[])


def test_summary_calculates_statistics():
    results = [
        {
            "score": 2,
            "survival_steps": 20,
            "episode_return": 15.0,
            "final_length": 5,
            "termination_reason": "collision",
        },
        {
            "score": 4,
            "survival_steps": 40,
            "episode_return": 25.0,
            "final_length": 7,
            "termination_reason": "collision",
        },
    ]

    summary = summarize_results(results)

    assert summary["episodes"] == 2
    assert summary["mean_score"] == 3
    assert summary["mean_survival_steps"] == 30
    assert summary["mean_episode_return"] == 20
    assert summary["mean_final_length"] == 6
    assert summary["board_completion_rate"] == 0


def test_summary_rejects_empty_results():
    with pytest.raises(ValueError):
        summarize_results([])


def test_csv_export_creates_readable_file(tmp_path):
    results = evaluate_agent(
        RandomAgent(seed=42),
        make_env,
        seeds=[42],
    )

    output = tmp_path / "evaluation.csv"

    save_results_csv(results, output)

    assert output.exists()

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert int(rows[0]["seed"]) == 42


def test_hamiltonian_agent_completes_board():
    results = evaluate_agent(
        HamiltonianAgent(),
        lambda: SnakeEnv(
            size=10,
            max_episode_steps=10000,
        ),
        seeds=[42],
    )

    assert results[0]["termination_reason"] == "board_filled"
    assert results[0]["final_length"] == 100
