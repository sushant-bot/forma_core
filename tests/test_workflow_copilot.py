from workflow.copilot import _analyze_convergence


def test_convergence_analysis_handles_short_generation_logs():
    gen_log = [
        {"best": 10.0, "total_gens": 3},
        {"best": 9.0, "total_gens": 3},
        {"best": 8.0, "total_gens": 3},
    ]

    insights = _analyze_convergence(gen_log)

    assert isinstance(insights, list)
    assert any(
        insight.title == "GA may not have fully converged"
        for insight in insights
    )