from behaviour_prediction.monitor.forecast import make_pairs


def test_forecast_pairs_do_not_cross_runs_or_include_baseline_or_missing_horizon():
    rows = [dict(run=run, step=step) for run, steps in
            [("a", [0, 10, 20, 40, 50]), ("b", [0, 10, 20])]
            for step in steps]
    pairs = make_pairs(rows)
    assert [(rows[i]["run"], rows[i]["step"], rows[j]["step"]) for i, j in pairs] == [
        ("a", 10, 20), ("a", 40, 50), ("b", 10, 20)]
    assert all(rows[i]["run"] == rows[j]["run"] for i, j in pairs)


def test_forecast_reports_single_class_training_fallback(tmp_path, monkeypatch):
    import torch
    from behaviour_prediction.monitor import forecast
    rows = [dict(run=run, step=step, label=int(run != "a"))
            for run in ("a", "b", "c") for step in (10, 20)]
    monkeypatch.setattr(forecast, "paired_data", lambda *a:
        ({"schema": {"features": ["x"]}}, rows,
         {"summary": torch.arange(6, dtype=torch.float64).reshape(-1, 1)}, []))
    report = forecast.run_forecast([], "unused", tmp_path/"forecast")
    assert report["single_class_training_folds"] == ["a"]
    assert report["methods"]["activation_forecast"]["overall"]["checkpoints"] == 3
