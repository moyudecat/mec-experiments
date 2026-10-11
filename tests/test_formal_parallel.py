from mec_dualbp.experiments.exp1_formal_protocol import generate_formal_samples
from mec_dualbp.experiments.exp1_formal_runtime import run_parallel_units, unit_hash_bundle
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash


def _specs(tmp_path, workers):
    sample = generate_formal_samples()[0]
    names = (
        "edge_core_latency_lower", "edge_core_latency_upper",
        "edge_comm_energy_lower", "edge_comm_energy_upper",
    )
    rows = []
    for offset, name in enumerate(names):
        method = {"name": "Dense-LHS_parallel_test", "n_samples": 64}
        rows.append({
            "key": {"sample_id": sample.sample_id, "task_name": name, "method_name": method["name"], "seed": 9900 + offset},
            "checkpoint_root": str(tmp_path / f"w{workers}" / "checkpoint"),
            "stale_after_seconds": 60,
            "hashes": unit_hash_bundle("formal", "code", sample.content_hash, canonical_json_hash({"task": name}), method),
            "parameters": sample.parameters, "gamma": sample.gamma,
            "dense_count": 64, "chunk_size": 17,
        })
    return rows


def test_process_pool_spawn_workers_1_2_4_are_consistent(tmp_path) -> None:
    runs = {workers: run_parallel_units(_specs(tmp_path, workers), workers) for workers in (1, 2, 4)}
    baseline = {row["unit_id"].split("::")[1]: row["result"]["response_value"] for row in runs[1]["results"]}
    for workers, run in runs.items():
        assert run["completed_units"] == 4 and run["worker_failures"] == 0
        observed = {row["unit_id"].split("::")[1]: row["result"]["response_value"] for row in run["results"]}
        assert observed == baseline
