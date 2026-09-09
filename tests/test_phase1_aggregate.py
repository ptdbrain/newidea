from scripts.aggregate_phase1 import aggregate_records


def _record(prompt_hash, attack, layer, semantic, rouge):
    return {
        "input hash": prompt_hash,
        "attack type": attack,
        "layer": layer,
        "BERTScore": semantic,
        "ROUGE_L[f1_score]": rouge,
        "time": 1.0,
    }


def test_aggregate_renames_legacy_metric_and_bootstraps_paired_delta():
    dataset = [
        {"prompt": "alpha", "sample_id": "sample-a", "source_row_id": "a", "prompt_token_length": 2, "length_stratum": "short"},
        {"prompt": "beta", "sample_id": "sample-b", "source_row_id": "b", "prompt_token_length": 3, "length_stratum": "short"},
    ]
    import hashlib

    hashes = [hashlib.sha1(item["prompt"].encode()).hexdigest() for item in dataset]
    raw = {
        "FP16": [_record(hashes[0], "inversion", 0, 1.0, 0.9), _record(hashes[1], "inversion", 0, 0.8, 0.7)],
        "KIVI4-FQ": [_record(hashes[0], "inversion", 0, 0.5, 0.4), _record(hashes[1], "inversion", 0, 0.4, 0.3)],
    }

    normalized, summary = aggregate_records(raw, dataset, bootstrap_reps=200, seed=42)

    assert normalized[0]["semantic_cosine_mpnet"] in {1.0, 0.8, 0.5, 0.4}
    fq = next(
        item
        for item in summary
        if item["condition"] == "KIVI4-FQ"
        and item["metric"] == "semantic_cosine_mpnet"
    )
    assert fq["delta_mean"] < 0
    assert fq["paired_n"] == 2
