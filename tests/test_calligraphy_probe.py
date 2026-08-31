import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_calligraphy_probe.py"
SPEC = importlib.util.spec_from_file_location("calligraphy_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


DATA_ROOT = ROOT / "data" / "raw" / "calligraphy" / "makemeahanzi"


def pilot_sample():
    records = MODULE.load_makemeahanzi_records(DATA_ROOT)
    record = next(
        item
        for item in records
        if item.character == "他" and item.n_strokes >= 4 and item.component_supports
    )
    return MODULE.build_calligraphy_sample(record, split="train")


def test_makemeahanzi_parser_and_five_point_sampling():
    records = MODULE.load_makemeahanzi_records(DATA_ROOT)
    assert len(records) > 800
    sample = pilot_sample()
    assert sample.n_strokes == len(sample.stroke_ids) // MODULE.STROKE_POINTS
    assert sample.points.shape[1] == 2
    assert sample.points.shape[0] == sample.n_strokes * MODULE.STROKE_POINTS
    assert np.allclose(sample.points[0], sample.stroke_points[0][0])
    assert np.allclose(sample.points[MODULE.STROKE_POINTS - 1], sample.stroke_points[0][-1])
    assert np.allclose(sample.arc_positions[::MODULE.STROKE_POINTS], 0.0)
    assert np.allclose(sample.arc_positions[MODULE.STROKE_POINTS - 1 :: MODULE.STROKE_POINTS], 1.0)
    # A stroke remains an ordered extended object; it is never collapsed to a
    # single node or represented by endpoints alone.
    assert np.all(sample.stroke_points.shape[1:] == (MODULE.STROKE_POINTS, 2))


def test_symmetric_connected_graph_and_direction_features():
    sample = pilot_sample()
    assert np.allclose(sample.adjacency, sample.adjacency.T, atol=1e-8)
    assert np.all(np.diag(sample.adjacency) == 0)
    assert MODULE.graph_is_connected(sample.adjacency)
    assert np.all(np.isfinite(sample.eigenvalues))
    assert np.all(np.diff(sample.eigenvalues) >= -1e-8)
    assert sample.node_features.shape == (len(sample.points), MODULE.NODE_FEATURE_DIM)
    # tangent x/y, arc position and start/end flags are explicit model inputs.
    assert np.allclose(np.linalg.norm(sample.tangents, axis=1), 1.0, atol=1e-5)
    assert set(np.unique(sample.is_endpoint)).issubset({0, 1})
    assert np.any(sample.is_endpoint[:, 0] == 1)
    assert np.any(sample.is_endpoint[:, 1] == 1)


def test_character_split_is_deterministic_and_disjoint():
    records = MODULE.load_makemeahanzi_records(DATA_ROOT)
    splits = MODULE.split_records(records, n_chars=24, seed=20260828)
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 16,
        "dev": 4,
        "heldout": 4,
    }
    chars = {key: {item.character for item in value} for key, value in splits.items()}
    assert not chars["train"] & chars["dev"]
    assert not chars["train"] & chars["heldout"]
    assert not chars["dev"] & chars["heldout"]
    again = MODULE.split_records(records, n_chars=24, seed=20260828)
    assert [[item.character for item in again[key]] for key in splits] == [
        [item.character for item in splits[key]] for key in splits
    ]


def test_intervention_energy_and_structural_invariants():
    sample = pilot_sample()
    interventions = MODULE.generate_interventions(
        [sample], seeds=[11], energies=[0.25], include_exact_modes=True
    )
    assert interventions["intervention_id"].is_unique
    assert len(interventions.attrs["delta_arrays"]) == len(interventions)
    required = {
        "identity",
        "common",
        "semantic_component",
        "matched_random_component",
        "stroke_rigid",
        "endpoint_local",
        "interior_local",
        "band_0",
        "band_5",
    }
    assert required.issubset(set(interventions["kind"]))
    validation = MODULE.validate_interventions([sample], interventions)
    assert validation["status"] == "pass", validation
    for row in interventions[interventions["valid"]].itertuples(index=False):
        delta = np.asarray(json.loads(row.delta), dtype=np.float64)
        if row.kind != "identity":
            assert abs(np.linalg.norm(delta) - row.epsilon) <= 1e-7
        if row.kind == "common":
            assert np.allclose(delta, delta[0], atol=1e-7)
        if row.kind == "stroke_rigid":
            support = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-9)
            assert len(support) == MODULE.STROKE_POINTS
            assert np.unique(sample.stroke_ids[support]).size == 1
            assert np.allclose(delta[support], delta[support[0]], atol=1e-7)
        if row.kind in {"endpoint_local", "interior_local"}:
            assert np.count_nonzero(np.linalg.norm(delta, axis=1) > 1e-9) == 1


def test_intervention_public_manifest_drops_runtime_ndarray_cache(tmp_path):
    sample = pilot_sample()
    interventions = MODULE.generate_interventions([sample], seeds=[11], energies=[0.25])
    public = MODULE._public_intervention_frame(interventions)
    assert public.attrs == {}
    path = tmp_path / "interventions.parquet"
    public.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    assert "_delta_array" not in restored.columns
    assert len(restored) == len(interventions)


def test_streamed_intervention_manifest_preserves_contract(tmp_path):
    sample = pilot_sample()
    path = tmp_path / "streamed_interventions.parquet"
    validation = MODULE.stream_intervention_manifest(
        [sample], path, seeds=[11], energies=[0.25], chunk_size=2
    )
    restored = pd.read_parquet(path)
    assert validation["status"] == "pass", validation
    assert validation["streamed"] is True
    assert validation["n_rows"] == len(restored)
    assert validation["n_valid"] == len(restored)
    assert "_delta_array" not in restored.columns
    assert restored["intervention_id"].is_unique


def test_vectorized_features_and_padded_grouping_preserve_sample_order():
    sample = pilot_sample()
    records = MODULE.load_makemeahanzi_records(DATA_ROOT)
    other_record = next(item for item in records if item.n_strokes != sample.n_strokes)
    other = MODULE.build_calligraphy_sample(other_record, split="dev")
    features, tangents, curvature, arc_positions, is_endpoint = MODULE._node_features_batch(
        sample.points[None, ...], sample.n_strokes
    )
    assert np.allclose(features[0], sample.node_features)
    assert np.allclose(tangents[0], sample.tangents)
    assert np.allclose(curvature[0], sample.curvature)
    assert np.allclose(arc_positions[0], sample.arc_positions)
    assert np.allclose(is_endpoint[0], sample.is_endpoint)

    batch = MODULE.pad_samples([sample, other])
    assert batch.features.shape[0] == 2
    assert np.allclose(batch.features[0, : len(sample.points)].numpy(), sample.node_features)
    assert np.allclose(batch.features[1, : len(other.points)].numpy(), other.node_features)
    assert bool(batch.mask[0, len(sample.points) :].any()) is False


def test_runtime_policy_is_explicit_about_cuda_and_batch_defaults():
    import torch

    device, amp_enabled, metadata = MODULE.resolve_runtime("auto", "auto", "none")
    assert metadata["resolved_device"] == str(device)
    assert amp_enabled is (device.type == "cuda")
    inference_batch, train_batch = MODULE.resolve_batch_sizes(device)
    if device.type == "cuda":
        assert (inference_batch, train_batch) == (256, 16)
    else:
        assert (inference_batch, train_batch) == (64, 16)
    with pytest.raises(ValueError):
        MODULE.resolve_batch_sizes(device, batch_size=0)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            MODULE.resolve_runtime("cuda", "auto", "none")
        with pytest.raises(RuntimeError):
            MODULE.resolve_runtime("auto", "auto", "v100")


def test_padded_point_models_and_vector_renderer():
    import torch

    sample = pilot_sample()
    batch = MODULE.pad_samples([sample])
    models = MODULE.build_point_models()
    z_set, recon_set = models["deepsets_ae"](
        batch.features, batch.mask
    )
    z_gat, recon_gat = models["gat_ae"](
        batch.features, batch.adjacency, batch.mask
    )
    assert z_set.shape == (1, 128)
    assert z_gat.shape == (1, 128)
    assert recon_set.shape[:2] == batch.features.shape[:2]
    assert recon_gat.shape[:2] == batch.features.shape[:2]
    with torch.inference_mode():
        assert torch.allclose(z_set, models["deepsets_ae"].encode(batch.features, batch.mask))
        assert torch.allclose(
            z_gat, models["gat_ae"].encode(batch.features, batch.adjacency, batch.mask)
        )
    deepsets_batch = MODULE.pad_samples([sample], include_adjacency=False)
    assert deepsets_batch.adjacency is None
    image = MODULE.render_vector(sample, sample.points)
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image, MODULE.render_vector(sample, sample.points))


def test_execution_contract_is_exploratory_and_isolated():
    contract = MODULE.build_execution_contract(n_chars=24, epochs=1, smoke=True)
    assert contract["study_mode"] == "exploratory_discovery"
    assert contract["data_domain"] == "chinese_calligraphy_vector_controlled"
    assert contract["bioinformatics_isolation"] is True
    assert contract["mainline_models"] == ["deepsets_ae", "gat_ae"]
    assert contract["auxiliary_model"] == "resnet18"
