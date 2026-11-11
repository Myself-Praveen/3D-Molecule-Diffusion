"""R3 + fix.md F11: the training probe measures real generation quality.

The old probe sampled 16 molecules of a hardcoded 18 atoms — QM9's size, while
BBBP's mean is 46 — and returned validity only, so ``best.pt`` was ranked by val
loss. On top of that its score (``conn_frac - nn_outside``) was computed over
*valid* molecules only, so a model that connected the handful it could parse
scored the same as one that connected everything: V7-nosnr's epoch 3400 was
saved on a 6.2 %-validity probe that measured 33.0 % validity at 200 samples.

These tests pin the corrected contract: sizes come from the training
histogram, validity GATES connectivity, and an underpowered probe scores -inf.
"""

from __future__ import annotations

import torch

import train


def _chain(counts, offset_atoms=0):
    """``counts`` carbon chains at 1.54 Å — valid and connected."""
    pos, z = [], []
    for c in counts:
        p = torch.zeros(c, 3)
        p[:, 0] = torch.arange(c, dtype=torch.float32) * 1.54
        pos.append(p)
        z.append(torch.full((c,), 6, dtype=torch.long))
    return torch.cat(pos), torch.cat(z)


def _over_valent(counts):
    """One carbon ringed by hydrogens at 1.0 Å — sanitization fails."""
    pos, z = [], []
    for c in counts:
        p = torch.zeros(c, 3)
        if c > 1:
            ang = torch.linspace(0.0, 6.0, c - 1)
            p[1:, 0] = torch.cos(ang)
            p[1:, 1] = torch.sin(ang)
        pos.append(p)
        z.append(torch.cat([torch.tensor([6]),
                            torch.ones(c - 1, dtype=torch.long)]))
    return torch.cat(pos), torch.cat(z)


def _patch(monkeypatch, builder):
    seen: dict = {}

    def fake_sample(model, coord_ddpm, type_ddpm, atom_counts, **kwargs):
        counts = [int(c) for c in atom_counts.tolist()]
        seen["counts"] = counts
        return builder(counts)

    monkeypatch.setattr("src.sampling.sample_molecules", fake_sample)
    return seen


def test_probe_draws_sizes_from_the_training_histogram(monkeypatch):
    seen = _patch(monkeypatch, lambda counts: _chain(counts))
    out = train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=3, ddim_steps=2, atom_count_pool=[40],
    )

    assert seen["counts"] == [40, 40, 40], "sizes must come from the real histogram"
    assert set(out) == {"validity", "conn_frac", "nn_outside", "score",
                        "n_valid", "n_samples"}
    assert out["n_valid"] == 3 and out["n_samples"] == 3


def test_probe_falls_back_to_qm9_size_without_a_pool(monkeypatch):
    seen = _patch(monkeypatch, lambda counts: _chain(counts))
    train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=2, ddim_steps=2,
    )
    assert seen["counts"] == [18, 18]


def test_probe_prefers_the_valid_model(monkeypatch):
    """F11: 16-of-16 valid must beat 4-of-16 valid, even at equal conn_frac."""
    full = {}
    _patch(monkeypatch, lambda counts: _chain(counts))
    full = train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=16, ddim_steps=2, atom_count_pool=[8],
    )

    def partial(counts):
        # First 4 molecules are proper chains, the rest cannot sanitize.
        good_pos, good_z = _chain(counts[:4])
        bad_pos, bad_z = _over_valent(counts[4:])
        return torch.cat([good_pos, bad_pos]), torch.cat([good_z, bad_z])

    _patch(monkeypatch, partial)
    sparse = train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=16, ddim_steps=2, atom_count_pool=[8],
    )

    assert sparse["n_valid"] == 4, sparse
    assert sparse["conn_frac"] == 100.0, "conditional conn_frac is a perfect 100"
    assert full["validity"] == 100.0
    assert full["score"] > sparse["score"], "validity must gate the score"


def test_probe_refuses_to_score_an_underpowered_sample(monkeypatch):
    _patch(monkeypatch, lambda counts: _over_valent(counts))
    out = train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=16, ddim_steps=2, atom_count_pool=[8],
    )
    assert out["n_valid"] == 0
    assert out["score"] == float("-inf")
