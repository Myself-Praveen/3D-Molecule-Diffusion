"""R3 (rescue plan): the training probe measures real generation quality.

The old probe sampled 16 molecules of a hardcoded 18 atoms — QM9's size, while
BBBP's mean is 46 — and returned validity only, so ``best.pt`` was ranked by val
loss. These tests pin that sizes now come from the training histogram and that
the probe reports the connectivity axes used for selection.
"""

from __future__ import annotations

import torch

import train


def test_probe_draws_sizes_from_the_training_histogram(monkeypatch):
    seen: dict = {}

    def fake_sample(model, coord_ddpm, type_ddpm, atom_counts, **kwargs):
        counts = [int(c) for c in atom_counts.tolist()]
        seen["counts"] = counts
        chunks = []
        for c in counts:
            chunks.append(torch.stack(
                [torch.arange(c, dtype=torch.float32) * 1.54,
                 torch.zeros(c), torch.zeros(c)],
                dim=1,
            ))
        return torch.cat(chunks), torch.full((sum(counts),), 6, dtype=torch.long)

    monkeypatch.setattr("src.sampling.sample_molecules", fake_sample)
    out = train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=3, ddim_steps=2, atom_count_pool=[40],
    )

    assert seen["counts"] == [40, 40, 40], "sizes must come from the real histogram"
    assert set(out) == {"validity", "conn_frac", "nn_outside", "score"}


def test_probe_falls_back_to_qm9_size_without_a_pool(monkeypatch):
    seen: dict = {}

    def fake_sample(model, coord_ddpm, type_ddpm, atom_counts, **kwargs):
        seen["counts"] = [int(c) for c in atom_counts.tolist()]
        n = int(sum(seen["counts"]))
        return torch.zeros(n, 3), torch.full((n,), 6, dtype=torch.long)

    monkeypatch.setattr("src.sampling.sample_molecules", fake_sample)
    train._quick_validity(
        torch.nn.Identity(), object(), object(), "cpu",
        num_samples=2, ddim_steps=2,
    )
    assert seen["counts"] == [18, 18]
