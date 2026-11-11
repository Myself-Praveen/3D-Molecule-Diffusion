import pytest
import torch
from src.guidance import valence_energy, valence_guidance


def test_valid_methane_has_near_zero_energy():
    # C at origin, 4 H at tetrahedral positions (~1.09 Å from carbon).
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ])
    types = torch.tensor([6, 1, 1, 1, 1])
    batch = torch.zeros(5, dtype=torch.long)
    energy = valence_energy(pos, batch, atom_types=types)
    assert energy.item() < 1e-4


def test_hypervalent_carbon_triggers_positive_energy():
    # C with 6 close carbon neighbors at 1.40 Å in an octahedral clump.
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.40, 0.0, 0.0], [-1.40, 0.0, 0.0],
        [0.0, 1.40, 0.0], [0.0, -1.40, 0.0],
        [0.0, 0.0, 1.40], [0.0, 0.0, -1.40],
    ])
    types = torch.full((7,), 6, dtype=torch.long)
    batch = torch.zeros(7, dtype=torch.long)
    energy = valence_energy(pos, batch, atom_types=types)
    assert energy.item() > 0.5


def test_guidance_preserves_center_of_mass():
    pos = torch.randn(10, 3)
    types = torch.full((10,), 6, dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    pos = pos - pos.mean(dim=0, keepdim=True)
    guided = valence_guidance(pos, batch, atom_types=types, weight=1.0)
    com = guided.mean(dim=0)
    assert torch.allclose(com, torch.zeros(3), atol=1e-5)


def test_guidance_clamps_maximum_displacement():
    pos = torch.zeros(6, 3)
    types = torch.full((6,), 6, dtype=torch.long)
    batch = torch.zeros(6, dtype=torch.long)
    max_step = 0.15
    guided = valence_guidance(pos, batch, atom_types=types, weight=10.0, max_step=max_step)
    displacement = (guided - pos).abs().max()
    assert displacement <= max_step + 1e-6


def test_none_atom_types_uses_carbon_fallback():
    # None atom types fall back to carbon radii, so two carbons at 1.40 Å
    # should still register a soft bond. With only two atoms in the graph the
    # soft valence of each is <= 1, so the energy itself is near zero; assert
    # the radii lookup instead.
    from src.guidance import atom_radii
    radii = atom_radii(None, 3, "cpu")
    assert torch.allclose(radii, torch.full((3,), 0.76))


def test_padding_atom_type_zero_has_no_valence():
    # Class 0 is the padding/dummy node and must have v_max = 0.
    # With two padding atoms the only perceived valence is the dummy-on-dummy
    # soft bond between them, which must not penalize the system.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.40, 0.0, 0.0]])
    types = torch.tensor([0, 0])
    batch = torch.zeros(2, dtype=torch.long)
    energy = valence_energy(pos, batch, atom_types=types)
    assert energy.item() < 1e-6
