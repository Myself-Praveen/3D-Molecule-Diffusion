"""Flower client for molecule-diffusion federated training (Step 2.2).

``MoleculeDiffusionClient`` wraps :class:`src.fed.trainer.LocalTrainer` in a
``flwr.client.NumPyClient`` interface:

- ``fit``: receive global weights → run E local epochs (with optional FedProx
  proximal term μ) → return updated *global-backbone* weights + num_examples.
- ``evaluate``: return local validation loss.

With personalization enabled (Step 3.2B) the type/bond heads are never sent
to or received from the server.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from flwr.client import NumPyClient

from src.fed.trainer import LocalTrainer


def state_dict_to_ndarrays(state: dict[str, torch.Tensor]) -> dict[str, "np.ndarray"]:
    return {k: v.cpu().numpy() for k, v in state.items()}


def ndarrays_to_state_dict(ndarrays: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in ndarrays.items()}


class MoleculeDiffusionClient(NumPyClient):
    """Flower NumPyClient delegating to a shared LocalTrainer."""

    def __init__(
        self,
        client_id: int,
        trainer: LocalTrainer,
        train_loader,
        val_loader,
        mu: float = 0.0,
        personalized: bool = False,
    ) -> None:
        self.client_id = client_id
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mu = mu
        self.personalized = personalized

    # -- Flower plumbing ----------------------------------------------------

    def get_properties(self, ins: dict[str, Any]) -> dict[str, Any]:
        return {"client_id": self.client_id}

    def get_parameters(self, config: dict[str, Any]):
        state = self.trainer.get_parameters(exclude_personal=self.personalized)
        return list(state_dict_to_ndarrays(state).values())

    def set_parameters(self, parameters) -> None:
        names = [
            k for k in self.trainer.model.state_dict()
            if not (self.personalized and k.startswith(("type_head", "bond_head")))
        ]
        ndarrays = OrderedDict(zip(names, parameters))
        self.trainer.set_parameters(ndarrays_to_state_dict(ndarrays))

    # -- Flower protocol ------------------------------------------------------

    def fit(self, parameters, config: dict[str, Any]):
        if len(parameters):
            self.set_parameters(parameters)
        global_state = (
            {k: v.clone() for k, v in self.trainer.model.state_dict().items()}
            if self.mu > 0 else None
        )
        metrics = self.trainer.train(
            self.train_loader,
            global_state=global_state,
            local_epochs=int(config.get("local_epochs", 1)),
            mu=self.mu,
            lr=float(config.get("lr", 1e-3)),
        )
        new_params = self.get_parameters(config)
        num_examples = sum(
            b.num_graphs for b in self.train_loader
        ) if hasattr(next(iter(self.train_loader)), "num_graphs") else 0
        # Fallback count via dataset length.
        if num_examples == 0:
            num_examples = len(self.train_loader.dataset)
        return new_params, num_examples, {
            "loss": metrics["loss"],
            "pos_loss": metrics["pos_loss"],
            "type_loss": metrics["type_loss"],
        }

    def evaluate(self, parameters, config: dict[str, Any]):
        if len(parameters):
            self.set_parameters(parameters)
        metrics = self.trainer.evaluate(self.val_loader)
        num_examples = len(self.val_loader.dataset)
        return float(metrics["loss"]), num_examples, {
            "pos_loss": metrics["pos_loss"],
            "type_loss": metrics["type_loss"],
        }
