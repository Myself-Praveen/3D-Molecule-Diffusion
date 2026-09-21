"""Federated server loop with weighted FedAvg / FedProx (Steps 2.2–2.3, 3.1).

Implements an in-process virtual-client simulation of the GraphGANFed
protocol: the global model is broadcast to every participating client each
round, clients run E local epochs, and updated backbone weights are averaged
weighted by dataset size (Eq. 8 of the paper). FedProx is realized through
the client-side proximal term (μ > 0); personalization keeps type/bond heads
local ("FedPer"-style).

The Flower-compatible ``MoleculeDiffusionClient`` (``src/fed/client.py``) can
be dropped into ``flwr.simulation`` unchanged; this loop exists so runs are
deterministic and dependency-stable across Flower releases.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.fed.trainer import LocalTrainer, merge_global_personal, split_global_personal


class FederatedServer:
    """Round-based coordinator for K simulated clients."""

    def __init__(
        self,
        trainers: list[LocalTrainer],
        train_loaders: list,
        val_loader,
        coord_ddpm,
        type_ddpm,
        config: dict,
        output_dir: str | Path = "outputs/fed",
    ) -> None:
        self.trainers = trainers
        self.train_loaders = train_loaders
        self.val_loader = val_loader
        self.coord_ddpm = coord_ddpm
        self.type_ddpm = type_ddpm
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mu = float(config["fed"].get("proximal_mu", 0.0))
        self.personalized = bool(config["fed"].get("personal_heads", False))
        self.history: list[dict] = []

    # ------------------------------------------------------------------ core

    @property
    def global_state(self) -> dict[str, torch.Tensor]:
        """Aggregated backbone weights live on trainer 0's model."""
        return self.trainers[0].get_parameters(exclude_personal=self.personalized)

    def _broadcast(self) -> dict[str, torch.Tensor]:
        """Snapshot global weights before this round (for proximal term)."""
        snapshot = {k: v.clone() for k, v in self.global_state.items()}
        backbone = split_global_personal(snapshot)[0]
        for trainer in self.trainers:
            if self.personalized:
                # Personal heads stay local; replace only the trunk.
                merged = merge_global_personal(
                    {k: v.clone() for k, v in backbone.items()},
                    split_global_personal(trainer.model.state_dict())[1],
                )
                trainer.model.load_state_dict(merged)
            else:
                trainer.set_parameters(backbone)
        return snapshot

    def _valence_weight(self, round_idx: int) -> float:
        """Effective λ₂ for this round with linear warmup.

        128-dim EGNNs explode on pathological early batches (1e28 losses seen
        round 1), so geometry pressure fades in only after weights stabilize:
        0 for the first 5 rounds, linear ramp over ``valence_warmup_rounds``.
        A warmup of 0 (default) reproduces the old constant-weight behavior.
        Reads the target stashed by ``run_round`` (the live key holds the
        previously scheduled value).
        """
        training = self.cfg["training"]
        target = float(training.get(
            "_valence_target", training.get("valence_loss_weight", 0.0)))
        warmup = int(self.cfg["training"].get("valence_warmup_rounds", 0))
        if warmup <= 0 or round_idx <= 5:
            frac = 0.0 if warmup > 0 and round_idx <= 5 else 1.0
        else:
            frac = min(1.0, (round_idx - 5) / warmup)
        return target * frac

    def run_round(self, round_idx: int, lr: float | None = None) -> dict:
        local_epochs = int(self.cfg["fed"].get("local_epochs", 1))
        snapshot = self._broadcast()

        # Publish the warmup-scaled λ₂ where LocalTrainers already read it
        # (self.cfg is shared by reference); target preserved on first call.
        training = self.cfg["training"]
        if "_valence_target" not in training:
            training["_valence_target"] = float(training.get("valence_loss_weight", 0.0))
        training["valence_loss_weight"] = self._valence_weight(round_idx)

        results = []
        for trainer, loader in zip(self.trainers, self.train_loaders):
            metrics = trainer.train(
                loader,
                global_state=snapshot if self.mu > 0 else None,
                local_epochs=local_epochs,
                mu=self.mu,
                lr=lr,
            )
            metrics["num_examples"] = sum(
                b.num_graphs for b in loader
            ) if hasattr(next(iter(loader)), "num_graphs") else len(loader.dataset)
            results.append(metrics)

        # Weighted FedAvg over backbone parameters (Eq. 8).
        client_states = [
            (trainer.get_parameters(exclude_personal=self.personalized),
             res["num_examples"])
            for trainer, res in zip(self.trainers, results)
        ]
        aggregated = weighted_avg(client_states)

        # Install aggregated weights into every trainer.
        for trainer in self.trainers:
            if self.personalized:
                merged = merge_global_personal(
                    aggregated,
                    split_global_personal(trainer.model.state_dict())[1],
                )
                trainer.model.load_state_dict(merged)
            else:
                trainer.set_parameters(aggregated)

        total_n = sum(r["num_examples"] for r in results)
        round_log = {
            "round": round_idx,
            "server_loss": sum(
                r["loss"] * r["num_examples"] for r in results
            ) / total_n,
            "valence_weight": float(self.cfg["training"].get("valence_loss_weight", 0.0)),
            "client_losses": [r["loss"] for r in results],
            "client_pos_losses": [r["pos_loss"] for r in results],
            "client_type_losses": [r["type_loss"] for r in results],
            "num_examples": [r["num_examples"] for r in results],
        }
        return round_log

    def evaluate_global(self) -> dict[str, float]:
        """Evaluate the current global weights on the shared validation set."""
        trainer = self.trainers[0]
        trainer.model.eval()
        return trainer.evaluate(self.val_loader)

    # ------------------------------------------------------------------ main

    def _atomic_torch_save(self, state: dict, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def fit(self, start_round: int = 1, run_rounds: int | None = None) -> list[dict]:
        rounds = int(self.cfg["fed"]["rounds"])
        eval_every = int(self.cfg["fed"].get("eval_every", 5))
        sample_every = int(self.cfg["fed"].get("sample_eval_every", 10))
        lr_schedule = self.cfg["fed"].get("lr_per_round")

        if run_rounds is not None and run_rounds > 0:
            end_round = min(rounds, start_round + run_rounds - 1)
        else:
            end_round = rounds
        if start_round > rounds:
            print(f"Federated training already complete "
                  f"(start_round={start_round} > total={rounds})")
            return self.history
        print(f"Running rounds {start_round}..{end_round} (total {rounds})")

        # Restore best_val from loaded history (resume) so best_global.pt logic continues
        best_val = float("inf")
        for h in self.history:
            v = (h.get("val") or {}).get("loss")
            if v is not None and v < best_val:
                best_val = v
        for round_idx in range(start_round, end_round + 1):
            t0 = time.time()
            lr = lr_schedule[round_idx - 1] if lr_schedule else None
            log = self.run_round(round_idx, lr=lr)
            log["seconds"] = round(time.time() - t0, 2)

            if round_idx % eval_every == 0 or round_idx == rounds:
                val_metrics = self.evaluate_global()
                log["val"] = val_metrics
                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    self._atomic_torch_save(
                        {
                            "round": round_idx,
                            "global_state": self.global_state,
                            "best_val": best_val,
                            "config": self.cfg,
                        },
                        self.output_dir / "best_global.pt",
                    )

            if (
                sample_every > 0
                and round_idx % sample_every == 0
            ):
                log["generation"] = self._quick_generation_metrics()

            print(
                f"Round {round_idx:3d} | server_loss={log['server_loss']:.4f} "
                f"| client_losses={['%.4f' % l for l in log['client_losses']]} "
                f"| {log['seconds']}s"
                + (f" | val={log['val']['loss']:.4f}" if "val" in log else "")
                + (f" | gen={log['generation']}" if "generation" in log else "")
            )
            self.history.append(log)
            self._save_history()
            # last_global.pt every round = resume point (max 1 round lost on kill)
            personal_states = None
            if self.personalized:
                personal_states = [
                    {k: v.cpu() for k, v in split_global_personal(
                        t.model.state_dict())[1].items()}
                    for t in self.trainers
                ]
            self._atomic_torch_save(
                {
                    "round": round_idx,
                    "global_state": {k: v.cpu() for k, v in self.global_state.items()},
                    "personal_states": personal_states,
                    "best_val": best_val,
                    "config": self.cfg,
                },
                self.output_dir / "last_global.pt",
            )

        self._atomic_torch_save(
            {
                "round": end_round,
                "global_state": self.global_state,
                "config": self.cfg,
            },
            self.output_dir / "final_global.pt",
        )
        if end_round < rounds:
            print(f"\nChunk done at round {end_round}/{rounds}. "
                  f"Resume with: python fed_train.py --config <cfg> --resume --run_rounds N")
        return self.history

    # ------------------------------------------------------------- sampling

    @torch.no_grad()
    def _quick_generation_metrics(self, num_samples: int | None = None) -> dict:
        """DDIM-sample a few molecules and score validity/uniqueness."""
        from src.sampling import sample_molecules
        from src.utils.evaluation import coords_and_types_to_mol, validity, uniqueness

        gen_cfg = self.cfg.get("generation", {})
        num_samples = num_samples or int(gen_cfg.get("num_samples", 32))
        max_atoms = int(gen_cfg.get("max_atoms", 12))
        device = next(self.trainers[0].model.parameters()).device

        counts = torch.randint(4, max_atoms + 1, (num_samples,))
        pos, z = sample_molecules(
            self.trainers[0].model,
            self.coord_ddpm,
            self.type_ddpm,
            atom_counts=counts,
            device=device,
            ddim_steps=int(gen_cfg.get("ddim_steps", 50)),
        )

        mols = []
        offset = 0
        for c in counts.tolist():
            mols.append(coords_and_types_to_mol(
                pos[offset:offset + c].numpy(), z[offset:offset + c].numpy(),
            ))
            offset += c
        return {
            "validity": round(validity(mols) * 100.0, 2),
            "uniqueness": round(uniqueness(mols) * 100.0, 2),
        }

    def _save_history(self) -> None:
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def weighted_avg(
    client_states: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Weighted parameter aggregation (Eq. 8): w = Σ (n_i/N) · w_i."""
    from src.fed.trainer import weighted_fedavg

    return weighted_fedavg(client_states)
