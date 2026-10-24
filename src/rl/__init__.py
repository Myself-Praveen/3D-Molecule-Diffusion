"""RL fine-tuning for BBB geometry/property steering (plan Phase 2).

``src/rl`` is deliberately **gated**: nothing here runs unless a config's
``rl.enabled`` block says so, because the plan's own rule is that RL must not
be applied before the geometry gate passes — DPO on atom soup trains the model
to reward atom soup.
"""
