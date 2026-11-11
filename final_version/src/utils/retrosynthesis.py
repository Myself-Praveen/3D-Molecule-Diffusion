"""Template-based retrosynthesis feasibility (lab-readiness stage 6).

The question a synthesis lab asks first is not "is this a valid molecule"
but "can I get it from things I can buy". This module answers that at the
connectivity level: given a curated building-block list and a set of net
reaction transformations, can the target be assembled within N steps.

**What this is:** a bounded template search over *net* transformations
(amide coupling, esterification, alkylation, Suzuki, reductive amination,
SNAr, sulfonamide formation). Templates describe the connectivity change,
not reagents, selectivity, yield or protecting-group strategy — an ester
template matches the same net bond whether the actual reagent is acetic
acid or acetic anhydride.

**What this is not:** a production retrosynthesis engine (ASKCOS-class
tools do reaction-condition prediction, selectivity, and commercial
availability at real depth). What this provides is the cheap first-pass
gate that is still meaningful: if a target cannot be assembled even at the
net-transformation level from common fragments, ordering it is not on the
table. The search is deliberately *conservative* — under the intermediate
cap or beyond one fragment-growth coupling it may report "infeasible" for a
molecule a full engine could route; it must never report "feasible" for
something its own templates cannot build.

Every ``feasible`` answer carries a reconstructable route (leaves,
template per step) rather than a bare boolean, so a chemist can audit the
claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

# ---------------------------------------------------------------------------
# Net reaction transformations
# ---------------------------------------------------------------------------
# (name, reaction SMARTS). Reactant order is tried BOTH ways by the search,
# since e.g. "which partner carries the acid" is a presentation detail at
# the net-connectivity level.

REACTION_TEMPLATES: list[tuple[str, str]] = [
    ("amide_formation",
     "[C:1](=[O:2])[OX2H1].[NX3;H1,H2:3]>>[C:1](=[O:2])[N:3]"),
    ("ester_formation",
     "[C:1](=[O:2])[OX2H1].[OX2H1:3]>>[C:1](=[O:2])[O:3]"),
    ("williamson_ether",
     "[CX4:1][Br,Cl,I].[OX2H1:2]>>[O:2][C:1]"),
    ("n_alkylation",
     "[NX3;H1,H2:1].[CX4:2][Br,Cl,I]>>[N:1][C:2]"),
    ("suzuki_coupling",
     "[c:1][Br,Cl,I].[c:2][B]([O])[O]>>[c:1][c:2]"),
    ("reductive_amination",
     "[CX3:1](=[O:2]).[NX3;H1,H2:3]>>[C:1][N:3]"),
    ("snar",
     "[c:1][F,Cl,Br].[NX3;H1,H2:2]>>[c:1][N:2]"),
    ("sulfonamide_formation",
     "[NX3;H1,H2:1].[SX4:2](=[OX1])(=[OX1])[Cl]>>[N:1][S:2]"),
]

# ---------------------------------------------------------------------------
# Curated building-block set: small, common, purchasable fragments
# ---------------------------------------------------------------------------
DEFAULT_BUILDING_BLOCKS: tuple[str, ...] = (
    # acids
    "CC(=O)O",                    # acetic acid
    "c1ccccc1C(=O)O",             # benzoic acid
    "OC(=O)c1ccccc1O",            # salicylic acid
    "CC(C)C(=O)O",                # isobutyric acid
    "OC(=O)CC(=O)O",              # succinic acid
    # alcohols
    "CCO",                        # ethanol
    "CO",                         # methanol
    "c1ccccc1O",                  # phenol
    "OCCO",                       # ethylene glycol
    # amines
    "CN",                         # methylamine
    "CCN",                        # ethylamine
    "CC(C)N",                     # isopropylylamine
    "c1ccccc1N",                  # aniline
    "Nc1ccc(O)cc1",               # 4-aminophenol
    "C1CCNCC1",                   # piperidine
    "C1COCCN1",                   # morpholine
    "CCN(CC)CC",                  # diethylamine
    "Nc1ccccc1C(=O)O",            # anthranilic acid
    # aryl halides (Suzuki / SNAr partners)
    "Brc1ccccc1",                 # bromobenzene
    "Clc1ccccc1",                 # chlorobenzene
    "Brc1ccncc1",                 # 2-bromopyridine
    # boronic acids
    "B(O)(O)c1ccccc1",            # phenylboronic acid
    # bare scaffolds
    "c1ccccc1",                   # benzene
    "c1ccncc1",                   # pyridine
    "c1ccsc1",                    # thiophene
    "C1CCCCC1",                   # cyclohexane
    "c1ccc2ccccc2c1",             # naphthalene
    # alkyl halides / sulfonyl chloride
    "CCBr",                       # bromoethane
    "BrCCBr",                     # 1,2-dibromoethane
    "CS(=O)(=O)Cl",               # methanesulfonyl chloride
)


def canonical(mol: Chem.Mol | str) -> str:
    """Canonical isomeric SMILES; explicit-H graphs (SDF input) are collapsed.

    Generated molecules come off an SDF with explicit H atoms while building
    blocks are written as SMILES without them — without RemoveHs the same
    compound would canonicalize two ways and never match.
    """
    if isinstance(mol, str):
        parsed = Chem.MolFromSmiles(mol)
        if parsed is None:
            raise ValueError(f"unparseable SMILES: {mol!r}")
        return Chem.MolToSmiles(parsed)
    work = mol
    try:
        work = Chem.RemoveHs(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(work)


@dataclass
class RouteStep:
    """One disconnection: product built from two parents via a template."""
    template: str
    left: str   # canonical SMILES of a parent (a building block at leaves)
    right: str


@dataclass
class RouteResult:
    feasible: bool
    steps: int | None = None
    leaves: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)   # leaf-ward order
    intermediates: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible, "steps": self.steps,
            "leaves": self.leaves, "templates": self.templates,
            "intermediates": self.intermediates, "note": self.note,
        }


_PARSE_CACHE: dict[str, Chem.Mol] = {}
_TEMPLATE_CACHE: dict[tuple[str, ...], list] = {}


def _rxns_for(template_names: tuple[str, ...]) -> list:
    """Compile the requested templates once, in REACTION_TEMPLATES order."""
    if template_names not in _TEMPLATE_CACHE:
        wanted = set(template_names)
        _TEMPLATE_CACHE[template_names] = [
            AllChem.ReactionFromSmarts(smarts)
            for name, smarts in REACTION_TEMPLATES if name in wanted
        ]
    return _TEMPLATE_CACHE[template_names]


class _Index:
    """Product index: canonical SMILES -> RouteStep, built breadth-first.

    Pairing policy (the conservative core): each layer pairs building blocks
    against the current layer in BOTH orders, but never layer x layer —
    that is where the search space explodes, and for a readiness gate a
    miss must fail safe (report infeasible) rather than run unbounded.
    """

    def __init__(self, blocks: tuple[str, ...], template_names: tuple[str, ...],
                 max_steps: int, max_intermediates: int):
        self.blocks = blocks
        self.step_of: dict[str, int] = {}
        self.route: dict[str, RouteStep] = {}
        self.block_mols: dict[str, Chem.Mol] = {}
        for smi in blocks:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                # Key by *canonical* SMILES: route provenance stores
                # canonical strings, so a non-canonical input block would
                # otherwise be unreachable in the walk (KeyError).
                can = Chem.MolToSmiles(m)
                self.block_mols[can] = m
                self.step_of[can] = 0
        wanted = set(template_names)
        # _rxns_for filters to `wanted` in REACTION_TEMPLATES order; filter
        # the names the same way so name/rxn stay aligned (zipping the full
        # name list against the filtered rxn list would mislabel templates).
        active = list(zip(
            [n for n, _ in REACTION_TEMPLATES if n in wanted],
            _rxns_for(template_names),
        ))
        current: list[str] = list(self.block_mols)
        self.capped = False
        self.max_steps_reached = 0
        for step in range(1, max_steps + 1):
            nxt: dict[str, RouteStep] = {}
            blocks_list = list(self.block_mols)
            for a_smi in blocks_list:
                mol_a = self.block_mols[a_smi]
                for b_smi in current:
                    if b_smi in self.block_mols:
                        mol_b = self.block_mols[b_smi]
                    else:
                        mol_b = _PARSE_CACHE.get(b_smi)
                        if mol_b is None:
                            mol_b = Chem.MolFromSmiles(b_smi)
                            if mol_b is None:
                                continue
                            _PARSE_CACHE[b_smi] = mol_b
                    for _name, rxn in active:
                        for pair in ((mol_a, mol_b), (mol_b, mol_a)):
                            try:
                                products = rxn.RunReactants(pair)
                            except Exception:
                                continue
                            for prod_tuple in products:
                                if not prod_tuple:
                                    continue
                                prod = prod_tuple[0]
                                try:
                                    Chem.SanitizeMol(prod)
                                    can = Chem.MolToSmiles(prod)
                                except Exception:
                                    continue
                                if can in self.step_of or can in nxt:
                                    continue
                                left = a_smi if pair[0] is mol_a else b_smi
                                right = b_smi if pair[1] is mol_b else a_smi
                                nxt[can] = RouteStep(_name,
                                                     canonical(left),
                                                     canonical(right))
            if not nxt:
                break
            # Record EVERY product found: the cap must never discard a
            # route that was actually discovered (that would turn a
            # conservative bound into arbitrary loss). The cap applies only
            # to the *pairing pool* for the next round — that is where the
            # search space explodes. sorted() keeps which products survive
            # into the pool reproducible (no set-iteration drift), and a
            # truncated pool is conservative by construction: a missed
            # route reports infeasible, never the reverse.
            ordered = sorted(nxt)
            truncated = len(ordered) > max_intermediates
            if truncated and step < max_steps:
                # Rounds remain: growth from the cut products is abandoned,
                # so a later infeasibility is provisional (flagged).
                self.capped = True
            for can in ordered:
                self.step_of[can] = step
                self.route[can] = nxt[can]
                if can not in _PARSE_CACHE:
                    m = Chem.MolFromSmiles(can)
                    if m is not None:
                        _PARSE_CACHE[can] = m
            current = ordered[:max_intermediates] if truncated else ordered
            self.max_steps_reached = step


@lru_cache(maxsize=8)
def _index(blocks: tuple[str, ...], template_names: tuple[str, ...],
           max_steps: int, max_intermediates: int) -> _Index:
    return _Index(blocks, template_names, max_steps, max_intermediates)


def find_route(
    target: Chem.Mol | str | None,
    building_blocks: Sequence[str] = DEFAULT_BUILDING_BLOCKS,
    templates: Sequence[str] | None = None,
    max_steps: int = 2,
    max_intermediates: int = 400,
) -> RouteResult:
    """Can ``target`` be assembled from ``building_blocks`` within max_steps?

    Returns a :class:`RouteResult`. ``steps == 0`` means the target IS a
    building block (purchasable as drawn). ``leaves`` are the building
    blocks used and ``templates`` the net transformation applied per step,
    so a feasible verdict is auditable rather than a bare boolean.

    Conservative by construction (module docstring): every product found is
    recorded, but the *growth pool* between rounds is capped at
    ``max_intermediates`` (sorted deterministically), so ``note`` on an
    infeasible verdict says when a bound was hit — "we stopped growing" must
    never be reported as a proven impossibility.
    """
    if target is None:
        return RouteResult(False, note="None target")
    if templates is None:
        template_names = tuple(name for name, _ in REACTION_TEMPLATES)
    else:
        template_names = tuple(templates)
    try:
        target_can = canonical(target)
    except ValueError as exc:
        return RouteResult(False, note=f"unparseable target: {exc}")

    blocks = tuple(building_blocks)
    idx = _index(blocks, template_names, int(max_steps),
                 int(max_intermediates))

    if target_can not in idx.step_of:
        note = "no template path within bounds"
        if idx.capped:
            note = ("intermediate layer hit its cap; infeasible is "
                    "conservative, not proven")
        return RouteResult(False, note=note)

    step_n = idx.step_of[target_can]
    if step_n == 0:
        return RouteResult(True, steps=0, leaves=[target_can],
                           note="is a building block")

    # Walk provenance back to the leaves.
    leaves: set[str] = set()
    tpls: set[str] = set()
    inters: set[str] = set()
    frontier = [target_can]
    seen: set[str] = set()
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        if idx.step_of.get(node, 99) == 0:
            leaves.add(node)
            continue
        step = idx.route[node]
        tpls.add(step.template)
        for parent in (step.left, step.right):
            if idx.step_of.get(parent, 99) == 0:
                leaves.add(parent)
            else:
                inters.add(parent)
                frontier.append(parent)
    # A found route is verified — a pool cap elsewhere never invalidates it.
    return RouteResult(
        True, steps=step_n, leaves=sorted(leaves),
        templates=sorted(tpls), intermediates=sorted(inters),
    )


def feasibility_rate(
    mols: Iterable[Chem.Mol | None],
    **kwargs,
) -> dict[str, float]:
    """Cohort feasibility over valid molecules (MOSES denominator convention)."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return {"RouteFeasible%": 0.0, "MeanRouteSteps": 0.0,
                "AlreadyBuildingBlock%": 0.0}
    results = [find_route(m, **kwargs) for m in valid]
    feas = [r for r in results if r.feasible]
    steps = [r.steps for r in feas if r.steps is not None]
    return {
        "RouteFeasible%": 100.0 * len(feas) / len(valid),
        "MeanRouteSteps": float(sum(steps) / len(steps)) if steps else 0.0,
        "AlreadyBuildingBlock%": 100.0 * sum(
            1 for r in results if r.feasible and r.steps == 0) / len(valid),
    }
