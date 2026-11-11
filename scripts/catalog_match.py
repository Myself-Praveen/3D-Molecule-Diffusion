"""Step 2 of lead triage: is this molecule already known or already for sale?

For each saturated lead we ask two public catalogs:

* **PubChem** — exact structure match by InChIKey, then any CAS-style registry
  numbers among that compound's synonyms. An exact hit means the "novel lead"
  is an existing compound with a vendor page.
* **ChEMBL** — similarity search at a Tanimoto cutoff; reports the closest
  known compound, its ChEMBL id and its measured activity count. A close analog
  means the series is explored and resynthesis is cheap.

Exact matches are what matter for procurement: a hit means buy it, a miss means
the molecule has to be made (Enamine REAL / on-demand), which is why the script
also writes a ``.smi`` upload file.

Usage
-----
    python scripts/catalog_match.py --input outputs/lead_triage/step1_saturated.smi \
        --outdir outputs/lead_triage --cutoff 85
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
HEADERS = {"User-Agent": "lead-triage/1.0 (molecule diffusion project)"}


def _get(url: str, *, params: dict | None = None, timeout: int = 30,
         attempts: int = 3):
    """GET with retries. Both catalogs intermittently return proxy 5xx pages
    or stall, which must not be reported as a "no hit"."""
    last = "unknown"
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            last = f"network: {type(exc).__name__}"
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None, "not_found"
        if r.status_code != 200:
            last = f"http_{r.status_code}"
            time.sleep(3 * (attempt + 1))
            continue
        try:
            return r.json(), "ok"
        except ValueError:
            last = "bad_json"
            time.sleep(3 * (attempt + 1))
    return None, last


def pubchem_exact(inchikey: str) -> dict:
    """Return {'cids': [...], 'cas': [...], 'names': [...]} for an exact match."""
    data, status = _get(f"{PUBCHEM}/compound/inchikey/{inchikey}/cids/JSON")
    if status != "ok":
        return {"status": status, "cids": [], "cas": [], "names": []}
    cids = data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return {"status": "no_hit", "cids": [], "cas": [], "names": []}

    cid = cids[0]
    cas, names = [], []
    syn, syn_status = _get(f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON")
    if syn_status == "ok":
        names = syn.get("InformationList", {}).get("Information", [{}])[0].get(
            "Synonym", [])
        cas = [s for s in names if CAS_RE.match(s)]
    return {"status": "hit", "cids": cids[:3], "cas": cas[:3], "names": names[:5]}


def chembl_similarity(smiles: str, cutoff: int) -> dict:
    """Closest ChEMBL compound at >= cutoff% Tanimoto."""
    # ChEMBL 500s on unescaped SMILES characters such as ()[]@=, so quote fully
    encoded = urllib.parse.quote(smiles, safe="")
    data, status = _get(f"{CHEMBL}/similarity/{encoded}/{cutoff}",
                        params={"format": "json", "limit": 1})
    if status != "ok":
        return {"status": status}
    mols = data.get("molecules", [])
    if not mols:
        return {"status": "no_hit"}
    top = mols[0]
    acts = 0
    act_data, act_status = _get(
        f"{CHEMBL}/activity",
        params={"molecule_chembl_id": top.get("molecule_chembl_id"),
                "format": "json", "limit": 1},
    )
    if act_status == "ok":
        acts = act_data.get("page_meta", {}).get("total_count", 0)
    return {
        "status": "hit",
        "id": top.get("molecule_chembl_id"),
        "pref_name": top.get("pref_name"),
        "max_similarity": top.get("similarity"),
        "max_phase": top.get("max_phase"),
        "n_activities": acts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, required=True,
                    help=".smi file from step 1 (SMILES<TAB>name)")
    ap.add_argument("--outdir", type=str, default="outputs/lead_triage")
    ap.add_argument("--cutoff", type=int, default=85,
                    help="ChEMBL Tanimoto cutoff in percent (85 = 0.85)")
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="pause between HTTP calls (be polite to public APIs)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        smiles, name = (line.split("\t") + [""])[:2]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({"name": name, "smiles": smiles, "error": "unparseable"})
            continue
        inchikey = Chem.MolToInchiKey(mol)
        pc = pubchem_exact(inchikey)
        time.sleep(args.sleep)
        cb = chembl_similarity(smiles, args.cutoff)
        time.sleep(args.sleep)

        rows.append({
            "name": name,
            "smiles": smiles,
            "inchikey": inchikey,
            "pubchem_status": pc["status"],
            "pubchem_cids": ";".join(str(c) for c in pc["cids"]),
            "cas": ";".join(pc["cas"]),
            "chembl_status": cb.get("status"),
            "chembl_id": cb.get("id", ""),
            "chembl_name": cb.get("pref_name") or "",
            "chembl_max_sim": cb.get("max_similarity", ""),
            "chembl_max_phase": cb.get("max_phase", ""),
            "chembl_activities": cb.get("n_activities", ""),
        })
        print(f"{name}: pubchem={pc['status']} chembl={cb.get('status')}"
              f"{' sim=' + str(cb.get('max_similarity')) if cb.get('status') == 'hit' else ''}")

    fields = ["name", "smiles", "inchikey", "pubchem_status", "pubchem_cids",
              "cas", "chembl_status", "chembl_id", "chembl_name",
              "chembl_max_sim", "chembl_max_phase", "chembl_activities", "error"]
    csv_path = outdir / "step2_catalog.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # upload file for Enamine REAL / Chemspace style on-demand search
    q_path = outdir / "enamine_query.smi"
    with q_path.open("w") as fh:
        for r in rows:
            if "smiles" in r and not r.get("error"):
                fh.write(f"{r['smiles']} {r['name']}\n")

    summary = {
        "n_leads": len(rows),
        "pubchem_exact_hits": sum(1 for r in rows if r.get("pubchem_status") == "hit"),
        "pubchem_with_cas": sum(1 for r in rows if r.get("cas")),
        "chembl_analog_hits": sum(1 for r in rows if r.get("chembl_status") == "hit"),
        "chembl_cutoff_pct": args.cutoff,
        "outputs": {"csv": str(csv_path), "enamine_query": str(q_path)},
    }
    (outdir / "step2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
