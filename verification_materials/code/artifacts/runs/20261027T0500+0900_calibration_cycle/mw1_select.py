"""MW1 selection exactly as prereg_b2_mw1.txt (rule written 04:54, before results): a per-attribute member-weight single is
kept only if v_hi15 > 0, v_oof > 0, t_hi15 > 0, t_oof > 0 and no source x multiplicity cell below -3 bps on either
surface; per attribute keep at most one direction (larger min of the four deltas). Writes cfg_mw1_combo.json (ONE config
combining the kept singles on top of ref) and mw1_select.json.
usage: mw1_select.py   (reads res_r2_{val,train}_s*/<name>.npz)
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FR = Path("/home/syouya/ml/lab/attribute_based_person_retrieval/artifacts/runs/20261026T2100+0900_final_round_search")
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(FR))
import fr_gate as g  # noqa: E402  (surface query ids, counts, sources, cells())


def per_query(surface: str) -> dict[str, np.ndarray]:
    out = {}
    for f in glob.glob(str(HERE / f"res_r2_{surface}_s*" / "*.npz")):
        out[Path(f).stem] = np.load(f)["oof"]
    return out


def main() -> None:
    pv, pt = per_query("val"), per_query("train")
    cfgs = {c["name"]: c for c in json.loads((HERE / "cfg_round2.json").read_text())}
    singles = sorted(n for n in cfgs if n.startswith("mw_a"))
    missing = [n for n in singles + ["ref"] if n not in pv or n not in pt]
    if missing:
        raise SystemExit(f"missing per-query results: {missing}")
    rows = []
    for n in singles:
        dv, dt = (pv[n] - pv["ref"]) * 1e4, (pt[n] - pt["ref"]) * 1e4
        r = {"name": n, "v_oof": float(dv.mean()), "v_hi15": float(dv[g.v_cnt >= 15].mean()),
             "t_oof": float(dt.mean()), "t_hi15": float(dt[g.t_cnt >= 15].mean()),
             "min_cell_v": min(v for v, _ in g.cells(dv, g.v_qs, g.v_cnt).values()),
             "min_cell_t": min(v for v, _ in g.cells(dt, g.t_qs, g.t_cnt).values())}
        r["min4"] = min(r["v_oof"], r["v_hi15"], r["t_oof"], r["t_hi15"])
        r["keep"] = r["min4"] > 0 and r["min_cell_v"] >= -3 and r["min_cell_t"] >= -3
        rows.append(r)
    best: dict[str, dict] = {}
    for r in rows:
        if r["keep"]:
            a = r["name"].split("_")[1]
            if a not in best or r["min4"] > best[a]["min4"]:
                best[a] = r
    kept = sorted(best.values(), key=lambda r: -r["min4"])
    combo = json.loads(json.dumps(cfgs["ref"]))
    combo["name"] = "mw1_combo"
    combo["dw"] = {}
    for r in kept:
        combo["dw"].update(cfgs[r["name"]]["dw"])
    (HERE / "cfg_mw1_combo.json").write_text(json.dumps([cfgs["ref"], combo], indent=1) + "\n")
    (HERE / "mw1_select.json").write_text(json.dumps({"kept": kept, "all": rows}, indent=1) + "\n")
    for r in kept:
        print(f"KEEP {r['name']}: v_oof {r['v_oof']:+.2f} v_hi15 {r['v_hi15']:+.2f} t_oof {r['t_oof']:+.2f} "
              f"t_hi15 {r['t_hi15']:+.2f} cells v {r['min_cell_v']:+.1f} t {r['min_cell_t']:+.1f}")
    print(f"{len(kept)} kept of {len(rows)}; combo dw = {combo['dw']}")


if __name__ == "__main__":
    main()
