"""Gate v2 of prereg_k2.txt for the K2 configs vs mw1_combo (bps). Writes k2_gate.json.
usage: k2_gate.py"""
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
import fr_gate as g  # noqa: E402


def load(surface, name):
    return np.load(glob.glob(str(HERE / f"res_k2_{surface}_s*" / f"{name}.npz"))[0])["oof"]


def pm():
    rows = {}
    for f in ("res_k2_pm_a.jsonl", "res_k2_pm_b.jsonl"):
        for l in (HERE / f).read_text().splitlines():
            r = json.loads(l)
            rows[r["name"]] = r
    return rows


def main():
    names = [c["name"] for c in json.loads((HERE / "cfg_k2.json").read_text())][1:]
    vb, tb = load("val", "mw1_combo"), load("train", "mw1_combo")
    P = pm()
    rng = np.random.default_rng(0)
    out = []
    for n in names:
        r = {"name": n}
        for surf, base, qs, cnt in (("v", vb, g.v_qs, g.v_cnt), ("t", tb, g.t_qs, g.t_cnt)):
            d = (load("val" if surf == "v" else "train", n) - base) * 1e4
            r[f"{surf}_oof"], r[f"{surf}_hi15"], r[f"{surf}_lo14"] = float(d.mean()), float(d[cnt >= 15].mean()), float(d[cnt <= 4].mean())
            r[f"{surf}_src"] = {str(s): float(d[qs == s].mean()) for s in np.unique(qs)}
            zs = []
            for s in np.unique(qs):
                for lo, hi in g.BINS:
                    m = (qs == s) & (cnt >= lo) & (cnt <= hi)
                    zs.append(float(d[m].mean() / (d[m].std(ddof=1) / np.sqrt(m.sum()))))
            r[f"{surf}_worst_z"] = min(zs)
            if surf == "t":
                r["t_ci_all"] = g.boot(d, rng)[:2]
                r["t_ci_hi15"] = g.boot(d[cnt >= 15], rng)[:2]
        r["pm_MEV"] = (P[n]["MEV"] - P["mw1_combo"]["MEV"]) * 1e4
        r["pm_SYN"] = (P[n]["SYN"] - P["mw1_combo"]["SYN"]) * 1e4
        r["pm_PUB"] = (P[n]["PUB"] - P["mw1_combo"]["PUB"]) * 1e4
        checks = {"pos4": all(r[k] > 0 for k in ("v_hi15", "t_hi15", "v_oof", "t_oof")),
                  "ci": r["t_ci_all"][0] > 0 or r["t_ci_hi15"][0] > 0,
                  "src": min(list(r["v_src"].values()) + list(r["t_src"].values())) >= -5,
                  "cellz": min(r["v_worst_z"], r["t_worst_z"]) >= -2.5,
                  "mimic": r["pm_MEV"] >= 0 and r["pm_PUB"] >= -1}
        r["checks"], r["pass"] = checks, all(checks.values())
        r["rank_key"] = min(r["v_hi15"], r["t_hi15"])
        out.append(r)
        print(f"{n}: v hi15 {r['v_hi15']:+.1f} oof {r['v_oof']:+.1f} lo {r['v_lo14']:+.1f} | t hi15 {r['t_hi15']:+.1f} oof {r['t_oof']:+.1f} "
              f"lo {r['t_lo14']:+.1f} | CI all [{r['t_ci_all'][0]:+.1f},{r['t_ci_all'][1]:+.1f}] hi15 [{r['t_ci_hi15'][0]:+.1f},{r['t_ci_hi15'][1]:+.1f}] "
              f"| src min {min(list(r['v_src'].values()) + list(r['t_src'].values())):+.1f} z {r['v_worst_z']:+.2f}/{r['t_worst_z']:+.2f} "
              f"| PM MEV {r['pm_MEV']:+.1f} SYN {r['pm_SYN']:+.1f} PUB {r['pm_PUB']:+.1f} -> {'PASS' if r['pass'] else 'FAIL ' + str([k for k, v in checks.items() if not v])}")
    passers = [r for r in out if r["pass"]]
    win = max(passers, key=lambda r: r["rank_key"])["name"] if passers else None
    (HERE / "k2_gate.json").write_text(json.dumps({"rows": out, "winner": win}, indent=1) + "\n")
    print("WINNER", win)


if __name__ == "__main__":
    main()
