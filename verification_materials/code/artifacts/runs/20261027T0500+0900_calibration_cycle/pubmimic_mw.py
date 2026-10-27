"""Public-structure mimic for configs with single-attribute member-weight groups ("a<j>" in "dw"), 10/27.
Runs pubmimic_eva3.py unchanged after adding the a<j> groups that fr_sweep2.py uses.
usage: pubmimic_mw.py CONFIGS.json OUT.jsonl   (env KNOB_PARAM as for fr_sweep2.py)"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/syouya/ml/lab/attribute_based_person_retrieval/artifacts/runs/20261026T2340+0900_eva_rankq_cycle")
import pubmimic_eva3 as pmx  # noqa: E402  (reads CONFIGS.json / OUT.jsonl from sys.argv at import)

pmx.cs.GROUPS.update({f"a{j}": [j] for j in range(40)})

if __name__ == "__main__":
    pmx.main()
