# Trained weights

| File | Size | SHA-256 | Download |
|---|---|---|---|
| `B156C_B155C_K2_RERANK.zip` (exact Codabench final submission 1011976: `run.py`, 3 TorchScript networks, heads, config, licences) | 1,333,500,417 bytes | `1279da344c73aa8c67ee55703ba7e2c6e7c1b9607e3a182e9c9e70499fb30844` | https://drive.google.com/drive/folders/11rweLAHfE9Guxz0lcI38KrlAe3X4Bgw_?usp=sharing |

Place the file in this folder (`weights/`). It is not stored in git because of its size.

## Reference run

The exact ZIP's `run.py` (`load_model()` then `rank_gallery()`, as in `docker/run_inference.sh`) on the public validation split (3,462 queries x 33,407 images, whole split as one gallery), run on 2026-10-27 with torch 2.8 on a shared RTX PRO 6000 (`UPAR_TIME_SCALE=4` because the GPU was shared), scored with `docker/evaluate_distances.py`:

| mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
|---|---|---|---|---|---|
| 0.837696 | 0.492363 | 0.480647 | 0.677354 | 0.741190 | 0.378281 |

`report.json`: status PASS (bit-identical to an independent reference implementation), distances 3462 x 33407 float32, SHA-256 of `distances.npy`
`33260cffcf110fb7c1d857f729eb31f66d58d477a68470ff3350221e2f615765`. The final models were trained on train+validation,
so these numbers check execution, not accuracy. Different GPUs/torch versions may change the last bits of the
distances; the metrics should agree to about 1e-4.
