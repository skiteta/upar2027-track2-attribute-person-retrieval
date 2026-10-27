"""Build B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR (package_v2) from the exact B124C ZIP.

Changes from B124C_CONSTANT_RESCUE_ALPHA05.zip:
  - assets/config.json: candidate = "B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR", post.entropy_delta = 0.35,
    and documented b128_quadj_hybrid_coevolve_anchor metadata dict.
  - run.py: extracts same-forward EVA02-L/14 backbone pooled features (model.head.backbone) and
    3-member post-processed probability std, passing them with queries, attribute_names, and
    _b121_score_distances to apply_feature_rerank.
  - b121_runtime.py: delegates apply_feature_rerank to run_b128c_quadj_rerank inside numpy_blas_threads(2).
  - b127_jaccard_prob_diffusion.py: added self-contained module from this run dir.
  - Every other member in the archive is verified byte-for-byte identical and hardlinked in package_v2/package.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from b127_jaccard_prob_diffusion import B134C_CONFIG

PARENT_DIR = ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05/package_v1"
PARENT = PARENT_DIR / "B124C_CONSTANT_RESCUE_ALPHA05.zip"
PARENT_EXTRACTED = PARENT_DIR / "package"
PARENT_SHA = "5fae8b2b0d28ce4e313249e5cfb47c6627d0164aaac5a9a54c07288e88c20da7"
CANDIDATE = "B139C_ANAT_MULTIRES_SPHERICAL_HEAD_RERANK"
OUT = HERE / "package_v14"

RUN_REPLACES = [
    (
        "_GALLERY_FEATURES = None\n",
        "_GALLERY_FEATURES = None\n_GALLERY_EVA_FEATURES = None\n_GALLERY_HP_FEATURES = None\n_GALLERY_MEMBER_STD = None\n",
    ),
    (
        "    if spec.get('gallery_feature_head'):\n"
        "        if not blocks or spec.get('flip_tta') or spec.get('backbone') != 'pe_core_l14_280':\n"
        "            raise RuntimeError('feature head requires exact original-view PE block member')\n"
        "        model.feature_head = load_feature_head(_HERE, spec, device, dtype)\n",
        "    if spec.get('gallery_feature_head'):\n"
        "        if not blocks or spec.get('flip_tta') or spec.get('backbone') != 'pe_core_l14_280':\n"
        "            raise RuntimeError('feature head requires exact original-view PE block member')\n"
        "        model.feature_head = load_feature_head(_HERE, spec, device, dtype)\n"
        "    elif spec.get('backbone') == 'eva02_l14_252':\n"
        "        if not blocks or spec.get('flip_tta'):\n"
        "            raise RuntimeError('EVA feature head requires exact original-view EVA02 block member')\n"
        "        model.feature_head = model.head.backbone\n"
        "    elif spec.get('backbone') in ('dinov3_hp16_192', 'dinov3_hp16_224'):\n"
        "        if not blocks or spec.get('flip_tta'):\n"
        "            raise RuntimeError('DINOv3-H+ feature head requires exact original-view DINOv3-H+ block member')\n"
        "        model.feature_head = lambda h_in, _norm=model.head.backbone.norm: _norm(h_in)[:, 5:].mean(dim=1)\n",
    ),
    (
        "    return model if blocks else _maybe_fuse(model, spec, device)\n",
        "    out_m = model if blocks else _maybe_fuse(model, spec, device)\n"
        "    out_m._mean = torch.tensor(spec[\"mean\"], dtype=torch.float32, device=device).view(1, 3, 1, 1)\n"
        "    out_m._std = torch.tensor(spec[\"std\"], dtype=torch.float32, device=device).view(1, 3, 1, 1)\n"
        "    out_m._correction = torch.log(torch.tensor(spec[\"pos_weight\"], dtype=torch.float32, device=device))\n"
        "    out_m._dtype = next(out_m.parameters()).dtype\n"
        "    return out_m\n",
    ),
    (
        "    mean = torch.tensor(spec[\"mean\"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)\n"
        "    std = torch.tensor(spec[\"std\"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)\n"
        "    x = ((x - mean) / std).contiguous()\n"
        "    correction = torch.log(torch.tensor(spec[\"pos_weight\"], dtype=torch.float32, device=x.device))\n"
        "    dtype = next(model.parameters()).dtype\n"
        "    x = x.to(dtype)\n",
        "    mean = getattr(model, \"_mean\", None)\n"
        "    if mean is None:\n"
        "        mean = torch.tensor(spec[\"mean\"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)\n"
        "    std = getattr(model, \"_std\", None)\n"
        "    if std is None:\n"
        "        std = torch.tensor(spec[\"std\"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)\n"
        "    x = ((x - mean) / std).contiguous()\n"
        "    correction = getattr(model, \"_correction\", None)\n"
        "    if correction is None:\n"
        "        correction = torch.log(torch.tensor(spec[\"pos_weight\"], dtype=torch.float32, device=x.device))\n"
        "    dtype = getattr(model, \"_dtype\", None) or next(model.parameters()).dtype\n"
        "    x = x.to(dtype)\n",
    ),
    (
        "    global _FALLBACK, _GALLERY_FEATURES\n",
        "    global _FALLBACK, _GALLERY_FEATURES, _GALLERY_EVA_FEATURES, _GALLERY_HP_FEATURES, _GALLERY_MEMBER_STD\n",
    ),
    (
        "    _GALLERY_FEATURES = np.full((total, 1024), np.nan, dtype=np.float32)\n",
        "    _GALLERY_FEATURES = np.full((total, 1024), np.nan, dtype=np.float32)\n"
        "    _GALLERY_EVA_FEATURES = np.full((total, 1024), np.nan, dtype=np.float32)\n"
        "    _GALLERY_HP_FEATURES = np.full((total, 1280), np.nan, dtype=np.float32)\n",
    ),
    (
        "            if specs[i].get('gallery_feature_head'):\n"
        "                feature = models[i].last_features\n"
        "                if feature is None or feature.ndim != 2 or feature.shape[1] != 1024 or feature.shape[0] < raw.shape[0]:\n"
        "                    raise RuntimeError('missing or incomplete same-forward PE pooled features')\n"
        "                _GALLERY_FEATURES[start:start + raw.shape[0]] = feature[:raw.shape[0]].float().cpu().numpy()\n",
        "            if specs[i].get('gallery_feature_head'):\n"
        "                feature = models[i].last_features\n"
        "                if feature is None or feature.ndim != 2 or feature.shape[1] != 1024 or feature.shape[0] < raw.shape[0]:\n"
        "                    raise RuntimeError('missing or incomplete same-forward PE pooled features')\n"
        "                _GALLERY_FEATURES[start:start + raw.shape[0]] = feature[:raw.shape[0]].float().cpu().numpy()\n"
        "            elif specs[i].get('backbone') == 'eva02_l14_252':\n"
        "                eva_feat = models[i].last_features\n"
        "                if eva_feat is None or eva_feat.ndim != 2 or eva_feat.ndim != 2 or eva_feat.shape[1] != 1024 or eva_feat.shape[0] < raw.shape[0]:\n"
        "                    raise RuntimeError('missing or incomplete same-forward EVA02 pooled features')\n"
        "                _GALLERY_EVA_FEATURES[start:start + raw.shape[0]] = eva_feat[:raw.shape[0]].float().cpu().numpy()\n"
        "            elif specs[i].get('backbone') in ('dinov3_hp16_192', 'dinov3_hp16_224'):\n"
        "                hp_feat = models[i].last_features\n"
        "                if hp_feat is None or hp_feat.ndim != 2 or hp_feat.shape[1] != 1280 or hp_feat.shape[0] < raw.shape[0]:\n"
        "                    raise RuntimeError('missing or incomplete same-forward DINOv3-H+ pooled features')\n"
        "                _GALLERY_HP_FEATURES[start:start + raw.shape[0]] = hp_feat[:raw.shape[0]].float().cpu().numpy()\n",
    ),
    (
        "    if not np.isfinite(_GALLERY_FEATURES).all():\n"
        "        raise RuntimeError('incomplete/nonfinite PE feature gallery')\n"
        "    return out, \"+\".join(s[\"name\"] for s in specs[:active])\n",
        "    if not np.isfinite(_GALLERY_FEATURES).all() or not np.isfinite(_GALLERY_EVA_FEATURES).all() or not np.isfinite(_GALLERY_HP_FEATURES).all():\n"
        "        raise RuntimeError('incomplete/nonfinite PE, EVA02, or DINOv3-H+ feature gallery')\n"
        "    _GALLERY_MEMBER_STD = np.std(\n"
        "        np.stack([_apply_post(p_m, _CONFIG.get('post')) for p_m in probs[:active]], axis=0),\n"
        "        axis=0,\n"
        "    ).astype(np.float32)\n"
        "    return out, \"+\".join(s[\"name\"] for s in specs[:active])\n",
    ),
    (
        "    return np.ascontiguousarray(out[:, reorder])\n",
        "    global _GALLERY_MEMBER_STD\n"
        "    if _GALLERY_MEMBER_STD is not None:\n"
        "        _GALLERY_MEMBER_STD = np.ascontiguousarray(_GALLERY_MEMBER_STD[:, reorder])\n"
        "    return np.ascontiguousarray(out[:, reorder])\n",
    ),
    (
        "def _b121_score_distances(probabilities, queries, attribute_names):\n"
        "    names = list(_CONFIG[\"attribute_names\"])\n"
        "    reorder = np.asarray([names.index(n) for n in attribute_names], dtype=np.int64)\n"
        "    weights = np.asarray(_CONFIG[\"attribute_distance_weights\"], dtype=np.float32)[reorder]\n"
        "    eps = np.float32(_CONFIG[\"bernoulli_nll_epsilon\"])\n"
        "    lam = np.float32(_CONFIG[\"bernoulli_nll_lambda\"])\n"
        "    ent = (_CONFIG.get(\"post\") or {}).get(\"entropy_delta\")\n"
        "    if ent:  # per-image entropy weights (identical to upar/postproc.py entropy_weight_matrix); absent = per-attribute weights\n"
        "        q = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)\n"
        "        h = np.exp((1.0 - (-(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0))) * float(ent))\n"
        "        weights = (weights.astype(np.float64)[None, :] * (h / h.mean(axis=1, keepdims=True))).astype(np.float32)\n"
        "    clipped = np.clip(probabilities, eps, np.float32(1.0) - eps)\n"
        "    neg = -np.log1p(-clipped).astype(np.float32, copy=False)\n"
        "    pos = -np.log(clipped).astype(np.float32, copy=False)\n"
        "    base = (np.float32(1.0) - lam) * (probabilities * weights).sum(axis=1, dtype=np.float32) \\\n"
        "        + lam * (neg * weights).sum(axis=1, dtype=np.float32)\n"
        "    delta = (np.float32(1.0) - lam) * weights * (np.float32(1.0) - np.float32(2.0) * probabilities) \\\n"
        "        + lam * weights * (pos - neg)\n",
        "def _b121_score_distances(probabilities, queries, attribute_names, ent_override=None, lam_override=None, err_gamma=1.0, uniform_weights=False, lam_neg_override=None, err_gamma_neg=None, q_pos_rho=0.0, dich_sym_loss=0.0, dich_pos_nll_add=0.0, pos_q_mul=1.0, _pre=None):\n"
        "    names = list(_CONFIG[\"attribute_names\"])\n"
        "    reorder = np.asarray([names.index(n) for n in attribute_names], dtype=np.int64)\n"
        "    if uniform_weights:\n"
        "        weights = np.ones(len(attribute_names), dtype=np.float32)\n"
        "    else:\n"
        "        weights = np.asarray(_CONFIG[\"attribute_distance_weights\"], dtype=np.float32)[reorder]\n"
        "    eps = np.float32(_CONFIG[\"bernoulli_nll_epsilon\"])\n"
        "    lam = np.float32(_CONFIG[\"bernoulli_nll_lambda\"] if lam_override is None else lam_override)\n"
        "    lam0 = lam if lam_neg_override is None else np.float32(lam_neg_override)\n"
        "    ent = (_CONFIG.get(\"post\") or {}).get(\"entropy_delta\") if ent_override is None else ent_override\n"
        "    if _pre is not None:\n"
        "        one_minus_ent, clipped, neg, pos = _pre\n"
        "    else:\n"
        "        one_minus_ent = None\n"
        "        clipped = np.clip(probabilities, eps, np.float32(1.0) - eps)\n"
        "        neg = -np.log1p(-clipped).astype(np.float32, copy=False)\n"
        "        pos = -np.log(clipped).astype(np.float32, copy=False)\n"
        "    if ent:  # per-image entropy weights (identical to upar/postproc.py entropy_weight_matrix); absent = per-attribute weights\n"
        "        if one_minus_ent is None:\n"
        "            q = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)\n"
        "            one_minus_ent = 1.0 - (-(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0))\n"
        "        h = np.exp(one_minus_ent * float(ent))\n"
        "        weights = (weights.astype(np.float64)[None, :] * (h / h.mean(axis=1, keepdims=True))).astype(np.float32)\n"
        "    eg_pos = float(err_gamma)\n"
        "    eg_neg = eg_pos if err_gamma_neg is None else float(err_gamma_neg)\n"
        "    pmul = np.float32(pos_q_mul)\n"
        "    if eg_pos != 1.0 or eg_neg != 1.0 or lam != lam0 or pmul != 1.0 or float(q_pos_rho) != 0.0 or float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:\n"
        "        f0 = (clipped ** np.float32(eg_neg)).astype(np.float32, copy=False) if eg_neg != 1.0 else probabilities\n"
        "        f1 = ((np.float32(1.0) - clipped) ** np.float32(eg_pos)).astype(np.float32, copy=False) if eg_pos != 1.0 else (np.float32(1.0) - probabilities)\n"
        "        loss0 = (np.float32(1.0) - lam0) * f0 + lam0 * neg\n"
        "        loss1 = pmul * ((np.float32(1.0) - lam) * f1 + lam * pos)\n"
        "        if float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:\n"
        "            dich_idx = [i for i, n in enumerate(attribute_names) if str(n) in (\"Gender-Female\", \"UpperBody-Length-Short\", \"LowerBody-Length-Short\")]\n"
        "            if float(dich_sym_loss) > 0.0 and dich_idx:\n"
        "                s_d = np.float32(dich_sym_loss)\n"
        "                lam_d0 = np.float32(float(lam) + float(dich_pos_nll_add))\n"
        "                f0_pos = (clipped[:, dich_idx] ** np.float32(eg_pos)).astype(np.float32, copy=False) if eg_pos != 1.0 else probabilities[:, dich_idx]\n"
        "                loss0_pos = pmul * ((np.float32(1.0) - lam_d0) * f0_pos + lam_d0 * neg[:, dich_idx])\n"
        "                loss0 = loss0.copy()\n"
        "                loss0[:, dich_idx] = (np.float32(1.0) - s_d) * loss0[:, dich_idx] + s_d * loss0_pos\n"
        "            if float(dich_pos_nll_add) != 0.0 and dich_idx:\n"
        "                dl_p = np.float32(dich_pos_nll_add)\n"
        "                loss1 = loss1.copy()\n"
        "                loss1[:, dich_idx] = loss1[:, dich_idx] + pmul * dl_p * (pos[:, dich_idx] - f1[:, dich_idx])\n"
        "        d0 = weights * loss0\n"
        "        d1 = weights * loss1\n"
        "        base = d0.sum(axis=1, dtype=np.float32)\n"
        "        delta = d1 - d0\n"
        "    else:\n"
        "        base = (np.float32(1.0) - lam) * (probabilities * weights).sum(axis=1, dtype=np.float32) \\\n"
        "            + lam * (neg * weights).sum(axis=1, dtype=np.float32)\n"
        "        delta = (np.float32(1.0) - lam) * weights * (np.float32(1.0) - np.float32(2.0) * probabilities) \\\n"
        "            + lam * weights * (pos - neg)\n",
    ),
    (
        "    distances = apply_feature_rerank(distances, _GALLERY_FEATURES, probabilities, _b121_stage_log)\n"
        "    if time.time() - _T0 >= runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE):\n"
        "        raise RuntimeError('feature reranking exceeded fixed total budget; fail closed')\n",
        "    distances = apply_feature_rerank(\n"
        "        distances, _GALLERY_FEATURES, probabilities, _b121_stage_log,\n"
        "        queries, attribute_names, _b121_score_distances,\n"
        "        _GALLERY_EVA_FEATURES, _GALLERY_HP_FEATURES, _GALLERY_MEMBER_STD,\n"
        "    )\n"
        "    if time.time() - _T0 >= runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE) + 6.0:\n"
        "        raise RuntimeError('feature reranking exceeded fixed total budget; fail closed')\n",
    ),
]

RUNTIME_OLD_IMPORT = "from b121_fp32_cosine_graph import build_feature_graph\n"
RUNTIME_NEW_IMPORT = (
    "from b121_fp32_cosine_graph import build_feature_graph\n"
    "from b127_jaccard_prob_diffusion import run_b132c_pentj_body_rerank\n"
)

RUNTIME_OLD_FUNC = (
    "def apply_feature_rerank(distances, features, probabilities, log):\n"
    "    if features is None or features.shape != (len(probabilities), 1024):\n"
    "        raise RuntimeError('missing complete same-forward PE pooled gallery')\n"
    "    if not torch.cuda.is_available():\n"
    "        raise RuntimeError('mandatory CUDA FP32 unit-dot graph unavailable; no CPU fallback')\n"
    "    started = time.time()\n"
    "    log('feature_rerank_enter', device='cuda', dtype='float32', normalization='float64', tf32=False)\n"
    "    graph, original = build_feature_graph(features, k=20, chunk_size=256,\n"
    "                                          device='cuda', allow_gpu=True, log=log)\n"
    "    graph_seconds = time.time()-started\n"
    "    with numpy_blas_threads(2) as blas:\n"
    "        log('feature_gate_enter', numpy_blas=blas)\n"
    "        retained, count = retain_identical_predicted_profiles(probabilities, graph, original)\n"
    "        rescue = constant_rescue(probabilities, graph, original)\n"
    "        log('feature_smooth_enter', original_edges=int(original.sum()), kept_edges=int(count.sum()),\n"
    "            rescued_edges=int(np.count_nonzero(rescue)), rescue_constant=CONSTANT, symmetric=True, fitting=False, diffusion_alpha=.5)\n"
    "        output = smooth_with_rescue(distances, probabilities, graph, original, rescue, alpha=.5)\n"
    "    if not np.isfinite(output).all() or output.shape != distances.shape:\n"
    "        raise RuntimeError('invalid final gated distances')\n"
    "    log('feature_reranked', graph_seconds=round(graph_seconds, 4), total_seconds=round(time.time()-started, 4),\n"
    "        original_edges=int(original.sum()), kept_edges=int(count.sum()), isolated=int((count==0).sum()),\n"
    "        extra_encoder_forward=False, extra_pool_head_forward=True, numpy_blas=blas,\n"
    "        graph_backend='CUDA_FP32_unit_dot_TF32_off_CPU_deterministic_topk')\n"
    "    return output\n"
)

RUNTIME_NEW_FUNC = (
    "def apply_feature_rerank(distances, features, probabilities, log, queries, attribute_names, score_fn, eva_features, hp_features, member_std):\n"
    "    if features is None or features.shape != (len(probabilities), 1024):\n"
    "        raise RuntimeError('missing complete same-forward PE pooled gallery')\n"
    "    if eva_features is None or eva_features.shape != (len(probabilities), 1024):\n"
    "        raise RuntimeError('missing complete same-forward EVA02 pooled gallery')\n"
    "    if hp_features is None or hp_features.shape != (len(probabilities), 1280):\n"
    "        raise RuntimeError('missing complete same-forward DINOv3-H+ pooled gallery')\n"
    "    if member_std is None or member_std.shape != probabilities.shape:\n"
    "        raise RuntimeError('missing complete member_std gallery')\n"
    "    if not torch.cuda.is_available():\n"
    "        raise RuntimeError('mandatory CUDA FP32 unit-dot graph unavailable; no CPU fallback')\n"
    "    started = time.time()\n"
    "    log('feature_rerank_enter', device='cuda', dtype='float32', normalization='float64', tf32=False)\n"
    "    with numpy_blas_threads(2) as blas:\n"
    "        output = run_b132c_pentj_body_rerank(\n"
    "            probabilities, features, eva_features, hp_features, member_std,\n"
    "            queries, attribute_names, score_fn, log,\n"
    "        )\n"
    "    if not np.isfinite(output).all() or output.shape != distances.shape:\n"
    "        raise RuntimeError('invalid final gated distances')\n"
    "    log('feature_reranked', total_seconds=round(time.time()-started, 4),\n"
    "        extra_encoder_forward=False, extra_pool_head_forward=True, numpy_blas=blas,\n"
    "        graph_backend='CUDA_FP32_unit_dot_TF32_off_CPU_deterministic_topk')\n"
    "    return output\n"
)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("expected exactly one occurrence: " + old[:120])
    return text.replace(old, new)


def build_anat10_member_weights() -> list[list[float]]:
    w_pe, w_hp, w_eva = [], [], []
    for j in range(40):
        if j in (4, 5, 6):
            pe, hp, eva = 0.425, 0.340, 0.235
        elif j in (0, 1, 2, 3, 37, 38, 39):
            pe, hp, eva = 0.420, 0.345, 0.235
        elif j in (7, 20, 36):
            pe, hp, eva = 0.355, 0.410, 0.235
        else:
            pe, hp, eva = 0.370, 0.395, 0.235
        w_pe.append(round(pe, 6))
        w_hp.append(round(hp, 6))
        w_eva.append(round(eva, 6))
    return [w_pe, w_hp, w_eva]


def main() -> None:
    parent_complete = json.loads((PARENT_DIR / "complete.json").read_text())
    assert parent_complete["zip"]["sha256"] == PARENT_SHA, "exact B124C parent complete.json required"
    parent_hashes = {m["member"]: m["new_sha256"] for m in parent_complete["member_proof"]}
    names = sorted(parent_hashes.keys())
    for name, expected_h in parent_hashes.items():
        if sha(PARENT_EXTRACTED / name) != expected_h:
            raise ValueError(f"parent extracted file modified: {name}")
    assert not OUT.exists(), "fresh output only"
    started = time.monotonic()
    package = OUT / "package"
    package.mkdir(parents=True)
    traced_dir = HERE / "traced_b138c_dich3_ltype2_r00"
    canary_ref_0 = json.loads((traced_dir / "canary_ref_0.json").read_text())
    canary_ref_1 = json.loads((traced_dir / "canary_ref_1.json").read_text())
    canary_ref_2 = json.loads((traced_dir / "canary_ref_2.json").read_text())
    w_anat10 = build_anat10_member_weights()
    for name in names:
        dest = package / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        src_file = PARENT_EXTRACTED / name
        if name == "run.py":
            src = src_file.read_text()
            for old_s, new_s in RUN_REPLACES:
                src = replace_once(src, old_s, new_s)
            dest.write_bytes(src.encode())
        elif name == "b121_runtime.py":
            src = src_file.read_text()
            src = replace_once(src, RUNTIME_OLD_IMPORT, RUNTIME_NEW_IMPORT)
            src = replace_once(src, RUNTIME_OLD_FUNC, RUNTIME_NEW_FUNC)
            dest.write_bytes(src.encode())
        elif name == "assets/config.json":
            cfg = json.loads(src_file.read_text())
            assert cfg["candidate"] == "B124C_CONSTANT_RESCUE_ALPHA05" and cfg["gallery_rerank"]["alpha"] == 0.5
            assert cfg["post"]["entropy_delta"] == 0.1875
            cfg["candidate"] = CANDIDATE
            cfg["post"]["entropy_delta"] = 0.35
            cfg["models"][0]["training_run"] = "artifacts/runs/20261015T0410+0900_b116g_pe/soup_pe4q_all_s7_spherical_qk120_b060"
            cfg["models"][1]["training_run"] = "artifacts/runs/20261017T0345+0900_hplus6/hp_dich3_ltype2_r00_body12_m15_r10_all_spherical_o103"
            cfg["models"][2]["training_run"] = "artifacts/runs/20261014T0100+0900_b113g/soup_eva224s7_eva252s7_spherical_t12"
            for m_idx in range(3):
                cfg["models"][m_idx]["member_weights"] = w_anat10[m_idx]
            cfg["canary"]["ref"][0] = canary_ref_0
            cfg["canary"]["ref"][1] = canary_ref_1
            cfg["canary"]["ref"][2] = canary_ref_2
            cfg["b139_anat_multires_spherical_head_rerank"] = B134C_CONFIG
            dest.write_bytes((json.dumps(cfg, indent=2) + "\n").encode())
        elif name in ("assets/model_0.weights.pt", "assets/model_1.weights.pt", "assets/model_2.weights.pt"):
            os.link(traced_dir / name, dest)
        else:
            os.link(src_file, dest)
    shutil.copyfile(HERE / "b127_jaccard_prob_diffusion.py", package / "b127_jaccard_prob_diffusion.py")
    artifact = OUT / (CANDIDATE + ".zip")
    with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for p in sorted(package.rglob("*")):
            if p.is_file():
                new.write(p, p.relative_to(package).as_posix())
    proof = []
    expected_changed = {
        "assets/config.json",
        "assets/model_0.weights.pt",
        "assets/model_1.weights.pt",
        "assets/model_2.weights.pt",
        "b121_runtime.py",
        "run.py",
        "b127_jaccard_prob_diffusion.py",
    }
    with zipfile.ZipFile(artifact) as new:
        new_names = sorted(i.filename for i in new.infolist() if not i.is_dir())
        if set(new_names) != set(names) | {"b127_jaccard_prob_diffusion.py"}:
            raise ValueError("member set must be parent + b127_jaccard_prob_diffusion.py")
        for name in new_names:
            after = sha_bytes(new.read(name))
            before = parent_hashes.get(name)
            changed = before != after
            if changed != (name in expected_changed):
                raise ValueError("unexpected member delta: " + name)
            proof.append({"member": name, "parent_sha256": before, "new_sha256": after, "changed": changed})
    cfg = json.loads((package / "assets/config.json").read_text())
    report = {
        "status": "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED",
        "candidate": CANDIDATE,
        "pid": os.getpid(),
        "b139_anat_multires_spherical_head_rerank": cfg["b139_anat_multires_spherical_head_rerank"],
        "zip": {"path": str(artifact), "sha256": sha(artifact), "size": artifact.stat().st_size},
        "parent": {"path": str(PARENT), "sha256": PARENT_SHA},
        "member_proof": proof,
        "config": cfg,
        "module": {
            "path": str(HERE / "b127_jaccard_prob_diffusion.py"),
            "sha256": sha(HERE / "b127_jaccard_prob_diffusion.py"),
        },
        "wall_seconds": time.monotonic() - started,
    }
    (OUT / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate": report["candidate"],
                "zip_sha256_prefix": report["zip"]["sha256"][:16],
                "changed_members": [m["member"] for m in proof if m["changed"]],
                "wall_seconds": round(report["wall_seconds"], 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

