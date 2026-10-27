"""Turn the deployed package_v14 rerank module into a knob-parametrised copy (rerank_param.py).

Every literal is replaced by a K[...] lookup whose default equals the deployed literal, keeping the arithmetic form,
so K=None must reproduce the deployed distances bit-for-bit (checked by equiv_check_v1.py). Each replacement asserts
its exact occurrence count against the deployed source.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("/home/syouya/ml/lab/attribute_based_person_retrieval/artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/package_v14/package/b127_jaccard_prob_diffusion.py")
text = SRC.read_text()

REPL = [
    # graph builder: per-branch k
    ('    gamma_q: float = 2.6,\n    chunk_size: int = 512,\n) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:',
     '    gamma_q: float = 2.6,\n    chunk_size: int = 512,\n    ks: dict | None = None,\n) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:', 1),
    ('    specs = [("pe", 20), ("eva", 20), ("j25_k28", 28), ("j50_k28", 28), ("tri25_k28", 28)]',
     '    ks = ks or {"pe": 20, "eva": 20, "j25_k28": 28, "j50_k28": 28, "tri25_k28": 28}\n'
     '    specs = [(nm, ks[nm]) for nm in ("pe", "eva", "j25_k28", "j50_k28", "tri25_k28")]', 1),
    ('    pre_k = min(96, n - 1)', '    pre_k = min(max(96, max(ks.values()) + 8), n - 1)', 1),
    ('                ("pe", d_pe, 20),\n                ("eva", d_eva, 20),\n                ("j25_k28", d_j25, 28),\n                ("j50_k28", d_j50, 28),\n                ("tri25_k28", d_tri25, 28),',
     '                ("pe", d_pe, ks["pe"]),\n                ("eva", d_eva, ks["eva"]),\n                ("j25_k28", d_j25, ks["j25_k28"]),\n                ("j50_k28", d_j50, ks["j50_k28"]),\n                ("tri25_k28", d_tri25, ks["tri25_k28"]),', 1),
    # entry
    ('    score_fn,\n    log,\n) -> np.ndarray:\n    """Execute full GPU-accelerated B132C',
     '    score_fn,\n    log,\n    K: dict | None = None,\n) -> np.ndarray:\n    """Execute full GPU-accelerated B132C', 1),
    ('    t0 = time.monotonic()\n    u_pe = unit_features_fp64(pe_features)',
     '    K = {**DEFAULT_K, **(K or {})}\n    t0 = time.monotonic()\n    u_pe = unit_features_fp64(pe_features)', 1),
    ('    u_hp = unit_features_fp64(hp_features)\n\n    graphs, hubs = build_pent_graphs_and_hubs(u_pe, u_eva, u_hp, gamma_q=2.6, chunk_size=512)',
     '    u_hp = unit_features_fp64(hp_features)\n    if K["dba_n"] > 0:\n        u_pe, u_eva, u_hp = _dba(u_pe, K), _dba(u_eva, K), _dba(u_hp, K)\n\n'
     '    graphs, hubs = build_pent_graphs_and_hubs(u_pe, u_eva, u_hp, gamma_q=2.6, chunk_size=512, ks=K["ks"])\n'
     '    if K["extra_graphs"]:\n        graphs.update(_extra_graphs(u_pe, u_eva, u_hp, K["extra_graphs"]))', 1),
    ('    branch_specs = [\n        ("pe", 0.275, 1.0, 0.0, 0.0),\n        ("eva", 0.230, 0.0, 1.0, 0.0),\n        ("j25_k28", 0.235, 0.75, 0.25, 0.0),\n        ("j50_k28", 0.190, 0.50, 0.50, 0.0),\n        ("tri25_k28", 0.070, 0.5625, 0.1875, 0.25),\n    ]',
     '    branch_specs = K["branch_specs"]', 1),
    ('torch.clamp(0.40 * dot_pe + 0.40 * dot_eva + 0.20 * dot_hp, 0.0, 1.0)',
     'torch.clamp(K["cc"][0] * dot_pe + K["cc"][1] * dot_eva + K["cc"][2] * dot_hp, 0.0, 1.0)', 1),
    ('cos_sub = 0.65 * cos_b + 0.35 * cos_c', 'cos_sub = K["cos_mix"][0] * cos_b + K["cos_mix"][1] * cos_c', 1),
    ('(j_sub ** 0.80)', '(j_sub ** K["jpow"])', 1),
    ('            sp_t = torch.zeros((rows, width), dtype=torch.float64, device="cuda")\n',
     '            sp_t = torch.zeros((rows, width), dtype=torch.float64, device="cuda")\n'
     '            cosv_t = torch.zeros_like(sp_t) if K["nd"] else None\n', 1),
    ('                jac_t[s:e] = j_sub\n',
     '                jac_t[s:e] = j_sub\n'
     '                if K["nd"]:\n'
     '                    cosv_t[s:e] = torch.where(vt_sub, cos_sub, 0.0)\n', 1),
    ('(dp >= 0.172)', '(dp >= K["ham_tau"])', 1),
    ('(dp ** 1.83)', '(dp ** K["margin"])', 1),
    ('valid_t & (ham <= 5)', 'valid_t & (ham <= K["max_ham"])', 1),
    ('torch.where(ham0_t <= 5, ew2_t, 0.0)', 'torch.where(ham0_t <= K["max_ham"], ew2_t, 0.0)', 1),
    ('denom = 0.65 * c_denom + 0.35 * torch.clamp(w_raw_sum + 5.0, min=1.0)',
     'denom = K["den"][0] * c_denom + K["den"][1] * torch.clamp(w_raw_sum + K["den"][2], min=1.0)', 1),
    ('torch.tensor(0.95, dtype=torch.float64, device="cuda"),\n                    0.95 / torch.clamp(w_sum, min=1e-12),',
     'torch.tensor(K["step"], dtype=torch.float64, device="cuda"),\n                    K["step"] / torch.clamp(w_sum, min=1e-12),', 1),
    ('conf = torch.exp(0.25 * (1.0 - 4.0 * p_cur * (1.0 - p_cur))) * torch.exp(-0.40 * ms_g)',
     'conf = torch.exp(K["conf_ent"] * (1.0 - 4.0 * p_cur * (1.0 - p_cur))) * torch.exp(-K["conf_ms"] * ms_g)', 1),
    ('p_mid_t = _gpu_diff(p0_g, ew0_t, 3)', 'p_mid_t = _gpu_diff(p0_g, ew0_t, K["s1"])', 1),
    ('pd_t = _gpu_diff(p0_g, ew_t, 6)',
     'pd_t = _gpu_diff(p0_g, ew_t, K["s2"])\n'
     '            if K["adapt"]:  # density-adaptive extra diffusion: dense attribute-consistent clusters diffuse longer\n'
     '                ad = K["adapt"]\n'
     '                pd_hi = _gpu_diff(p0_g, ew_t, ad["hi"])\n'
     '                dens = (ew_t > 0).sum(dim=1, keepdim=True).to(torch.float64) if ad["mode"].startswith("cnt") else ew_t.sum(dim=1, keepdim=True)\n'
     '                if ad["mode"].endswith("rel"):  # relative to this gallery\'s mean density (gallery-only statistic)\n'
     '                    dens = dens / torch.clamp(dens.mean(), min=1e-12)\n'
     '                g_ad = torch.clamp((dens - ad["a"]) / ad["b"], 0.0, 1.0).to(torch.float32)\n'
     '                if ad["mode"] == "const":\n'
     '                    g_ad = torch.full_like(g_ad, float(ad["w"]))\n'
     '                pd_t = pd_t + g_ad * (pd_hi - pd_t)\n'
     '            if K["nd"]:  # extra diffusion restricted to near-duplicate edges (top cosine quantile of this gallery)\n'
     '                nd = K["nd"]\n'
     '                tau_nd = torch.quantile(cosv_t[valid_t].float(), float(nd["q"])).to(torch.float64)\n'
     '                g_e = torch.where(valid_t, torch.sigmoid((cosv_t - tau_nd) / float(nd["s"])), 0.0)\n'
     '                pd_t = _gpu_diff(pd_t.to(torch.float64), ew_t * g_e, int(nd["steps"]))\n'
     '            pd_t = pd_t', 1),  # the deployed line's trailing .to(torch.float64) attaches here
    ('gate_t = torch.clamp((jac_t - 0.24) / 0.30, 0.0, 1.0)', 'gate_t = torch.clamp((jac_t - K["jgate"][0]) / K["jgate"][1], 0.0, 1.0)', 1),
    ('ew_t = 0.45 * ew0_t + 0.55 * ((1.0 - gate_t) * ew2_norm_t + gate_t * ew2_t)',
     'ew_t = K["coev"][0] * ew0_t + K["coev"][1] * ((1.0 - gate_t) * ew2_norm_t + gate_t * ew2_t)', 1),
    ('torch.log(q_t / (1.0 - q_t)) / 1.05)', 'torch.log(q_t / (1.0 - q_t)) / K["post_tau"])', 1),
    ('fac_d_t = (np.float32(0.205) / c_clamp_f32)', 'fac_d_t = (np.float32(K["fac_d"]) / c_clamp_f32)', 1),
    ('fac_dl_t = (np.float32(0.235) / c_clamp_f32)', 'fac_dl_t = (np.float32(K["fac_dl"]) / c_clamp_f32)', 1),
    ('fac_s_t = (np.float32(0.34) / c_clamp_f32)', 'fac_s_t = (np.float32(K["fac_s"]) / c_clamp_f32)', 1),
    ('fac_s2_t = (np.float32(0.04) / c_clamp_f32)', 'fac_s2_t = (np.float32(K["fac_s2"]) / c_clamp_f32)', 1),
    ('fac_ex_t = (np.float32(0.07) / c_clamp_f32)', 'fac_ex_t = (np.float32(K["fac_ex"]) / c_clamp_f32)', 1),
    ('exact_ens = (np.float32(0.78) * exact_ens + np.float32(0.22) * ex_b_sum)',
     'exact_ens = (np.float32(K["ex_ens"][0]) * exact_ens + np.float32(K["ex_ens"][1]) * ex_b_sum)', 1),
    ('top_m = min(11, num_g)', 'top_m = min(K["top_m"], num_g)', 1),
    ('pre_m = min(64, num_g)', 'pre_m = min(K["pre_m"], num_g)', 1),
    ('hk = min(12, num_g)', 'hk = min(K["hk"], num_g)', 1),
    ('bk = min(max(10, int(round(10.0 * (num_g / 10000.0)))), num_g)',
     'bk = min(max(K["bk"][0], int(round(K["bk"][1] * (num_g / 10000.0)))), num_g)', 1),
    ('sv0 = ((hv0.mean() ** 0.35) / (hv0 ** 0.35 + 1e-6))', 'sv0 = ((hv0.mean() ** K["hub_norm"]) / (hv0 ** K["hub_norm"] + 1e-6))', 1),
    ('kap = np.float32(0.20)', 'kap = np.float32(K["kap"])', 1),
    ('ex_kap = np.float32(0.24)', 'ex_kap = np.float32(K["ex_kap"])', 1),
    ('(np.float32(0.55) * d_b[q0:q1] + np.float32(0.45) * d_ens_q)',
     '(np.float32(K["dmix"][0]) * d_b[q0:q1] + np.float32(K["dmix"][1]) * d_ens_q)', 1),
    ('(np.float32(0.55) * d_b_l1[q0:q1] + np.float32(0.45) * d_l1_ens_q)',
     '(np.float32(K["dmix"][0]) * d_b_l1[q0:q1] + np.float32(K["dmix"][1]) * d_l1_ens_q)', 1),
    ('                for _ in range(2):\n                    nxt = torch.empty_like(cur)',
     '                for _ in range(K["d_steps"]):\n                    nxt = torch.empty_like(cur)', 1),
    ('                for _ in range(2):\n                    nxt_l = torch.empty_like(cur_l)',
     '                for _ in range(K["l_steps"]):\n                    nxt_l = torch.empty_like(cur_l)', 1),
    ('ex_h_q = 0.50 * ex_q0 + 0.50 * ex_diff_gpu', 'ex_h_q = K["exh"][0] * ex_q0 + K["exh"][1] * ex_diff_gpu', 1),
    ('z_base = (1.0 - 0.72) * dq_z + 0.72 * ex_z', 'z_base = (1.0 - K["ex_w"]) * dq_z + K["ex_w"] * ex_z', 1),
    ('z_w = z_base + 0.04 * (delta_dq * delta_ex_s)', 'z_w = z_base + K["cross"] * (delta_dq * delta_ex_s)', 1),
    ('z_comb = z_w + 0.030 * torch.abs(dq_z - ex_z)', 'z_comb = z_w + K["disagree"] * torch.abs(dq_z - ex_z)', 1),
    ('w_m = torch.softmax(-top_d / 0.052, dim=1)', 'w_m = torch.softmax(-top_d / K["a_temp"], dim=1)', 1),
    ('ms_k1 = min(6, top_m)', 'ms_k1 = min(K["ms_k1"], top_m)', 1),
    ('torch.softmax(-top_d[:, :ms_k1] / 0.035, dim=1)', 'torch.softmax(-top_d[:, :ms_k1] / K["ms_temp"], dim=1)', 1),
    ('w_m = 0.78 * w_m + 0.22 * w_m1', 'w_m = (1.0 - K["ms_blend"]) * w_m + K["ms_blend"] * w_m1', 0),
    ('mm_joint = torch.clamp(0.45 * mm_pe + 0.35 * mm_eva + 0.20 * mm_hp, min=0.0)',
     'mm_joint = torch.clamp(K["mm"][0] * mm_pe + K["mm"][1] * mm_eva + K["mm"][2] * mm_hp, min=0.0)', 1),
    ('w_m = w_m * torch.exp(0.25 * coh)', 'w_m = w_m * torch.exp(K["coh"] * coh)', 1),
    ('for sk, sw in ANCHOR_SPEC_WEIGHTS.items():', 'for sk, sw in K["anchor_w"].items():', 1),
    ('torch.clamp(c_s, min=0.0) ** 2.6)', 'torch.clamp(c_s, min=0.0) ** K["a_gamma"])', 1),
    ('d_out_q = d_comb_gpu - 0.126 * d_std * sim_tot', 'd_out_q = d_comb_gpu - K["a_alpha"] * d_std * sim_tot', 1),
    ('base_ex = (1.0 - 0.15) * ex_h_q + 0.15 * (ex_h_std', 'base_ex = (1.0 - K["dq_blend"]) * ex_h_q + K["dq_blend"] * (ex_h_std', 1),
    ('ex_q_eff = base_ex - 0.120 * ex_h_std * sim_tot', 'ex_q_eff = base_ex - K["ex_vis"] * ex_h_std * sim_tot', 1),
    ('loc_norm = loc_spread / 0.35', 'loc_norm = loc_spread / K["loc"]', 1),
    ('eff_scale = (1.0 - 0.12) * ex_eff_std + 0.12 * loc_norm', 'eff_scale = (1.0 - K["loc_lam"]) * ex_eff_std + K["loc_lam"] * loc_norm', 1),
    ('head_exp = torch.exp(-delta_ex_h / 0.225)', 'head_exp = torch.exp(-delta_ex_h / K["head_tau"])', 1),
    ('d_out_q = d_out_q - 0.044 * d_std * head_exp', 'd_out_q = d_out_q - K["head_beta"] * d_std * head_exp', 1),
    ('hb_exp = torch.exp(-delta_ex_h / 0.55)', 'hb_exp = torch.exp(-delta_ex_h / K["body_head_tau"])', 1),
    ('d_l1_out_q = d_l1_gpu - (1.35 * 0.126) * dl_std * sim_tot - (3.5 * 0.044) * dl_std * hb_exp',
     'd_l1_out_q = d_l1_gpu - (K["body_vis"] * K["a_alpha"]) * dl_std * sim_tot - (K["body_keep"] * K["head_beta"]) * dl_std * hb_exp', 1),
    ('gate = 0.33 * (1.0 - torch.exp(-uc / 0.355))', 'gate = K["body_alpha"] * (1.0 - torch.exp(-uc / K["body_tau"]))', 1),
    ('            output[q0:q1] = d_out_q.cpu().numpy()',
     '            if K["aprf"]:  # attribute-space pseudo-relevance feedback / soft-query expansion\n'
     '              for ap in (K["aprf"] if isinstance(K["aprf"], list) else [K["aprf"]]):\n'
     '                m_ap = min(int(ap["m"]), num_g)\n'
     '                for _round in range(int(ap.get("rounds", 1))):\n'
     '                    top_ap = torch.topk(d_out_q, k=m_ap, dim=1, largest=False, sorted=True).indices\n'
     '                    if ap.get("tau"):  # rank-decayed prototype weights\n'
     '                        wr = torch.exp(-torch.arange(m_ap, device="cuda", dtype=torch.float32) / float(ap["tau"]))\n'
     '                        proto = (p_ens_g32[top_ap] * wr[None, :, None]).sum(dim=1) / wr.sum()\n'
     '                    else:\n'
     '                        proto = p_ens_g32[top_ap].mean(dim=1)\n'
     '                    if ap.get("mode") == "nll":\n'
     '                        d_ap = torch.from_numpy(score_fn(p_ens, proto.cpu().numpy().astype(np.float32), attribute_names, **K["sc_ex"])).to("cuda")\n'
     '                    else:\n'
     '                        d_ap = proto @ (1.0 - p_ens_g32).T + (1.0 - proto) @ p_ens_g32.T\n'
     '                    if ap.get("smooth"):  # one graph-diffusion step of the prototype distance over the gallery branches\n'
     '                        d_sm = torch.zeros_like(d_ap)\n'
     '                        for bw_s, safe_s, w_s, *_rest in gpu_branches:\n'
     '                            c_s = torch.clamp((w_s > 0).sum(dim=1), min=1).to(torch.float32)[None, :]\n'
     '                            nb = (d_ap[:, safe_s] - d_ap[:, :, None]) * w_s[None, :, :]\n'
     '                            d_sm = d_sm + bw_s * (d_ap + float(ap["smooth"]) / c_s * nb.sum(dim=2))\n'
     '                        d_ap = d_sm\n'
     '                    z_ap = (d_ap - d_ap.mean(dim=1, keepdim=True)) / torch.clamp(d_ap.std(dim=1, keepdim=True), min=1e-6)\n'
     '                    beta_q = float(ap["beta"])\n'
     '                    if ap.get("bgate"):  # per-query strength from the expected number of exact matches (own probabilities)\n'
     '                        bg = ap["bgate"]\n'
     '                        qa = torch.from_numpy(np.asarray(queries[q0:q1], dtype=np.float32)).to("cuda")\n'
     '                        lp_a = torch.log(torch.clamp(p_ens_g32, 1e-5, 1 - 1e-5)); ln_a = torch.log1p(-torch.clamp(p_ens_g32, 1e-5, 1 - 1e-5))\n'
     '                        le_q = torch.logsumexp(qa @ lp_a.T + (1.0 - qa) @ ln_a.T, dim=1, keepdim=True)\n'
     '                        g_q = torch.clamp((le_q - float(np.log(bg["a"]))) / float(bg["b"]), 0.0, 1.0)\n'
     '                        beta_q = float(bg["lo"]) + (float(bg["hi"]) - float(bg["lo"])) * g_q\n'
     '                    d_new = d_out_q + beta_q * torch.clamp(d_out_q.std(dim=1, keepdim=True), min=1e-6) * z_ap\n'
     '                    if ap.get("protect"):  # keep the pre-expansion top-k0 on top, in their original order\n'
     '                        k0 = min(int(ap["protect"]), num_g)\n'
     '                        top0 = torch.topk(d_out_q, k=k0, dim=1, largest=False, sorted=True).indices\n'
     '                        floor = d_new.min(dim=1, keepdim=True).values\n'
     '                        offs = (torch.arange(k0, device="cuda", dtype=torch.float32)[None, :] - k0) * 1e-3\n'
     '                        d_new = d_new.scatter(1, top0, floor + offs * torch.clamp(d_out_q.std(dim=1, keepdim=True), min=1e-6))\n'
     '                    d_out_q = d_new\n'
     '            output[q0:q1] = d_out_q.cpu().numpy()', 1),
    ('        kap = np.float32(K["kap"])', '        kap = np.float32(K["kap"])\n        p_ens_g32 = torch.from_numpy(np.asarray(p_ens, dtype=np.float32)).to("cuda") if K["aprf"] else None', 1),
]


def main() -> None:
    out = text
    for old, new, count in REPL:
        c = out.count(old)
        if count == 0:  # w_m blend: deployed writes the literals; keep the literal form when defaults are used
            assert c == 1, (old, c)
            new = 'w_m = (K["ms_mix"][0] * w_m + K["ms_mix"][1] * w_m1)'
            out = out.replace(old, new)
            continue
        assert c == count, (old[:80], c)
        out = out.replace(old, new)
    # the score triple: route the three score_fn kwarg sets through K (same python floats, same call order)
    start = out.index("            try:\n                d_base = score_fn(")
    end = out.index("            return d_base, d_ex, d_l1")
    out = out[:start] + (
        '            d_base = score_fn(p_mat, queries, attribute_names, **K["sc_base"], _pre=pre)\n'
        '            d_ex = score_fn(p_mat, queries, attribute_names, **K["sc_ex"], _pre=pre)\n'
        '            d_l1 = score_fn(p_mat, queries, attribute_names, **K["sc_l1"], _pre=pre)\n'
    ) + out[end:]
    out += '''

DEFAULT_K = {
    "ks": {"pe": 20, "eva": 20, "j25_k28": 28, "j50_k28": 28, "tri25_k28": 28},
    "branch_specs": [
        ("pe", 0.275, 1.0, 0.0, 0.0),
        ("eva", 0.230, 0.0, 1.0, 0.0),
        ("j25_k28", 0.235, 0.75, 0.25, 0.0),
        ("j50_k28", 0.190, 0.50, 0.50, 0.0),
        ("tri25_k28", 0.070, 0.5625, 0.1875, 0.25),
    ],
    "cc": (0.40, 0.40, 0.20), "cos_mix": (0.65, 0.35), "jpow": 0.80, "ham_tau": 0.172, "margin": 1.83, "max_ham": 5,
    "den": (0.65, 0.35, 5.0), "step": 0.95, "conf_ent": 0.25, "conf_ms": 0.40, "s1": 3, "s2": 6,
    "jgate": (0.24, 0.30), "coev": (0.45, 0.55), "post_tau": 1.05,
    "fac_d": 0.205, "fac_dl": 0.235, "fac_s": 0.34, "fac_s2": 0.04, "fac_ex": 0.07,
    "sc_base": dict(lam_override=0.275, lam_neg_override=0.215, err_gamma=1.0, err_gamma_neg=0.97, dich_sym_loss=1.28,
                    dich_pos_nll_add=0.08),
    "sc_ex": dict(ent_override=0.45, lam_override=0.43, lam_neg_override=0.30, err_gamma=1.25, err_gamma_neg=1.25,
                  dich_sym_loss=1.25),
    "sc_l1": dict(ent_override=0.15, lam_override=0.05, lam_neg_override=0.0, err_gamma=0.90, err_gamma_neg=0.75,
                  uniform_weights=True, dich_sym_loss=0.0, pos_q_mul=0.70),
    "ex_ens": (0.78, 0.22), "top_m": 11, "pre_m": 64, "hk": 12, "bk": (10, 10.0), "hub_norm": 0.35,
    "kap": 0.20, "ex_kap": 0.24, "dmix": (0.55, 0.45), "d_steps": 2, "l_steps": 2, "exh": (0.50, 0.50),
    "ex_w": 0.72, "cross": 0.04, "disagree": 0.030, "a_temp": 0.052, "ms_k1": 6, "ms_temp": 0.035,
    "ms_mix": (0.78, 0.22), "mm": (0.45, 0.35, 0.20), "coh": 0.25, "anchor_w": dict(ANCHOR_SPEC_WEIGHTS),
    "a_gamma": 2.6, "a_alpha": 0.126, "dq_blend": 0.15, "ex_vis": 0.120, "loc": 0.35, "loc_lam": 0.12,
    "head_tau": 0.225, "head_beta": 0.044, "body_head_tau": 0.55, "body_vis": 1.35, "body_keep": 3.5,
    "body_alpha": 0.33, "body_tau": 0.355,
    "dba_n": 0, "dba_pow": 3.0,
    "extra_graphs": [],
    "adapt": None,
    "nd": None,
    "aprf": None,
}


def _extra_graphs(u_pe: np.ndarray, u_eva: np.ndarray, u_hp: np.ndarray, specs: list) -> dict:
    """Extra mutual-k graphs (name, k, w_pe, w_eva, w_hp) in the same packed (neighbours, count) format; plain topk
    (ties broken by torch), used only by knob screens - the deployed five graphs keep their exact builder."""
    import torch
    from b121_fp32_cosine_graph import strict_fp32
    n = len(u_pe)
    out = {}
    with strict_fp32(), torch.inference_mode():
        x_pe = torch.from_numpy(u_pe.astype(np.float32)).to("cuda")
        x_eva = torch.from_numpy(u_eva.astype(np.float32)).to("cuda")
        x_hp = torch.from_numpy(u_hp.astype(np.float32)).to("cuda")
        for name, k, a, b, c in specs:
            d_t = torch.empty((n, k), dtype=torch.int64, device="cuda")
            for s in range(0, n, 512):
                e = min(n, s + 512)
                mat = np.float32(a) * (x_pe[s:e] @ x_pe.T) + np.float32(b) * (x_eva[s:e] @ x_eva.T)
                if c > 0.0:
                    mat = mat + np.float32(c) * (x_hp[s:e] @ x_hp.T)
                mat[torch.arange(e - s, device="cuda"), torch.arange(s, e, device="cuda")] = -float("inf")
                d_t[s:e] = torch.topk(mat, k=k, dim=1).indices
            row = torch.arange(n, device="cuda")[:, None, None]
            mut = (d_t[d_t] == row).any(dim=2)
            cnt = mut.sum(dim=1)
            col = torch.arange(k, device="cuda")[None, :]
            key = torch.where(mut, col, torch.full_like(col, k))
            packed = torch.gather(d_t, 1, torch.argsort(key, dim=1, stable=True))
            out[name] = (torch.where(col < cnt[:, None], packed, torch.full_like(packed, -1)).cpu().numpy(), cnt.cpu().numpy())
    return out


def _dba(u: np.ndarray, K: dict) -> np.ndarray:
    """Database-side augmentation (graph features only): u_i <- normalise(u_i + sum_{j in topn(i)} max(cos,0)^pow u_j)."""
    import torch
    n_nb = int(K["dba_n"])
    x = torch.from_numpy(u.astype(np.float32)).to("cuda")
    out = np.empty_like(u)
    for s in range(0, len(u), 1024):
        e = min(len(u), s + 1024)
        sim = x[s:e] @ x.T
        sim[torch.arange(e - s, device="cuda"), torch.arange(s, e, device="cuda")] = -2.0
        vals, idx = torch.topk(sim, k=n_nb, dim=1)
        w = torch.clamp(vals, min=0.0) ** float(K["dba_pow"])
        agg = x[s:e].double() + (w[:, :, None].double() * x[idx].double()).sum(dim=1)
        out[s:e] = agg.cpu().numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)
'''
    import sys
    dest = HERE / (sys.argv[1] if len(sys.argv) > 1 else "rerank_param.py")
    assert dest.name != "rerank_param.py" or not (HERE / "rerank_param.frozen_for_b140c.py").exists(), "rerank_param.py is frozen (B140C verifier pins it)"
    dest.write_text(out)
    print("written", len(out))


if __name__ == "__main__":
    main()
