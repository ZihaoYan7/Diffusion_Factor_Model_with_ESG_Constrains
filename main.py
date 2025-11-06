# -*- coding: utf-8 -*-
"""
Diffusion Factor Model + ESG (5Y rolling; M/Q rebalance)

仅新增：业绩归因分解（堆叠条形图 + CSV）
- 严格不改变原有训练/采样/回测/作图/CSV 的任何逻辑与输出
- 归因实现遵循 Lo & Zhang (2024/2025) 的三分解（基准 / 静态成本 / 信息项）

新增输出：
    <out>/attribution_bars.csv
    <out>/attribution_bars_<method-slug>.png

原有说明略（保持不变）
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MOSEKLM_LICENSE_FILE", "/root/autodl-tmp/mosek.lic")  # 若未安装可忽略

import gc, glob, argparse, warnings, re
from typing import List, Tuple, Dict, Optional, Iterable
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# 你的项目模块（保持不变）
import config.config as cfg
from diffusion_factor_model import Unet, GaussianDiffusion, Trainer


# ============================ Utils（原有，保持不变） ============================

def set_seed(seed: int):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)

def month_ends(idx: pd.DatetimeIndex) -> List[pd.Timestamp]:
    return list(pd.Series(1.0, index=idx).resample('M').last().index)

def quarter_ends(idx: pd.DatetimeIndex) -> List[pd.Timestamp]:
    return list(pd.Series(1.0, index=idx).resample('Q').last().index)

def choose_hw_and_n(target_n: int) -> Tuple[int, int, int]:
    for n in (4096, 2048, 1024):
        if target_n >= n:
            if n == 4096: return 64, 64, 4096
            if n == 2048: return 32, 64, 2048
            return 32, 32, 1024
    return 32, 32, 1024

def _slug_int(x) -> str:
    return re.sub(r'[^0-9]', '', str(x))


# ============================ 日频安全聚合（保持不变） ============================

def dedupe_daily(df: pd.DataFrame, key_id: str) -> pd.DataFrame:
    tmp = df.rename(columns={c: c.upper() for c in df.columns}).copy()
    has_ret = 'RET' in tmp.columns
    if 'DATE' not in tmp.columns:
        if 'date' in df.columns:
            tmp['DATE'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("input df must have 'date'")
    else:
        tmp['DATE'] = pd.to_datetime(tmp['DATE'])
    if 'PRICE' in tmp.columns and 'PRC' not in tmp.columns:
        tmp.rename(columns={'PRICE': 'PRC'}, inplace=True)
    if 'VOLUME' in tmp.columns and 'VOL' not in tmp.columns:
        tmp.rename(columns={'VOLUME': 'VOL'}, inplace=True)
    agg_map = {}
    if has_ret: agg_map['RET'] = 'mean'
    if 'PRC' in tmp.columns: agg_map['PRC'] = 'last'
    if 'VOL' in tmp.columns: agg_map['VOL'] = 'sum'
    use_cols = ['DATE', key_id.upper()] + list(agg_map.keys())
    tmp = tmp[use_cols].copy()
    if has_ret: tmp['RET'] = pd.to_numeric(tmp['RET'], errors='coerce')
    g = tmp.groupby(['DATE', key_id.upper()], as_index=False).agg(agg_map)
    if 'RET' not in g.columns: g['RET'] = np.nan
    g = g[['DATE', key_id.upper(), 'RET']].rename(columns={'DATE': 'date', key_id.upper(): key_id, 'RET': 'RET'})
    return g

def load_returns_permno_panel(returns_dir: str, start: str, end: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    fps = sorted(glob.glob(os.path.join(returns_dir, "ret_*.csv")))
    if not fps: raise FileNotFoundError(f"No ret_*.csv at {returns_dir}")
    frames = []
    for fp in fps:
        df = pd.read_csv(fp, low_memory=False, dtype={'PERMNO': 'Int64', 'CUSIP': str})
        need = {'PERMNO', 'date', 'CUSIP', 'RET'}
        if not need.issubset(df.columns): raise ValueError(f"{fp} must have {need}")
        df['date'] = pd.to_datetime(df['date']); df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
        frames.append(df[['PERMNO', 'date', 'CUSIP', 'RET']])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[(all_df['date'] >= pd.to_datetime(start)) & (all_df['date'] <= pd.to_datetime(end))]
    dedup = dedupe_daily(all_df.rename(columns={'date':'DATE'}), key_id='PERMNO')
    panel = dedup.pivot(index='date', columns='PERMNO', values='RET').sort_index()
    permno2cusip: Dict[int, str] = {}
    for pid, g in all_df.dropna(subset=['CUSIP']).groupby('PERMNO'):
        vals, cnt = np.unique(g['CUSIP'].astype(str).str.strip().values, return_counts=True)
        permno2cusip[int(pid)] = str(vals[int(np.argmax(cnt))])
    return panel, permno2cusip

def load_returns_cusip_panel(returns_dir: str, start: str, end: str,
                             blocklist: Optional[Iterable[str]] = None) -> pd.DataFrame:
    fps = sorted(glob.glob(os.path.join(returns_dir, "ret_*.csv")))
    if not fps: raise FileNotFoundError(f"No ret_*.csv at {returns_dir}")
    frames = []
    for fp in fps:
        df = pd.read_csv(fp, low_memory=False, dtype={'CUSIP': str})
        cols = {c.lower(): c for c in df.columns}
        need = ['date', 'cusip', 'ret']
        if not all(k in cols for k in need): raise ValueError(f"{fp} must have PERMNO,date,CUSIP,RET")
        df.rename(columns={cols['date']: 'date', cols['cusip']: 'CUSIP', cols['ret']: 'RET'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']); df['RET']  = pd.to_numeric(df['RET'], errors='coerce')
        frames.append(df[['date', 'CUSIP', 'RET']])
    all_ = pd.concat(frames, ignore_index=True)
    all_ = all_[(all_['date'] >= pd.to_datetime(start)) & (all_['date'] <= pd.to_datetime(end))]
    if blocklist:
        block = set([str(x).strip() for x in blocklist if str(x).strip()])
        if block: all_ = all_[~all_['CUSIP'].isin(block)]
    dedup = dedupe_daily(all_.rename(columns={'date':'DATE'}), key_id='CUSIP')
    panel = dedup.pivot(index='date', columns='CUSIP', values='RET').sort_index()
    return panel

def load_esg_scores(esg_dir: str, start: str, end: str, field='esg_13') -> pd.DataFrame:
    fps = sorted(glob.glob(os.path.join(esg_dir, "ESGscore_*.csv")))
    if not fps: raise FileNotFoundError(f"No ESGscore_*.csv at {esg_dir}")
    frames = []
    for fp in fps:
        df = pd.read_csv(fp, dtype={'CUSIP': str})
        if 'CUSIP' not in df.columns or field not in df.columns:
            raise ValueError(f"{fp} must have CUSIP and {field}")
        df['CUSIP'] = df['CUSIP'].astype(str).str.strip()
        df['ESG'] = pd.to_numeric(df[field], errors='coerce')
        base = os.path.basename(fp); digits = ''.join([c for c in base if c.isdigit()])
        year = int(digits[:4]) if len(digits) >= 4 else None
        if year is None: raise ValueError(f"Cannot infer year from filename {base}")
        df['date'] = pd.to_datetime(f"{year}-12-31")
        frames.append(df[['date', 'CUSIP', 'ESG']])
    yearly = pd.concat(frames, ignore_index=True).dropna(subset=['CUSIP'])
    yearly = yearly.groupby(['date', 'CUSIP'], as_index=False)['ESG'].mean()
    pivot = yearly.pivot_table(index='date', columns='CUSIP', values='ESG', aggfunc='mean').sort_index()
    monthly = pd.date_range(start=start, end=end, freq='M')
    esg = pivot.reindex(monthly).ffill().bfill(); esg.index.name = None
    return esg


# ============================ Winsorize / 标准化（保持不变） ============================

def my_winsorize(data: np.ndarray) -> np.ndarray:
    X = np.array(data, dtype=float, copy=True)
    T, N = X.shape
    for j in range(N):
        col = X[:, j]
        mask_valid = np.isfinite(col)
        if mask_valid.sum() < 50: continue
        c = col[mask_valid]
        mean = np.nanmean(c)
        lo = max(np.nanpercentile(c, 2.5), -0.2)
        hi = min(np.nanpercentile(c, 97.5), 0.2)
        m_low = (col < lo) & mask_valid
        if m_low.any():
            pool = col[(~m_low) & mask_valid & (col < mean)]
            if pool.size == 0: pool = col[(~m_low) & mask_valid]
            if pool.size == 0:
                col[m_low] = lo
            else:
                uniq, cnt = np.unique(pool, return_counts=True)
                prob = cnt / cnt.sum(); col[m_low] = np.random.choice(uniq, size=m_low.sum(), p=prob)
        m_high = (col > hi) & mask_valid
        if m_high.any():
            pool = col[(~m_high) & mask_valid & (col > mean)]
            if pool.size == 0: pool = col[(~m_high) & mask_valid]
            if pool.size == 0:
                col[m_high] = hi
            else:
                uniq, cnt = np.unique(pool, return_counts=True)
                prob = cnt / cnt.sum(); col[m_high] = np.random.choice(uniq, size=m_high.sum(), p=prob)
        X[:, j] = col
    return X

def my_winsorize_light(data: np.ndarray, pct: float = 0.5, cap: float = 0.2) -> np.ndarray:
    X = np.array(data, dtype=float, copy=True)
    for i in range(X.shape[1]):
        m = X[:, i].mean()
        lo = max(np.percentile(X[:, i], pct), -cap)
        hi = min(np.percentile(X[:, i], 100 - pct), cap)
        mask = X[:, i] < lo
        pool = X[(~mask) & (X[:, i] < m), i]
        if pool.size > 0:
            u, c = np.unique(pool, return_counts=True)
            X[mask, i] = np.random.choice(u, size=mask.sum(), p=c / c.sum())
        else:
            X[mask, i] = lo
        mask = X[:, i] > hi
        pool = X[(~mask) & (X[:, i] > m), i]
        if pool.size > 0:
            u, c = np.unique(pool, return_counts=True)
            X[mask, i] = np.random.choice(u, size=mask.sum(), p=c / c.sum())
        else:
            X[mask, i] = hi
    return X

def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = x.mean(axis=0); s = x.std(axis=0); s[s < 1e-8] = 1.0
    return (x - m) / s, m, s

def unstandardize(x: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return x * s + m


# ============================ Real-only 作图筛选（保持不变） ============================

def smooth_unimodal_mask(data_wins: np.ndarray, bins: int = 81,
                         plateau_allow: int = 3, spike_ratio_thresh: float = 20.0) -> np.ndarray:
    T, N = data_wins.shape
    keep = np.ones(N, dtype=bool)
    for j in range(N):
        col = data_wins[:, j]; col = col[np.isfinite(col)]
        if col.size < 50: keep[j] = False; continue
        span = min(0.2, max(abs(col.min()), abs(col.max())))
        if not np.isfinite(span) or span < 1e-6: keep[j] = False; continue
        b = bins if bins % 2 == 1 else bins + 1
        edges = np.linspace(-span, span, b + 1)
        counts, _ = np.histogram(col, bins=edges)
        if counts.sum() == 0: keep[j] = False; continue
        smooth = np.convolve(counts.astype(float), [1,4,6,4,1], mode='same')
        mx = smooth.max()
        if mx <= 0: keep[j] = False; continue
        near = smooth >= 0.99 * mx
        plat = best = 0
        for v in near:
            if v: plat += 1; best = max(best, plat)
            else: plat = 0
        if best > plateau_allow: keep[j] = False; continue
        if mx / (smooth.mean() + 1e-8) > spike_ratio_thresh: keep[j] = False; continue
    return keep

def is_abnormal_spiky(col: np.ndarray, bins: int = 301, center_win: int = 2,
                      spike_ratio_thresh: float = 15.0, neighbor_frac_thresh: float = 0.18,
                      tail_mass_frac: float = 0.07, max_local_peaks: int = 3) -> bool:
    col = np.clip(col[np.isfinite(col)], -0.2, 0.2)
    if col.size < 50: return True
    edges = np.linspace(-0.2, 0.2, bins + 1)
    counts, _ = np.histogram(col, bins=edges)
    total = counts.sum()
    if total == 0: return True
    center = bins // 2
    idx_max = int(np.argmax(counts))
    cmax = counts[idx_max]
    mean_c = counts.mean() + 1e-8
    lo = max(0, idx_max - center_win)
    hi = min(bins, idx_max + center_win + 1)
    neighbor = counts[lo:hi].sum() - cmax
    tail = counts[:max(0, center-10)].sum() + counts[min(bins, center+11):].sum()
    cond_spike = (abs(idx_max - center) <= center_win and
                  (cmax / mean_c) > spike_ratio_thresh and
                  neighbor < neighbor_frac_thresh * cmax and
                  tail > tail_mass_frac * total)
    smooth = np.convolve(counts.astype(float), [1,4,6,4,1], mode='same')
    locmax = np.sum((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:]))
    cond_multi = (locmax >= max_local_peaks)
    return bool(cond_spike or cond_multi)

def zero_mass_spike_mask(data_wins: np.ndarray, frac_thresh: float = 0.25,
                         eps_floor: float = 5e-4, eps_mult: float = 0.12) -> np.ndarray:
    T, N = data_wins.shape
    bad = np.zeros(N, dtype=bool)
    for j in range(N):
        col = data_wins[:, j]; col = col[np.isfinite(col)]
        if col.size < 50: bad[j] = True; continue
        med = np.median(col); sd  = np.std(col) + 1e-12
        eps = max(eps_floor, eps_mult * sd)
        frac0 = np.mean(np.abs(col - med) <= eps)
        if frac0 >= frac_thresh: bad[j] = True
    return bad

def center_mass_ratio(col: np.ndarray, bins: int = 301) -> float:
    x = np.clip(col[np.isfinite(col)], -0.2, 0.2)
    if x.size < 50: return 1.0
    edges = np.linspace(-0.2, 0.2, bins + 1)
    cnt, _ = np.histogram(x, bins=edges)
    total = cnt.sum() + 1e-12
    return cnt[bins // 2] / total

def select_columns_with_shape_filter(win_panel: pd.DataFrame, capN: int,
                                     min_stock_cov: float, preN_list: List[int]) -> Tuple[List[str], np.ndarray]:
    stock_cov_series = win_panel.notna().mean(axis=0)
    base_cols = list(stock_cov_series[stock_cov_series >= min_stock_cov].sort_values(ascending=False).index)
    if len(base_cols) == 0:
        base_cols = list(stock_cov_series.sort_values(ascending=False).index)
    tried = set()
    for preN in preN_list:
        preN = min(preN, len(base_cols))
        if preN in tried: continue
        tried.add(preN)
        cols_pre = base_cols[:preN]
        Xw = my_winsorize(win_panel[cols_pre].values)
        uni = smooth_unimodal_mask(Xw, bins=81, plateau_allow=3, spike_ratio_thresh=20.0)
        abn = np.array([is_abnormal_spiky(Xw[:, k]) for k in range(Xw.shape[1])], dtype=bool)
        zms = zero_mass_spike_mask(Xw, frac_thresh=0.25, eps_floor=5e-4, eps_mult=0.12)
        keep = uni & (~abn) & (~zms)
        kept_cols = np.array(cols_pre)[keep]
        if kept_cols.size >= capN:
            stds = np.nanstd(Xw[:, keep], axis=0)
            order = np.argsort(stds)[::-1][:capN]
            chosen = kept_cols[order].tolist()
            print(f"[shape] preN={preN:4d}  keep={keep.sum():4d}  (uni drop={(~uni).sum()}, spiky drop={abn.sum()}, zero-mass drop={zms.sum()})  -> chosen={len(chosen)}")
            return chosen, Xw[:, keep][:, order]
    for preN in [4096, len(base_cols)]:
        preN = min(preN, len(base_cols))
        if preN in tried: continue
        tried.add(preN)
        cols_pre = base_cols[:preN]
        Xw = my_winsorize(win_panel[cols_pre].values)
        uni = smooth_unimodal_mask(Xw, bins=81, plateau_allow=5, spike_ratio_thresh=25.0)
        abn = np.array([is_abnormal_spiky(Xw[:, k], spike_ratio_thresh=15.0,
                                          neighbor_frac_thresh=0.18, tail_mass_frac=0.05)
                        for k in range(Xw.shape[1])], dtype=bool)
        zms = zero_mass_spike_mask(Xw, frac_thresh=0.30, eps_floor=5e-4, eps_mult=0.12)
        keep = uni & (~abn) & (~zms)
        kept_cols = np.array(cols_pre)[keep]
        if kept_cols.size >= capN:
            stds = np.nanstd(Xw[:, keep], axis=0)
            order = np.argsort(stds)[::-1][:capN]
            chosen = kept_cols[order].tolist()
            print(f"[shape-relaxed] preN={preN:4d}  keep={keep.sum():4d}  -> chosen={len(chosen)}")
            return chosen, Xw[:, keep][:, order]
    fallback = base_cols[:capN]
    print(f"[WARN] shape filter insufficient; fallback to coverage-Top {capN}.")
    Xw_fallback = my_winsorize(win_panel[fallback].values)
    return fallback, Xw_fallback


# ============================ Mild shape filter / 训练候选（保持不变） ============================

def is_spiky_zero_center(col: np.ndarray, spike_ratio: float = 25.0,
                         neighbor_frac: float = 0.10, tail_frac: float = 0.06,
                         center_win: int = 2, bins: int = 301) -> bool:
    x = col[np.isfinite(col)]
    if x.size < 50: return True
    edges = np.linspace(-0.2, 0.2, bins + 1)
    cnt, _ = np.histogram(x, bins=edges)
    if cnt.sum() == 0: return True
    idx = int(np.argmax(cnt)); cmax = cnt[idx]; mean_c = cnt.mean() + 1e-8
    center = bins // 2
    lo = max(0, idx - center_win); hi = min(bins, idx + center_win + 1)
    neighbor = cnt[lo:hi].sum() - cmax
    lo_tail = max(0, center - 10); hi_tail = min(bins, center + 11)
    tail = cnt[:lo_tail].sum() + cnt[hi_tail:].sum()
    return (abs(idx - center) <= center_win) and (cmax/mean_c > spike_ratio) and (neighbor < neighbor_frac * cmax) and (tail > tail_frac * cnt.sum())

def mild_shape_filter(win_panel: pd.DataFrame, pool_cols: List[int]) -> List[int]:
    X = my_winsorize(win_panel[pool_cols].values)
    keep = []
    for j, col_id in enumerate(pool_cols):
        col = X[:, j]
        if np.nanstd(col) < 1e-5: continue
        if is_spiky_zero_center(col, spike_ratio=25.0, neighbor_frac=0.10, tail_frac=0.06, center_win=2, bins=301):
            continue
        keep.append(col_id)
    return keep

def select_universe_1024(win_panel: pd.DataFrame, esg_series: pd.Series,
                         min_stock_coverage: float, target_n: int = 1024) -> List[int]:
    with_esg = esg_series.index[~esg_series.isna()].tolist()
    avail_cols = [c for c in win_panel.columns if c in with_esg]
    if len(avail_cols) < target_n: return avail_cols
    cov = win_panel[avail_cols].notna().mean(axis=0).sort_values(ascending=False)
    sorted_cols = list(cov.index)
    pre_2048 = sorted_cols[:min(2048, len(sorted_cols))]
    keep_2048 = mild_shape_filter(win_panel, pre_2048)
    selected = keep_2048
    if len(selected) < target_n:
        remain = [c for c in sorted_cols[:min(1536, len(sorted_cols))] if c not in selected]
        keep_1536 = mild_shape_filter(win_panel, remain)
        selected = selected + [c for c in keep_1536 if c not in selected]
    if len(selected) < target_n:
        for c in sorted_cols:
            if c not in selected:
                selected.append(c)
            if len(selected) >= target_n: break
    return selected[:min(target_n, len(selected))]

def greedy_intersection_days(win_panel: pd.DataFrame, cols_order: List[int],
                             min_T_targets: List[int]) -> Tuple[List[int], np.ndarray]:
    T = win_panel.shape[0]
    base_mask_all = np.ones(T, dtype=bool)
    for T_min in min_T_targets:
        chosen: List[int] = []
        mask_all = base_mask_all.copy()
        for c in cols_order:
            col_mask = win_panel[c].notna().values
            new_mask = mask_all & col_mask
            if new_mask.sum() >= T_min or len(chosen) == 0:
                chosen.append(c); mask_all = new_mask
                if len(chosen) == 1024 and mask_all.sum() >= T_min:
                    return chosen, mask_all
    chosen, mask_all = [], base_mask_all.copy()
    for c in cols_order:
        col_mask = win_panel[c].notna().values
        new_mask = mask_all & col_mask
        if new_mask.sum() >= 1 or len(chosen) == 0:
            chosen.append(c); mask_all = new_mask
        if len(chosen) == 1024 and mask_all.sum() >= 1:
            return chosen, mask_all
    return cols_order[:min(1024, len(cols_order))], mask_all


# ============================ 组合优化（保持不变） ============================

class MosekUnavailable(Exception): pass

def try_mosek_solve(mu: np.ndarray, Sigma: np.ndarray, esg: np.ndarray,
                    eta: float, wmax: float, esg_floor: float) -> np.ndarray:
    try:
        from mosek.fusion import Model, Domain, Expr, ObjectiveSense
    except Exception as e:
        raise MosekUnavailable(str(e))
    n = len(mu)
    GT = np.linalg.cholesky(Sigma + 1e-9 * np.eye(n)).T
    try:
        with Model("MV_ESG") as M:
            w = M.variable("w", n, Domain.inRange(-wmax, wmax))
            s = M.variable("s", 1, Domain.unbounded())
            M.constraint(Expr.sum(w), Domain.equalsTo(1.0))
            M.constraint(Expr.dot(esg, w), Domain.greaterThan(esg_floor))
            M.constraint(Expr.vstack(s, 0.5, Expr.mul(GT, w)), Domain.inRotatedQCone())
            M.objective(ObjectiveSense.Maximize, Expr.sub(Expr.dot(mu, w), Expr.mul(0.5 * eta, s)))
            M.solve()
            return np.array(w.level())
    except Exception as e:
        raise MosekUnavailable(str(e))

def slsqp_solve(mu: np.ndarray, Sigma: np.ndarray, esg: np.ndarray,
                eta: float, wmax: float, esg_floor: float) -> np.ndarray:
    n = len(mu)
    def obj(w):  return -(w @ mu - 0.5 * eta * (w @ Sigma @ w))
    def grad(w): return -(mu - eta * (Sigma @ w))
    cons = [
        {'type':'eq',   'fun':lambda w: np.sum(w) - 1.0,     'jac':lambda w: np.ones_like(w)},
        {'type':'ineq', 'fun':lambda w: w @ esg - esg_floor, 'jac':lambda w: esg}
    ]
    bounds = [(-wmax, wmax)] * n
    x0 = np.ones(n) / n
    res = minimize(obj, x0, jac=grad, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'maxiter':600, 'ftol':1e-9, 'disp':False})
    if not res.success:
        res = minimize(obj, x0, constraints=cons, bounds=bounds, method='SLSQP',
                       options={'maxiter':1000, 'ftol':1e-9, 'disp':False})
    return res.x

def solve_mv_with_esg(mu, Sigma, esg, eta=3.0, wmax=0.05, esg_floor=None) -> np.ndarray:
    if esg_floor is None:
        esg = np.zeros_like(mu); esg_floor = -1e9
    try:
        return try_mosek_solve(mu, Sigma, esg, eta, wmax, esg_floor)
    except MosekUnavailable as e:
        print(f"[WARN] MOSEK failed ({e}); fallback SciPy SLSQP.")
        return slsqp_solve(mu, Sigma, esg, eta, wmax, esg_floor)


# ============================ DFM 训练与采样（保持不变） ============================

def pick_dim_mults(H: int, W: int) -> Tuple[int, ...]:
    m = min(H, W)
    return (1, 2, 4, 8) if m >= 32 else (1, 2, 4) if m >= 16 else (1, 2) if m >= 8 else (1,)

@torch.no_grad()
def sample_selected_timesteps(trainer: Trainer, diffusion: GaussianDiffusion,
                              batch_size: int, H: int, W: int, keep_steps: List[int]) -> Dict[int, np.ndarray]:
    ts = np.array([200 - s for s in keep_steps], dtype=int)
    try:
        out = trainer.model.sample(batch_size=batch_size, save_timesteps=ts)  # (B,len(ts),1,H,W)
        arr = out.detach().cpu().numpy()
        return {keep_steps[k]: arr[:, k, 0].reshape(batch_size, -1) for k in range(len(keep_steps))}
    except Exception:
        all_steps = diffusion.sample(batch_size=batch_size, return_all_timesteps=True)  # (B,201,1,H,W)
        arr = all_steps.cpu().numpy()
        res = {}
        for k, s in enumerate(keep_steps):
            idx = 200 - s
            res[s] = arr[:, idx, 0].reshape(batch_size, -1)
        return res

def train_dfm_and_generate(Z: np.ndarray, H: int, W: int, seed: int,
                           epochs: int, batch_size: int, lr: float, warmup_iters: int,
                           T_max: int, eta_min: float, save_dir: str,
                           synth_total: int, synth_batch: int, keep_steps: List[int]) -> Dict[int, str]:
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.clip(Z, -10.0, 10.0)
    train_tensor = torch.from_numpy(Z.reshape(Z.shape[0], 1, H, W)).float()
    dataset = TensorDataset(train_tensor)
    if len(dataset) == 0: raise RuntimeError("Empty dataset after preprocessing.")
    model = Unet(dim=256, channels=1, filter_size=7, dim_mults=pick_dim_mults(H, W))
    diffusion = GaussianDiffusion(model, image_size=(H, W), latent_dim=H * W,
                                  timesteps=200, objective='pred_noise',
                                  beta_schedule='cosine', auto_normalize=False)
    print("[cfg] dim=256, filter=7x7, dim_mults=", pick_dim_mults(H, W))
    print(f"[train] epochs={epochs}, lr={lr}, warmup={warmup_iters}, cosine_Tmax={T_max}, eta_min={eta_min}")
    trainer = Trainer(diffusion, dataset,
                      train_batch_size=min(batch_size, len(dataset)),
                      train_lr=lr, train_epochs=epochs, adamw_weight_decay=0.01,
                      cosine_scheduler=True, warm_up=True, warmup_iters=warmup_iters,
                      T_max=T_max, eta_min=eta_min, gradient_accumulate_every=1,
                      ema_decay=0.995, split_batches=False, save_and_sample_every=10 ** 9,
                      results_folder=save_dir, param_path="", amp=False)
    trainer.train()
    saved: Dict[int, List[np.ndarray]] = {s: [] for s in keep_steps}
    remain = int(max(4096, synth_total)); bs = int(min(128, max(1, synth_batch)))
    while remain > 0:
        cur = min(bs, remain)
        by = sample_selected_timesteps(trainer, diffusion, cur, H, W, keep_steps)
        for s in keep_steps: saved[s].append(by[s])
        del by; gc.collect()
        try: torch.cuda.empty_cache()
        except: pass
        remain -= cur
    out_paths: Dict[int, str] = {}
    for s in keep_steps:
        mat = np.concatenate(saved[s], axis=0)
        f = os.path.join(save_dir, f"step{s}_samples.npy")
        np.save(f, mat); out_paths[s] = f
        print(f"[save] step{s} -> {f}, shape={mat.shape}")
    try:
        if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
            trainer.accelerator.wait_for_everyone()
            try: trainer.accelerator.free_memory()
            except Exception: pass
    except Exception: pass
    del dataset, train_tensor, model, diffusion, trainer
    gc.collect()
    try: torch.cuda.empty_cache()
    except: pass
    return out_paths


# ============================ 8‑panel 直方图（仅改颜色） ============================

def _extreme_idx(mat: np.ndarray) -> Dict[str, int]:
    m = mat.mean(axis=0); s = mat.std(axis=0)
    return {'max_var':int(np.nanargmax(s)), 'min_var':int(np.nanargmin(s)),
            'max_mean':int(np.nanargmax(m)), 'min_mean':int(np.nanargmin(m))}

def plot_8panel_synth_vs_real(synth: np.ndarray, real_w: np.ndarray,
                              real_cols: List[str], out_path: str, bins: int = 120):
    s_idx = _extreme_idx(synth); r_idx = _extreme_idx(real_w)
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), dpi=160)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    specs = [('Max variance','max_var'),('Min variance','min_var'),('Max mean','max_mean'),('Min mean','min_mean')]
    for r, (title, key) in enumerate(specs):
        ax = axes[r, 0]
        ax.hist(synth[:, s_idx[key]], bins=bins, alpha=0.9, edgecolor='none', color='tab:green')
        ax.set_title(f"({chr(ord('a') + r)}) {title} — Synthetic")
        ax.grid(alpha=0.25, linestyle=':')
        axr = axes[r, 1]
        j = r_idx[key]; cusip = real_cols[j] if (0 <= j < len(real_cols)) else "NA"
        axr.hist(real_w[:, j], bins=bins, alpha=0.90, edgecolor='none', color='tab:blue')
        axr.set_title(f"({chr(ord('a') + r)}) {title} — Real (CUSIP {cusip})")
        axr.set_ylabel("Frequency"); axr.grid(alpha=0.25, linestyle=':')
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"[fig] 8-panel (Synth vs Real-only) → {out_path}")


# ============================ 估计与绩效（保持不变） ============================

def estimate_mu_sigma(data: np.ndarray, use_lw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    mu = data.mean(axis=0)
    Sigma = LedoitWolf().fit(data).covariance_ if use_lw else np.cov(data.T, bias=False)
    return mu, Sigma

def compute_metrics(curve: np.ndarray, w_seq: List[Optional[np.ndarray]]) -> Dict[str, float]:
    if curve.size == 0:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan, turnover=np.nan)
    mu = curve.mean(); sd = curve.std(ddof=1)
    ann_ret = mu * 252.0; ann_vol = sd * np.sqrt(252.0)
    sharpe = ann_ret / (ann_vol + 1e-12)
    wealth = np.cumprod(1.0 + curve)
    peak = np.maximum.accumulate(wealth) + 1e-12
    max_dd = float(np.max(1.0 - wealth / peak))
    turns = []
    for i in range(1, len(w_seq)):
        if w_seq[i-1] is None or w_seq[i] is None: continue
        turns.append(0.5 * np.sum(np.abs(w_seq[i] - w_seq[i-1])))
    turnover = float(np.mean(turns)) if len(turns) else np.nan
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd, turnover=turnover)


# ============================ 归因分解（新增，仅新增） ============================

def _daily_cs_sigma_r(ret_win: np.ndarray) -> float:
    """训练窗日度横截面标准差的时间平均，用作 σ_r。"""
    if ret_win.ndim != 2: return float('nan')
    cs = np.nanstd(ret_win, axis=1, ddof=1)
    val = float(np.nanmean(cs))
    if not np.isfinite(val) or val < 1e-12: val = 1e-3
    return val

def _rho_rx(ret_win: np.ndarray, x: np.ndarray) -> float:
    """训练窗内按日计算 Corr(r_t, x) 并时间平均，得到 ρ。若样本不足，退化为 Corr(mean_r, x)。"""
    T, N = ret_win.shape
    x = np.asarray(x).reshape(-1)
    mask_x = np.isfinite(x)
    vals = []
    for t in range(T):
        r = ret_win[t, :]
        mask = mask_x & np.isfinite(r)
        if mask.sum() >= 5:
            c = np.corrcoef(r[mask], x[mask])[0, 1]
            if np.isfinite(c): vals.append(c)
    if len(vals) >= max(10, T//10):
        return float(np.nanmean(vals))
    # 退化：资产均值与 x 的横截面相关
    m = np.nanmean(ret_win, axis=0)
    mask = mask_x & np.isfinite(m)
    if mask.sum() >= 5:
        c = np.corrcoef(m[mask], x[mask])[0, 1]
        if np.isfinite(c): return float(c)
    return 0.0

def attrib_decompose(mu: np.ndarray, S: np.ndarray, x_esg: np.ndarray,
                     ret_win: np.ndarray, w_star: np.ndarray, gamma: float) -> Dict[str, float]:
    """
    按 Lo & Zhang（正态情形）计算单窗三分解（单位：日度效用），返回 benchmark/static/information/total。
    m_X = m + ρ * (σ_r/σ_x) * (x - mean_x)
    S_X = S - ρ^2 * σ_r^2 * I
    """
    n = len(mu); I = np.eye(n)
    sigma_x = float(np.nanstd(x_esg) + 1e-12)
    sigma_r = _daily_cs_sigma_r(ret_win)
    rho = _rho_rx(ret_win, x_esg)
    # 条件矩与超额协方差
    m_X = mu + (rho * sigma_r / sigma_x) * (x_esg - np.nanmean(x_esg))
    S_X = S - (rho ** 2) * (sigma_r ** 2) * I
    # 数值稳定性修正
    S_X = 0.5 * (S_X + S_X.T)
    try:
        min_eig = float(np.linalg.eigvalsh(S_X).min())
        if min_eig < 1e-8:
            S_X = S_X + (abs(min_eig) + 1e-6) * I
    except Exception:
        S_X = S + 1e-6 * I
    # v_MVO / v_CSTR / v_SHR
    v_MVO = np.linalg.solve(S + 1e-9 * I, mu) / gamma
    v_CSTR = w_star - v_MVO
    v_SHR = v_MVO + 0.5 * v_CSTR
    # 三分解
    bench = float(m_X @ v_MVO - 0.5 * gamma * (v_MVO @ S_X @ v_MVO))
    static = float(-0.5 * gamma * (v_CSTR @ S @ v_CSTR))
    info = float((m_X - mu) @ v_CSTR + gamma * (v_SHR @ (S_X - S) @ v_CSTR))
    total = bench + static + info
    # 验证闭合：应当 ≈ m_X' w* - 0.5 γ w*' S_X w*
    direct = float(m_X @ w_star - 0.5 * gamma * (w_star @ S_X @ w_star))
    if np.isfinite(direct) and np.isfinite(total):
        gap = abs(total - direct)
        if gap > 1e-6:
            # 打印一次，不影响流程
            print(f"[attrib] warn: closure gap={gap:.3e} (numerical)")
    return dict(benchmark=bench, static=static, information=info, total=total)


# ============================ Main（原有主体 + 末尾新增归因汇总） ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns_dir", type=str, required=True)
    ap.add_argument("--esg_dir",     type=str, required=True)
    ap.add_argument("--start", type=str, default="2003-01-01")
    ap.add_argument("--end",   type=str, default="2019-12-31")
    ap.add_argument("--min_stock_coverage", type=float, default=0.99)
    ap.add_argument("--target_n", type=int, default=1024)
    ap.add_argument("--seed",     type=int, default=3407)
    ap.add_argument("--epochs",       type=int,   default=600)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--warmup_iters", type=int,   default=20)
    ap.add_argument("--eta_min",      type=float, default=1e-5)
    ap.add_argument("--rebalance",   type=str, default="Q", choices=["M", "Q"])
    ap.add_argument("--keep_steps",  type=str, default="160,180")
    ap.add_argument("--synth_total", type=int, default=8192)
    ap.add_argument("--synth_batch", type=int, default=64)
    ap.add_argument("--eta",        type=float, default=3.0)
    ap.add_argument("--wmax",       type=float, default=0.05)
    ap.add_argument("--esg_levels", type=str,   default="none,50p,75p,90p")
    ap.add_argument("--esg_field",  type=str,   default="esg_13", choices=["esg_13", "esg_7"])
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--min_day_coverage", type=float, default=0.95)
    ap.add_argument("--real_blocklist", type=str, default="", help="path to CUSIP.txt; optional")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    keep_steps = [int(s.strip()) for s in args.keep_steps.split(",") if s.strip()]
    print(f"[cfg] MODEL_DIM={getattr(cfg,'MODEL_DIM','?')}  TIMESTEPS={getattr(cfg,'TIMESTEPS','?')}  BETA_SCHEDULE={getattr(cfg,'BETA_SCHEDULE','?')}")

    # IO（保持不变）
    print("[IO] load returns (PERMNO, safe-agg→pivot) ...")
    panel_permno, permno2cusip = load_returns_permno_panel(args.returns_dir, args.start, args.end)
    print(f"[RET/PERMNO] panel: T={panel_permno.shape[0]}, N={panel_permno.shape[1]} (with NaNs)")
    HARDCODE_BLOCK = {"63935M20", "63007510", "12525D10", "13261810", "31943910"}
    blocklist = set(HARDCODE_BLOCK)
    if args.real_blocklist and os.path.exists(args.real_blocklist):
        with open(args.real_blocklist, "r") as f:
            more = {line.strip() for line in f if line.strip()}
            blocklist |= more
        print(f"[blocklist] loaded {len(blocklist)} CUSIPs (hard-coded + file).")
    else:
        print(f"[blocklist] use hard-coded CUSIPs only: {sorted(blocklist)}")
    print("[IO] load returns (CUSIP, safe-agg→pivot) for Real-only hist ...")
    panel_cusip = load_returns_cusip_panel(args.returns_dir, args.start, args.end, blocklist=sorted(blocklist))
    print(f"[RET/CUSIP ] panel: T={panel_cusip.shape[0]}, N={panel_cusip.shape[1]} (with NaNs)")
    print("[IO] load ESG ...")
    esg_monthly = load_esg_scores(args.esg_dir, args.start, args.end, field=args.esg_field)
    print(f"[ESG] monthly: T={esg_monthly.shape[0]}, N={esg_monthly.shape[1]} (field={args.esg_field})\n")

    edges_all = quarter_ends(panel_permno.index) if args.rebalance == "Q" else month_ends(panel_permno.index)
    wy = 5
    start_dt, end_dt = pd.to_datetime(args.start), pd.to_datetime(args.end)
    rebal_dates = [d for d in edges_all if d >= (start_dt + pd.DateOffset(years=wy)) and d <= end_dt]

    port_curve: Dict[str, List[float]] = {}
    port_dates: List[pd.Timestamp] = []
    w_history: Dict[str, List[Optional[np.ndarray]]] = {}

    # 新增：按方法×ESG 收集“每窗”归因结果（只对 Real / Diff 两条主方法；EW 不做归因）
    attrib_bucket: Dict[Tuple[str,str], List[Dict[str, float]]] = {}

    Hdim, Wdim, capN = choose_hw_and_n(args.target_n)
    preN_list = [max(2 * capN, 2048), 3072, 4096, panel_cusip.shape[1]]

    for me in rebal_dates:
        win_start = me - pd.DateOffset(years=wy) + pd.DateOffset(days=1)
        win_end   = me

        win_panel = panel_permno[(panel_permno.index >= win_start) & (panel_permno.index <= win_end)]
        if win_panel.empty:
            print(f"[skip] {me.date()} empty window"); continue

        mapping = {pid: permno2cusip.get(int(pid), None) for pid in win_panel.columns}
        cols_use = [pid for pid, cs in mapping.items() if cs is not None]
        sub_panel = win_panel[cols_use].copy()
        sub_panel.columns = [mapping[int(pid)] for pid in cols_use]  # CUSIP 列

        esg_row = esg_monthly.loc[esg_monthly.index <= me].tail(1)
        if esg_row.empty:
            print(f"[skip] {me.date()} no ESG row"); continue
        esg_series_full = esg_row.iloc[0]  # index=CUSIP

        candidates = select_universe_1024(sub_panel, esg_series_full,
                                          min_stock_coverage=args.min_stock_coverage,
                                          target_n=capN)
        if len(candidates) < capN:
            print(f"[warn] {me.date()} only {len(candidates)} assets after shape filter; will proceed with fewer.")

        chosen, day_mask = greedy_intersection_days(sub_panel, candidates, min_T_targets=[750, 500, 250])
        sub_full = sub_panel[chosen].iloc[day_mask].dropna(how='any')
        if sub_full.empty:
            print(f"[skip] {me.date()} intersection T=0 even after greedy; skip window.")
            continue

        sub_w = my_winsorize(sub_full.values)
        Z, mean_vec, std_vec = standardize(sub_w)
        esg_vec = esg_series_full.loc[chosen].astype(float).values

        print(f"[train-ready] {me.date()} → T={Z.shape[0]}, N={Z.shape[1]}, HxW={Hdim}x{Wdim}, mean(std)={Z.mean():.4f}({Z.std():.4f})")

        exp_dir = os.path.join(args.out, f"wy5_{me.strftime('%Y%m')}_H{_slug_int(Hdim)}W{_slug_int(Wdim)}")
        os.makedirs(exp_dir, exist_ok=True)
        gen_paths = train_dfm_and_generate(
            Z.reshape(Z.shape[0], Hdim, Wdim), Hdim, Wdim, seed=args.seed,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, warmup_iters=args.warmup_iters,
            T_max=args.epochs, eta_min=args.eta_min,
            save_dir=exp_dir, synth_total=args.synth_total, synth_batch=args.synth_batch,
            keep_steps=keep_steps
        )

        gen_synth = None
        for step, path in gen_paths.items():
            arr = np.load(path)
            arr = unstandardize(arr, mean_vec, std_vec)
            arr = my_winsorize_light(arr, pct=0.5, cap=0.2)
            if step == 160:
                gen_synth = arr
        if gen_synth is None:
            k0 = next(iter(gen_paths.keys()))
            arr = np.load(gen_paths[k0])
            gen_synth = my_winsorize_light(unstandardize(arr, mean_vec, std_vec), pct=0.5, cap=0.2)

        # Real-only 直方图专用（保持不变）
        win_panel_cusip = panel_cusip[(panel_cusip.index >= win_start) & (panel_cusip.index <= win_end)]
        real_cols, _ = select_columns_with_shape_filter(win_panel=win_panel_cusip,
                                                        capN=capN,
                                                        min_stock_cov=args.min_stock_coverage,
                                                        preN_list=preN_list)
        sub_real = win_panel_cusip[real_cols].copy()
        real_ok = False
        for thr in [args.min_day_coverage, 0.90, 0.85, 0.80]:
            day_cov = sub_real.notna().mean(axis=1)
            sub_t = sub_real[day_cov >= thr].dropna(axis=0, how='any')
            if sub_t.shape[0] == 0: continue
            if thr != args.min_day_coverage:
                print(f"[day-cov/real] relaxed to {thr:.2f} → T={sub_t.shape[0]}")
            Xtmp = my_winsorize(sub_t.values)
            bad_zero  = zero_mass_spike_mask(Xtmp, frac_thresh=0.25, eps_floor=5e-4, eps_mult=0.12)
            bad_spiky = np.array([is_abnormal_spiky(Xtmp[:, k]) for k in range(Xtmp.shape[1])], dtype=bool)
            bad_uni   = ~smooth_unimodal_mask(Xtmp, bins=81, plateau_allow=3, spike_ratio_thresh=20.0)
            bad_center = np.array([center_mass_ratio(Xtmp[:, k]) > 0.10 for k in range(Xtmp.shape[1])], dtype=bool)
            keep_mask = ~(bad_zero | bad_spiky | bad_uni | bad_center)
            if keep_mask.sum() == 0:
                print("[re-shape] all columns filtered at this row-coverage; try next threshold."); continue
            dropped = (~keep_mask).sum()
            if dropped: print(f"[re-shape] drop {dropped} columns after row-filter; keep {keep_mask.sum()}")
            real_w_for_plot = Xtmp[:, keep_mask]
            real_cols = [c for c, keep in zip(real_cols, keep_mask) if keep]
            real_ok = True; break
        if not real_ok or real_w_for_plot.shape[1] == 0:
            print(f"[WARN] Real-only hist: no valid columns after final shape checks; draw synthetic-only.")
            real_w_for_plot = None

        try:
            if real_w_for_plot is not None:
                plot_8panel_synth_vs_real(gen_synth, real_w_for_plot,
                                          real_cols[:real_w_for_plot.shape[1]],
                                          out_path=os.path.join(exp_dir, "asset_distributions_8panel.png"),
                                          bins=args.bins)
            else:
                print("[note] draw synthetic-only 4 rows (no real overlay).")
                fig, axes = plt.subplots(4, 1, figsize=(7, 10), dpi=160)
                specs = [('Max variance','max_var'),('Min variance','min_var'),
                         ('Max mean','max_mean'),('Min mean','min_mean')]
                s_idx = _extreme_idx(gen_synth)
                for r, (title, key) in enumerate(specs):
                    ax = axes[r]
                    ax.hist(gen_synth[:, s_idx[key]], bins=args.bins, alpha=0.9, edgecolor='none', color='tab:green')
                    ax.set_title(f"({chr(ord('a') + r)}) {title} — Synthetic")
                    ax.grid(alpha=0.25, linestyle=':')
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, "asset_distributions_synth_only.png")); plt.close()
        except Exception as e:
            print(f"[warn] 8-panel failed: {e}")

        # μ,Σ（Real/Diff）
        mu_diff, Sigma_diff = estimate_mu_sigma(gen_synth, use_lw=True)
        mu_real, Sigma_real = estimate_mu_sigma(sub_w,      use_lw=True)

        # 下个持有期
        next_edges = [d for d in edges_all if d > me]
        if not next_edges: break
        nxt_end = next_edges[0]
        hold = panel_permno[(panel_permno.index > me) & (panel_permno.index <= nxt_end)]
        hold = hold[[p for p in hold.columns if p in permno2cusip]]
        hold.columns = [permno2cusip[int(p)] for p in hold.columns]
        hold = hold[chosen].dropna(how='any')
        if hold.empty: continue
        hold_vals = hold.values

        # ESG floors
        def parse_esg_levels(vec: np.ndarray, spec: str) -> Dict[str, Optional[float]]:
            out: Dict[str, Optional[float]] = {}
            for tok in [s.strip().lower() for s in spec.split(",") if s.strip()]:
                if tok == 'none':
                    out[tok] = None
                elif tok.endswith('p'):
                    p = float(tok[:-1]); out[tok] = np.nanpercentile(vec, p)
                else:
                    try: out[tok] = float(tok)
                    except: out[tok] = None
            return out
        floors = parse_esg_levels(esg_vec, args.esg_levels)

        combos = {
            'Diff Emp+Diff LW': (mu_diff, Sigma_diff),
            'Real Emp+Real LW': (mu_real, Sigma_real),
            'EW': (np.zeros_like(mu_real), np.eye(len(mu_real)))
        }
        for name, (mu_use, Sigma_use) in combos.items():
            for lbl, thr in floors.items():
                key = f"{name} | ESG:{lbl}"
                if name == 'EW':
                    w = np.ones(len(mu_real)) / len(mu_real)
                else:
                    w = solve_mv_with_esg(mu_use, Sigma_use, esg_vec.astype(float),
                                          eta=args.eta, wmax=args.wmax, esg_floor=thr)
                    s = w.sum()
                    if abs(s) > 1e-12: w = w / s
                rr = hold_vals @ w
                port_curve.setdefault(key, []); port_curve[key].extend(rr.tolist())
                w_history.setdefault(key, []).append(w.copy())

                # === 新增：记录“单窗”归因（三分解） ===
                if name != 'EW':
                    comp = attrib_decompose(mu_use, Sigma_use, esg_vec.astype(float),
                                            ret_win=sub_w, w_star=w, gamma=args.eta)
                    attrib_bucket.setdefault((name, lbl), []).append(comp)

        port_dates.extend(list(hold.index))
        gc.collect()
        try: torch.cuda.empty_cache()
        except: pass

    # ---- 原有：拼日收益、作累计图、写 performance_summary.csv（保持不变） ----
    if not port_dates:
        print("[WARN] No portfolio output — all windows skipped."); return
    idx = pd.DatetimeIndex(port_dates)
    df_ret = pd.DataFrame(index=idx)
    for k, v in port_curve.items():
        if len(v) == len(idx): df_ret[k] = np.array(v)
    df_ret.sort_index(inplace=True)
    ret_csv = os.path.join(args.out, "returns_by_strategy.csv")
    df_ret.to_csv(ret_csv, index=True)
    print(f"[csv] daily returns by strategy → {ret_csv}")

    plt.figure(figsize=(9, 5), dpi=200)
    for col in df_ret.columns:
        plt.plot(df_ret.index, np.log1p(df_ret[col].values).cumsum(), label=col, linewidth=1.3)
    plt.legend(fontsize=7, ncol=2)
    plt.ylabel("Cumulative Log-Return")
    title_reb = "Quarterly" if args.rebalance == "Q" else "Monthly"
    plt.title(f"Cumulative Log-Return ({title_reb} rebalance, 5Y rolling)")
    plt.tight_layout()
    out_pnl = os.path.join(args.out, "cumlog_pnl_wy5.png")
    plt.savefig(out_pnl); plt.close()
    print(f"[fig] cumulative pnl (ALL) → {out_pnl}")

    method2cols: Dict[str, List[str]] = {}
    for col in df_ret.columns:
        m = col.split(" | ESG:")[0]
        method2cols.setdefault(m, []).append(col)
    def esg_sort_key(s: str) -> Tuple[int, str]:
        lvl = s.split(" | ESG:")[1] if " | ESG:" in s else ""
        order = {"none": 0, "50p": 1, "75p": 2, "90p": 3}
        return (order.get(lvl, 99), lvl)
    for method, cols in method2cols.items():
        cols_sorted = sorted(cols, key=esg_sort_key)
        plt.figure(figsize=(9, 5), dpi=200)
        for col in cols_sorted:
            plt.plot(df_ret.index, np.log1p(df_ret[col].values).cumsum(), label=col.split(" | ESG:")[1], linewidth=1.6)
        plt.legend(title="ESG floor", fontsize=8)
        plt.ylabel("Cumulative Log-Return")
        plt.title(f"{method} — {title_reb} rebalance, 5Y rolling")
        plt.tight_layout()
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', method).strip('_').lower()
        out_p = os.path.join(args.out, f"cumlog_pnl_{slug}.png")
        plt.savefig(out_p); plt.close()
        print(f"[fig] cumulative pnl (per-method) → {out_p}")

    rows = []
    for key in df_ret.columns:
        metr = compute_metrics(df_ret[key].values, w_history.get(key, []))
        metr['strategy'] = key
        rows.append(metr)
    perf = pd.DataFrame(rows)[['strategy', 'ann_ret', 'ann_vol', 'sharpe', 'max_dd', 'turnover']]
    perf_path = os.path.join(args.out, "performance_summary.csv")
    perf.to_csv(perf_path, index=False)
    print(f"[csv] performance → {perf_path}")

    # ---- 新增：归因 CSV 与堆叠条形图（按方法一张，X=ESG 强度；Y=年化CE/效用） ----
    if len(attrib_bucket) > 0:
        out_rows = []
        # 收集全方法-ESG的年化平均
        methods = sorted(set(k[0] for k in attrib_bucket.keys()))
        esg_order = {"none": 0, "50p": 1, "75p": 2, "90p": 3}
        for mth in methods:
            # 排序后的 ESG 列表
            lvls = sorted([lvl for (meth, lvl) in attrib_bucket.keys() if meth == mth],
                          key=lambda x: esg_order.get(x, 99))
            # 作图数据
            x_labels, y_bench, y_static, y_info, y_total = [], [], [], [], []
            for lvl in lvls:
                arr = attrib_bucket[(mth, lvl)]
                # 跨窗求平均，并年化
                bench = float(np.nanmean([a['benchmark'] for a in arr])) * 252.0
                statc = float(np.nanmean([a['static']     for a in arr])) * 252.0
                info  = float(np.nanmean([a['information'] for a in arr])) * 252.0
                total = float(np.nanmean([a['total']       for a in arr])) * 252.0
                out_rows.append(dict(method=mth, esg_level=lvl, benchmark=bench,
                                     static=statc, information=info, total=total,
                                     annualized=True, windows=len(arr)))
                x_labels.append(lvl); y_bench.append(bench); y_static.append(statc); y_info.append(info); y_total.append(total)

            # 画图（堆叠：基准 -> 静态成本 -> 信息）
            if x_labels:
                x = np.arange(len(x_labels))
                plt.figure(figsize=(9, 5), dpi=200)
                b1 = plt.bar(x, y_bench, label='Benchmark')
                b2 = plt.bar(x, y_static, bottom=y_bench, label='Static cost')
                bottom_bench_static = (np.array(y_bench) + np.array(y_static)).tolist()
                b3 = plt.bar(x, y_info, bottom=bottom_bench_static, label='Information')
                plt.xticks(x, x_labels)
                plt.ylabel("Annualized CE utility")
                plt.title(f"Expected-utility attribution by ESG floor — {mth}")
                plt.legend()
                plt.tight_layout()
                slug = re.sub(r'[^a-zA-Z0-9]+', '_', mth).strip('_').lower()
                fig_path = os.path.join(args.out, f"attribution_bars_{slug}.png")
                plt.savefig(fig_path); plt.close()
                print(f"[fig] attribution bars → {fig_path}")

        # 写 CSV
        df_attr = pd.DataFrame(out_rows,
                               columns=['method','esg_level','benchmark','static','information','total','annualized','windows'])
        csv_path = os.path.join(args.out, "attribution_bars.csv")
        df_attr.to_csv(csv_path, index=False)
        print(f"[csv] attribution bars → {csv_path}")

    print("\n[Done] 5-year rolling finished (with attribution bars).")


if __name__ == "__main__":
    main()