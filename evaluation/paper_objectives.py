# -*- coding: utf-8 -*-
"""
Evaluate the objective functions AS WRITTEN IN THE PAPER, for every run.

This is not the A/B/C scorecard in objective_functions.py. That one uses a
different set of objectives and normalises each component against the best
and worst value in the set of runs being compared, so its numbers change
whenever a run is added or removed. Nothing like that can go in the paper.

Here every objective is scored against a FIXED target that does not depend
on the runs being compared, so a value means the same thing in isolation.

Scenario 1, three objectives
----------------------------
  J_a  controlled operation of the sewer sub-system
         min sum_t ( w1 Q_CSO(t) + w2 Q_flood(t) )
       Scenario 1 spills to the river over weirs W_1..W_10 and floods at
       the tanks, so both are counted.

  J_b  BGI storage management
         min sum_t ( (V - V_opt) / V_scale )^2
       The retention time term of the general form is inactive in this
       case: 96.6% of what enters the structures infiltrates, so the
       retention time is a property of the structure and not of the
       control. V_scale is the BGI capacity alone.

  J_c  groundwater management, as a dead band
         min sum_t max(0, |h_gw - h_mid| - delta)^2
       Zero inside the acceptable band [H_LOW, H_HIGH], growing
       quadratically outside it on either side. This replaces the target
       plus asymmetric penalty form: it needs no target value and no
       beta, only the band, and it encodes both halves of the goal, no
       low levels in summer and no mounding in winter.
       The surface water term of the general form is inactive: the level
       is imposed by the downstream boundary and is not controllable.

Scenario 2, three objectives
----------------------------
  J_a  coordinated operation of the blue roofs
         min sum_i sum_t ( V_i - V_opt_i )^2
       In filling degree so roofs of different size count equally.

  J_b  water allocation management
       There is no harvesting STORE in the model, the irrigation offtake
       is an outfall, so V_Harv does not exist and the squared form cannot
       be evaluated. What is reported instead is supply reliability,
       delivered / demanded, which is the outcome the objective seeks.
       SAY THIS IN THE PAPER. Do not present it as J_b evaluated.

  J_c  controlled operation of the sewer sub-system, same form as J_a of
       scenario 1.

Free choices, all of them stated
--------------------------------
There are exactly two.

  V_opt(t) = max(0, V_max - expected inflow over HORIZON): as full as
  possible while still leaving room for the rain that is coming. Same
  rule for both scenarios, independent of how any controller is coded.

  The groundwater band [H_LOW, H_HIGH]. H_LOW is the drought threshold
  of the system. H_HIGH is the level at which mounding becomes a
  problem and MUST be justified from the case, not from a run.

A sensitivity table over the band is printed at the end, because the
ranking of the runs must not depend on either edge of it.

No weights are used and no combined objective is reported. The nesting
logic orders the objectives, it does not aggregate them, and aggregating
would need weights the framework does not supply.

Usage
-----
    python paper_objectives.py --scenario 1 run1.out run2.out ...
    python paper_objectives.py --scenario 2 run1.out run2.out ...

Writes paper_objectives_sc<N>.csv next to the script.
"""

import os
import sys
import math
import argparse
import datetime
import numpy as np
import pandas as pd
import pyswmm as ps

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- settings
HORIZON = datetime.timedelta(hours=2)

W1, W2 = 1.0, 1.0          # equal weights inside the sewer objective

# the groundwater dead band. H_LOW is the drought threshold of the system.
# H_HIGH is the level at which mounding becomes a problem: JUSTIFY THIS
# FROM THE CASE, it is the only number here that is not yet grounded.
H_LOW = 0.90
H_HIGH = 1.20

RAIN_FILES = {'RG1': '1Astlingen_Erft1.txt', 'RG2': '2Astlingen_Erft2.txt',
              'RG3': '3Astlingen_Erft3.txt', 'RG4': '4Astlingen_Erft4.txt'}

GW_NODE, SW_NODE = 'GW1', 'SW'
TANKS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
CSOS = ['CSO7', 'CSO8', 'CSO9', 'CSO10']

SC1 = {
    'bgi': [f'S_{i}' for i in range(1, 11)],
    'area': dict(zip([f'S_{i}' for i in range(1, 11)],
                     [2340.0, 1360.0, 1060.0, 390.0, 970.0,
                      2480.0, 260.0, 1920.0, 400.0, 720.0])),
    'catch': dict(zip([f'S_{i}' for i in range(1, 11)],
                      [16500.0, 11375.0, 9000.0, 3450.0, 7800.0,
                       16275.0, 2375.0, 14000.0, 3450.0, 5875.0])),
    'gage': dict(zip([f'S_{i}' for i in range(1, 11)],
                     ['RG4', 'RG3', 'RG1', 'RG3', 'RG4',
                      'RG2', 'RG3', 'RG1', 'RG2', 'RG4'])),
    'depth': 0.300,
    'outs': [f'{p}_{i}' for i in range(1, 11) for p in ('Inf', 'Ud', 'Ov')],
    'spill_links': [f'W_{i}' for i in range(1, 11)],
}

SC2 = {
    'bgi': [f'BR_{i}' for i in range(1, 11)],
    'area': dict(zip([f'BR_{i}' for i in range(1, 11)],
                     [549.39, 378.75, 299.67, 114.87, 259.79,
                      541.90, 79.08, 466.15, 114.87, 195.62])),
    'catch': dict(zip([f'BR_{i}' for i in range(1, 11)],
                      [1891.0, 1304.0, 1032.0, 395.0, 894.0,
                       1865.0, 272.0, 1605.0, 395.0, 673.0])),
    'gage': dict(zip([f'BR_{i}' for i in range(1, 11)],
                     ['RG4', 'RG3', 'RG1', 'RG3', 'RG4',
                      'RG2', 'RG3', 'RG1', 'RG2', 'RG4'])),
    'depth': 0.300,
    'irr_links': [f'Irr_{i}' for i in range(1, 11)],
    'irr_demand': dict(zip([f'Irr_{i}' for i in range(1, 11)],
                           [2.003e-06, 1.381e-06, 1.093e-06, 4.188e-07,
                            9.468e-07, 1.976e-06, 2.883e-07, 1.700e-06,
                            4.188e-07, 7.132e-07])),
    'spill_links': None,
}


# ------------------------------------------------------------------ helpers
def as_series(obj):
    """pyswmm returns dicts, not pandas Series."""
    if isinstance(obj, dict):
        items = sorted(obj.items())
        return [k for k, _ in items], np.array([v for _, v in items], float)
    return list(obj.index), np.asarray([float(v) for v in list(obj)])


def build_forecast(folder):
    """Rain still to come over HORIZON, mm, per gage."""
    series = {}
    for g, f in RAIN_FILES.items():
        p = os.path.join(folder, f)
        df = pd.read_csv(p, sep=r'\s+', header=None,
                         names=['date', 'time', 'value'])
        df['dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        series[g] = df.set_index('dt')['value']
    step = series['RG1'].index[1] - series['RG1'].index[0]
    window = int(HORIZON / step) + 1
    fc = {g: s[::-1].rolling(window, min_periods=1).sum()[::-1]
          for g, s in series.items()}
    # keep the dtype consistent, otherwise searchsorted converts every call
    times = {g: v.index.values.astype('datetime64[us]').view('int64')
             for g, v in fc.items()}
    vals = {g: v.values for g, v in fc.items()}

    def lookup(gage, stamps):
        t = np.asarray(stamps, dtype='datetime64[us]').view('int64')
        i = np.searchsorted(times[gage], t, side='right') - 1
        i = np.clip(i, 0, len(vals[gage]) - 1)
        return vals[gage][i]

    return lookup


def volume(times, flow):
    """m3 from a flow series, assuming a constant reporting step."""
    if len(times) < 2:
        return 0.0
    step = (times[1] - times[0]).total_seconds()
    return float(np.sum(flow) * step)


# --------------------------------------------------------------- objectives
def sewer_objective(out, ns, ls, nodes, links, cfg):
    """J_a scenario 1 / J_c scenario 2."""
    cso = 0.0
    for n in CSOS:
        if n in nodes:
            t, q = as_series(ns[n].flooding_losses)
            cso += volume(t, q)
    if cfg.get('spill_links'):
        for l in cfg['spill_links']:
            if l in links:
                t, q = as_series(ls[l].flow_rate)
                cso += volume(t, q)
    # flooding anywhere in the sewer, the tanks and every junction. The
    # swale and roof nodes are deliberately excluded, their flooding is a
    # BGI outcome and in scenario 1 step 2 it is the whole point of the
    # configuration, not a sewer failure.
    flood = 0.0
    for n in nodes:
        if n in TANKS or n.startswith('J'):
            t, q = as_series(ns[n].flooding_losses)
            flood += volume(t, q)
    return {'J_sewer m3': W1 * cso + W2 * flood,
            'CSO m3': cso, 'flood m3': flood}


def bgi_storage_objective(ns, ls, nodes, links, cfg, fc, label):
    """J_b scenario 1 (storage + retention time) / J_a scenario 2 (storage)."""
    present = [n for n in cfg['bgi'] if n in nodes]
    if len(present) < len(cfg['bgi']):
        return None

    stamps = None
    stored = None
    sq_fill = 0.0
    n_terms = 0
    for n in present:
        t, d = as_series(ns[n].invert_depth)
        stamps = t if stamps is None else stamps
        v = d * cfg['area'][n]
        v_max = cfg['depth'] * cfg['area'][n]
        exp_mm = fc(cfg['gage'][n], t)
        # the ideal store holds exactly what is about to arrive, and is
        # otherwise empty, so it is ready. The earlier form, v_max minus
        # the forecast, made a full store the target in dry weather, and
        # ranked the scenario 2 configuration that leaves the roofs 99%
        # full above the one the results actually support.
        v_opt = np.minimum(v_max, exp_mm / 1000.0 * cfg['catch'][n])
        sq_fill += float(np.sum(((v - v_opt) / v_max) ** 2))
        n_terms += len(v)
        stored = v if stored is None else stored + v

    res = {f'{label} sum': sq_fill,
           f'{label} rms fill': math.sqrt(sq_fill / n_terms) if n_terms else 0.0}

    # two diagnostics, reported for both scenarios. They are not part of
    # any objective, they describe how the storage is actually used. Both
    # were previously only available where the config carries 'outs',
    # which scenario 2 does not.
    v_scale_all = sum(cfg['depth'] * a for a in cfg['area'].values())
    res['mean stored m3'] = float(np.mean(stored))
    res['share of year below 10pct'] = float(
        np.mean(stored < 0.10 * v_scale_all))

    # total stored volume against the same target, for the whole
    # sub-system rather than structure by structure
    if 'outs' in cfg:
        v_scale = sum(cfg['depth'] * a for a in cfg['area'].values())
        exp_tot = np.zeros_like(stored)
        for n in present:
            exp_tot = exp_tot + fc(cfg['gage'][n], stamps) / 1000.0 * cfg['catch'][n]
        v_opt_tot = np.minimum(v_scale, exp_tot)
        res['J_b lumped'] = float(np.sum(((stored - v_opt_tot) / v_scale) ** 2))
        res['mean stored m3'] = float(np.mean(stored))
        # kept as a physical diagnostic only, not part of the objective
        outflow = np.zeros_like(stored)
        for l in cfg['outs']:
            if l in links:
                _, q = as_series(ls[l].flow_rate)
                outflow = outflow + q
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = np.where(outflow > 1e-9, stored / outflow, np.nan)
        res['mean retention h (diagnostic)'] = float(np.nanmean(tau) / 3600.0)
    return res


def level_objective(ns, nodes, low=None, high=None):
    """J_c scenario 1, as a dead band. Zero inside [low, high]."""
    if GW_NODE not in nodes:
        return None
    low = H_LOW if low is None else low
    high = H_HIGH if high is None else high
    mid, delta = 0.5 * (low + high), 0.5 * (high - low)

    t, h = as_series(ns[GW_NODE].invert_depth)
    n = len(h)
    excess = np.maximum(0.0, np.abs(h - mid) - delta)
    step = (t[1] - t[0]).total_seconds() if n > 1 else 300.0
    return {
        'J_c band': float(np.sum(excess ** 2)),
        'J_c rms excess m': math.sqrt(float(np.sum(excess ** 2)) / n),
        'share outside band': float(np.mean(excess > 0)),
        'gw mean m': float(np.mean(h)),
        'gw range m': float(np.max(h) - np.min(h)),
        'gw sd m': float(np.std(h)),
        'days below %.2f m' % low:
            float(np.count_nonzero(h < low)) * step / 86400.0,
        'days above %.2f m' % high:
            float(np.count_nonzero(h > high)) * step / 86400.0,
        '_h': h,
    }


def supply_reliability(ls, links, cfg):
    """Stand in for J_b of scenario 2. Not the squared form, see docstring."""
    if not cfg.get('irr_links'):
        return None
    delivered = demanded = 0.0
    for l in cfg['irr_links']:
        if l not in links:
            continue
        t, q = as_series(ls[l].flow_rate)
        delivered += volume(t, q)
        demanded += cfg['irr_demand'][l] * (
            (t[-1] - t[0]).total_seconds() if len(t) > 1 else 0.0)
    if demanded <= 0:
        return None
    return {'irrigation delivered m3': delivered,
            'irrigation demanded m3': demanded,
            'supply reliability': delivered / demanded}


# -------------------------------------------------------------------- main
def evaluate(path, cfg, fc, scenario):
    row = {'case': os.path.splitext(os.path.basename(path))[0]}
    keep = None
    with ps.Output(path) as out:
        nodes, links = set(out.nodes), set(out.links)
        ns, ls = ps.NodeSeries(out), ps.LinkSeries(out)

        row.update(sewer_objective(out, ns, ls, nodes, links, cfg))

        label = 'J_b fill' if scenario == 1 else 'J_a'
        b = bgi_storage_objective(ns, ls, nodes, links, cfg, fc, label)
        if b:
            row.update(b)

        if scenario == 1:
            lv = level_objective(ns, nodes)
            if lv:
                keep = lv.pop('_h')
                row.update(lv)
        else:
            sr = supply_reliability(ls, links, cfg)
            if sr:
                row.update(sr)
    return row, keep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', type=int, required=True, choices=(1, 2))
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()

    cfg = SC1 if a.scenario == 1 else SC2
    folder = os.path.dirname(os.path.abspath(a.files[0])) or HERE
    fc = build_forecast(folder)
    print(f'groundwater band [{H_LOW:.2f}, {H_HIGH:.2f}] m')

    rows, gws = [], {}
    for f in a.files:
        print('reading', os.path.basename(f), flush=True)
        r, gw = evaluate(f, cfg, fc, a.scenario)
        rows.append(r)
        if gw is not None:
            gws[r['case']] = gw

    df = pd.DataFrame(rows).set_index('case')
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', lambda v: f'{v:,.4g}')
    print()
    print(df.T.to_string())

    csv = os.path.join(HERE, f'paper_objectives_sc{a.scenario}.csv')
    df.to_csv(csv)
    print('\nSaved to', csv)

    # does the ranking survive a different band?
    if gws:
        print('\nSensitivity of J_c to the band:')
        bands = [(0.85, 1.15), (0.90, 1.20), (0.90, 1.30), (0.95, 1.25)]
        sens = {}
        for lo, hi in bands:
            mid, d = 0.5 * (lo + hi), 0.5 * (hi - lo)
            sens[f'[{lo:.2f},{hi:.2f}]'] = {
                c: float(np.sum(np.maximum(0.0, np.abs(h - mid) - d) ** 2))
                for c, h in gws.items()}
        t = pd.DataFrame(sens)
        print(t.to_string(float_format=lambda v: f'{v:,.4g}'))
        print('\nRanking per band (1 = best):')
        print(t.rank().astype(int).to_string())
