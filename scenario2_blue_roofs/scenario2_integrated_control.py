# -*- coding: utf-8 -*-
"""
@author: Andres Felipe Cortes

Scenario 2 of the Astlingen benchmark: integrated real time control of
blue roofs and the combined sewer, one year at a 5 minute step. Two
objectives are pursued at the same time: minimise the discharge to the
receiving water, and maximise blue roof storage and irrigation.

The .inp has no [CONTROLS] rules, the orifices V2, V3, V4 and V6 are
driven from here instead.
"""

import os
import sys
import time
import datetime

import numpy as np
import pandas as pd
import pyswmm as ps

""" Paths. Everything is relative to this file so the folder can be moved"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAIN_DIR = os.path.join(BASE_DIR, '..', 'rainfall')

INP_FILE = 'scenario2_blue_roofs.inp'
RUN_NAME = os.path.splitext(INP_FILE)[0]

INP_PATH = os.path.join(BASE_DIR, INP_FILE)
OUT_PATH = os.path.join(BASE_DIR, RUN_NAME + '.out')
RPT_PATH = os.path.join(BASE_DIR, RUN_NAME + '.rpt')

RAIN_FILES = {'RG1': '1Astlingen_Erft1.txt',
              'RG2': '2Astlingen_Erft2.txt',
              'RG3': '3Astlingen_Erft3.txt',
              'RG4': '4Astlingen_Erft4.txt'}

LOOKAHEAD = datetime.timedelta(hours=2)

""" The roofs and the irrigation offtakes are driven by a flow limit in
    m3/s, not by a setting. SWMM reads a flow limit of exactly zero as no
    limit at all, which is fully open, so a closed link needs a very small
    non zero number instead"""

OPEN_FLOW = 0
CLOSE_FLOW = 0.0000001

IRR_DEMAND = {'Irr_1': 2.003e-06,
              'Irr_2': 1.381e-06,
              'Irr_3': 1.093e-06,
              'Irr_4': 4.188e-07,
              'Irr_5': 9.468e-07,
              'Irr_6': 1.976e-06,
              'Irr_7': 2.883e-07,
              'Irr_8': 1.700e-06,
              'Irr_9': 4.188e-07,
              'Irr_10': 7.132e-07}

""" The blue roofs. BR_1 to BR_10 are 0.30 m deep and add 900 m3 of
    storage over 10,326 m2 of roof. Each one has a controlled drain O_i to
    the sewer and an irrigation offtake Irr_i. BR_ROOF is the contributing
    roof area and BR_AREA the plan area of the store"""

BR_GAGE = {1: 'RG4', 2: 'RG3', 3: 'RG1', 4: 'RG3', 5: 'RG4',
           6: 'RG2', 7: 'RG3', 8: 'RG1', 9: 'RG2', 10: 'RG4'}

BR_ROOF = {1: 1891.0, 2: 1304.0, 3: 1032.0, 4: 395.0, 5: 894.0,
           6: 1865.0, 7: 272.0, 8: 1605.0, 9: 395.0, 10: 673.0}

BR_AREA = {1: 549.39, 2: 378.75, 3: 299.67, 4: 114.87, 5: 259.79,
           6: 541.90, 7: 79.08, 8: 466.15, 9: 114.87, 10: 195.62}

BR_MAX_DEPTH = 0.300
FULL_LEVEL = 0.255
RESERVE_LEVEL = 0.10 * BR_MAX_DEPTH

""" The nodes on each roof's own downstream path: the CSO node it
    discharges into where there is one, and the tank its water ends up in.
    BR_1 to BR_6 reach their tank through plain junctions. A node counts
    as at risk once it is 80% up its own max depth"""

BR_WATCH = {1: ['T1'], 2: ['T2'], 3: ['T3'], 4: ['T4'], 5: ['T5'],
            6: ['T6'], 7: ['CSO7', 'T4'], 8: ['CSO8', 'T3'],
            9: ['CSO9', 'T6'], 10: ['CSO10', 'T1']}

NODE_MAX_DEPTH = {'CSO7': 1.2, 'CSO8': 2.0, 'CSO9': 2.0, 'CSO10': 1.0,
                  'T1': 5.0, 'T2': 5.0, 'T3': 5.0,
                  'T4': 5.0, 'T5': 5.0, 'T6': 5.0}

RISK_FRACTION = 0.80
RISK_LEVEL = {node: RISK_FRACTION * d for node, d in NODE_MAX_DEPTH.items()}

""" Equal filling degree on the sewer tanks. The water stored on the roofs
    upstream of a tank goes into the numerator of the filling degree only,
    never into the denominator: the roofs are unevenly distributed, one
    tank carrying 24.7% of its combined storage on roofs against 8.1% for
    another, so dividing by the combined capacity would put each tank on a
    different scale"""

EFD_TANKS = ['T2', 'T3', 'T4', 'T6']
T_UPSTREAM_BR = {'T2': [2], 'T3': [3, 8], 'T4': [4, 7], 'T6': [6, 9]}

T_AREA = {'T2': 200.0, 'T3': 520.0, 'T4': 100.0, 'T6': 120.0}
T_MAX_DEPTH = 5.0
T_MAX_VOL = {t: T_AREA[t] * T_MAX_DEPTH for t in EFD_TANKS}

V_OF_TANK = {'T2': 'V2', 'T3': 'V3', 'T4': 'V4', 'T6': 'V6'}

EFD_BASE = {'V2': 0.2366, 'V3': 0.6508, 'V4': 0.3523, 'V6': 0.4303}
EFD_LOW = {'V2': 0.1075, 'V3': 0.3159, 'V4': 0.1894, 'V6': 0.1687}

""" The rainfall still to come over LOOKAHEAD is worked out once per gauge
    so the control loop only does a lookup. The timestamps are viewed as
    int64 because searching a datetime64 array with a scalar of a
    different unit converts the whole array on every call"""


def load_rain(path):
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['date', 'time', 'value'])
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    return df.set_index('datetime')['value']


rain_series = {gage: load_rain(os.path.join(RAIN_DIR, f))
               for gage, f in RAIN_FILES.items()}

STEP = rain_series['RG1'].index[1] - rain_series['RG1'].index[0]
WINDOW = int(LOOKAHEAD / STEP) + 1

rain_forecast = {gage: s[::-1].rolling(WINDOW, min_periods=1).sum()[::-1]
                 for gage, s in rain_series.items()}

FC_TIME = {gage: fs.index.values.astype('datetime64[us]').view('int64')
           for gage, fs in rain_forecast.items()}
FC_MM = {gage: fs.values for gage, fs in rain_forecast.items()}


def forecast_mm(gage, now):
    t = np.datetime64(now, 'us').view('int64')
    i = np.searchsorted(FC_TIME[gage], t, side='right') - 1
    return float(FC_MM[gage][i]) if i >= 0 else 0.0


""" The simulation. The tanks are solved first in each step and their
    verdict is recorded, so the roofs read the current decision instead of
    the previous one"""

print('Scenario 2, integrated control of blue roofs and the sewer')
print('  input  ', INP_PATH)
print('  output ', OUT_PATH)
print('One year at 5 minute steps, this takes several minutes.')
print()

start_clock = time.time()
last_shown = -1

with ps.Simulation(INP_PATH, RPT_PATH, OUT_PATH) as sim:
    links = ps.Links(sim)
    nodes = ps.Nodes(sim)
    gages = ps.RainGages(sim)
    sim.step_advance(300)

    # the roofs start closed so they fill rather than drain from the first
    # step
    for i in range(1, 11):
        links[f'O_{i}'].flow_limit = CLOSE_FLOW

    for step in sim:
        now = sim.current_time

        done = int(sim.percent_complete * 100)
        if done > last_shown:
            last_shown = done
            sys.stdout.write(f'\r  {done:3d}%  {now:%Y-%m-%d %H:%M}'
                             f'  elapsed {time.time() - start_clock:5.0f}s')
            sys.stdout.flush()

        # equal filling degree, counting the water on the roofs. The
        # fullest tank drains, the emptiest holds back
        efd_setting = {}
        fill_degree = {}
        for t in EFD_TANKS:
            volume = nodes[t].depth * T_AREA[t]
            for i in T_UPSTREAM_BR[t]:
                volume += nodes[f'BR_{i}'].depth * BR_AREA[i]
            fill_degree[t] = volume / T_MAX_VOL[t]

        for t in EFD_TANKS:
            others = [fill_degree[o] for o in EFD_TANKS if o != t]
            v = V_OF_TANK[t]
            if all(fill_degree[t] > o for o in others):
                setting = 1.0
            elif all(fill_degree[t] < o for o in others):
                setting = EFD_LOW[v]
            else:
                setting = EFD_BASE[v]
            links[v].target_setting = setting
            efd_setting[t] = setting

        forecast = {gage: forecast_mm(gage, now) for gage in RAIN_FILES}

        # the roofs. Irrigation is off while it rains and at demand
        # otherwise. The drain then follows five questions, first match
        # winning: release if nearly full, hold if the path downstream is
        # at risk or equal filling degree has flagged the tank as the
        # fullest, release ahead of a storm that will not fit, release in
        # dry weather above the irrigation reserve, otherwise hold
        for i in range(1, 11):
            gage = BR_GAGE[i]

            if gages[gage].rainfall > 0:
                links[f'Irr_{i}'].flow_limit = CLOSE_FLOW
            else:
                links[f'Irr_{i}'].flow_limit = IRR_DEMAND[f'Irr_{i}']

            depth = nodes[f'BR_{i}'].depth
            tank = BR_WATCH[i][-1]
            tank_is_fullest = efd_setting.get(tank, 0.0) >= 1.0
            uds_at_risk = (tank_is_fullest
                           or any(nodes[n].depth >= RISK_LEVEL[n]
                                  for n in BR_WATCH[i]))
            free_capacity = (BR_MAX_DEPTH - depth) * BR_AREA[i]
            expected_inflow = forecast[gage] / 1000.0 * BR_ROOF[i]

            if depth >= FULL_LEVEL:
                links[f'O_{i}'].flow_limit = OPEN_FLOW
            elif uds_at_risk:
                links[f'O_{i}'].flow_limit = CLOSE_FLOW
            elif expected_inflow > free_capacity:
                links[f'O_{i}'].flow_limit = OPEN_FLOW
            elif forecast[gage] == 0 and depth > RESERVE_LEVEL:
                links[f'O_{i}'].flow_limit = OPEN_FLOW
            else:
                links[f'O_{i}'].flow_limit = CLOSE_FLOW

print()
print()
print(f'Finished in {time.time() - start_clock:.0f} s')
print('  ' + os.path.basename(OUT_PATH))
print('  ' + os.path.basename(RPT_PATH))
print()
print('Score it with ../evaluation/paper_objectives.py --scenario 2')
