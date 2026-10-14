# -*- coding: utf-8 -*-
"""
@author: Andres Felipe Cortes

Scenario 1 of the Astlingen benchmark: integrated real time control of
swales, groundwater and the combined sewer, one year at a 5 minute step.
Three objectives are pursued at the same time: minimise the discharge to
the receiving water, maximise swale storage and infiltration, and
stabilise the groundwater table.

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

INP_FILE = 'scenario1_swales.inp'
RUN_NAME = os.path.splitext(INP_FILE)[0]

INP_PATH = os.path.join(BASE_DIR, INP_FILE)
OUT_PATH = os.path.join(BASE_DIR, RUN_NAME + '.out')
RPT_PATH = os.path.join(BASE_DIR, RUN_NAME + '.rpt')

RAIN_FILES = {'RG1': '1Astlingen_Erft1.txt',
              'RG2': '2Astlingen_Erft2.txt',
              'RG3': '3Astlingen_Erft3.txt',
              'RG4': '4Astlingen_Erft4.txt'}

LOOKAHEAD = datetime.timedelta(hours=2)

OPEN = 1.0
CLOSED = 0.0

SWALES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

""" The swales. Each one has three outlets: Inf_i infiltrates to the
    groundwater GW1, Ov_i is a 0.30 m weir spilling to the sewer, and Ud_i
    is the underdrain to the river SW. Only Ud_i is controlled"""

SWALE_GAGE = {1: 'RG4', 2: 'RG3', 3: 'RG1', 4: 'RG3', 5: 'RG4',
              6: 'RG2', 7: 'RG3', 8: 'RG1', 9: 'RG2', 10: 'RG4'}

SWALE_CATCH = {1: 16500.0, 2: 11375.0, 3: 9000.0, 4: 3450.0, 5: 7800.0,
               6: 16275.0, 7: 2375.0, 8: 14000.0, 9: 3450.0, 10: 5875.0}

SWALE_AREA = {1: 2340.0, 2: 1360.0, 3: 1060.0, 4: 390.0, 5: 970.0,
              6: 2480.0, 7: 260.0, 8: 1920.0, 9: 400.0, 10: 720.0}

SWALE_TANK = {1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4', 5: 'T5',
              6: 'T6', 7: 'T4', 8: 'T3', 9: 'T6', 10: 'T1'}

SWALE_PONDING = 0.300

""" Release thresholds, in m of the 0.300 m ponding depth. The swale
    releases at 0.255 m normally, holds up to 0.285 m when the table is
    low so the water infiltrates, and empties down to 0.010 m when the
    table is high so it goes to the river instead"""

RELEASE_NORMAL = 0.255
RELEASE_DRY_GW = 0.285
RELEASE_WET_GW = 0.010
SAFETY_LEVEL = 0.285

""" Equal filling degree on the sewer tanks. The nodes are 8 m deep but
    the overflow weirs sit at 5 m, so 5 m is the level that matters. The
    expected swale spill goes into the numerator of the filling degree
    only, never into the denominator, otherwise each tank ends up on a
    different scale and the four stop being comparable"""

EFD_TANKS = ['T2', 'T3', 'T4', 'T6']
T_UPSTREAM_SWALE = {'T2': [2], 'T3': [3, 8], 'T4': [4, 7], 'T6': [6, 9]}

T_AREA = {'T2': 200.0, 'T3': 520.0, 'T4': 100.0, 'T6': 120.0}
T_WEIR_CREST = 5.0
T_MAX_VOL = {t: T_AREA[t] * T_WEIR_CREST for t in EFD_TANKS}

V_OF_TANK = {'T2': 'V2', 'T3': 'V3', 'T4': 'V4', 'T6': 'V6'}

EFD_BASE = {'V2': 0.2366, 'V3': 0.6508, 'V4': 0.3523, 'V6': 0.4303}
EFD_LOW = {'V2': 0.1075, 'V3': 0.3159, 'V4': 0.1894, 'V6': 0.1687}

""" The groundwater. G_to_S drains the aquifer into the river and S_to_G
    recharges it. Nothing happens inside the dead band and the orifices
    are fully open once the error reaches GW_SPAN. Outside GW_FAR_LOW and
    GW_FAR_HIGH the swales are recruited as well. The table moves at most
    about 86 mm a day, so it is reset every six hours rather than every
    step"""

GW_TARGET = 1.02
GW_BAND = 0.05
GW_SPAN = 0.25

GW_FAR_HIGH = GW_TARGET + 0.15
GW_FAR_LOW = GW_TARGET - 0.10

DRAIN_GW = 'G_to_S'
RECHARGE_GW = 'S_to_G'

GW_UPDATE = datetime.timedelta(hours=6)

""" The swales and the sewer never meet in a pipe, they meet in the river.
    W_1 to W_6 are the tank overflows and W_7 to W_10 the CSO structures"""

SEWER_WEIRS = [f'W_{i}' for i in range(1, 11)]
SPILL_FLOW = 0.001

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


""" The simulation. The settings are written every time step, applying
    them once before the loop holds for orifices but not for weirs"""

print('Scenario 1, integrated control of swales, groundwater and sewer')
print('  input  ', INP_PATH)
print('  output ', OUT_PATH)
print('One year at 5 minute steps, this takes several minutes.')
print()

start_clock = time.time()
last_shown = -1
next_gw_update = None

with ps.Simulation(INP_PATH, RPT_PATH, OUT_PATH) as sim:
    links = ps.Links(sim)
    nodes = ps.Nodes(sim)
    sim.step_advance(300)

    for step in sim:
        now = sim.current_time

        done = int(sim.percent_complete * 100)
        if done > last_shown:
            last_shown = done
            sys.stdout.write(f'\r  {done:3d}%  {now:%Y-%m-%d %H:%M}'
                             f'  elapsed {time.time() - start_clock:5.0f}s')
            sys.stdout.flush()

        # infiltration and the passive spill are never controlled
        for i in SWALES:
            links[f'Inf_{i}'].target_setting = OPEN
            links[f'Ov_{i}'].target_setting = OPEN

        forecast = {gage: forecast_mm(gage, now) for gage in RAIN_FILES}

        # swale state, needed by both the tanks and the swales. Only the
        # part of the forecast rain that will not fit can reach the sewer
        swale_depth = {}
        expected_spill = {}
        for i in SWALES:
            d = nodes[f'S_{i}'].depth
            swale_depth[i] = d
            free = (SWALE_PONDING - d) * SWALE_AREA[i]
            inflow = forecast[SWALE_GAGE[i]] / 1000.0 * SWALE_CATCH[i]
            expected_spill[i] = max(inflow - free, 0.0)

        sewer_spilling = any(links[w].flow > SPILL_FLOW for w in SEWER_WEIRS)

        # equal filling degree, anticipating the swale spill. The fullest
        # tank drains, the emptiest holds back
        fill_degree = {}
        for t in EFD_TANKS:
            volume = nodes[t].depth * T_AREA[t]
            for i in T_UPSTREAM_SWALE[t]:
                volume += expected_spill[i]
            fill_degree[t] = volume / T_MAX_VOL[t]

        for t in EFD_TANKS:
            others = [fill_degree[o] for o in EFD_TANKS if o != t]
            v = V_OF_TANK[t]
            if all(fill_degree[t] > o for o in others):
                links[v].target_setting = 1.0
            elif all(fill_degree[t] < o for o in others):
                links[v].target_setting = EFD_LOW[v]
            else:
                links[v].target_setting = EFD_BASE[v]

        # the exchange orifices, reset every GW_UPDATE. They stay shut
        # while the sewer is already discharging into the river
        gw_depth = nodes['GW1'].depth
        error = gw_depth - GW_TARGET

        if next_gw_update is None or now >= next_gw_update:
            over = abs(error) - GW_BAND
            setting = min(over / (GW_SPAN - GW_BAND), 1.0) if over > 0 else CLOSED
            if sewer_spilling:
                links[DRAIN_GW].target_setting = CLOSED
                links[RECHARGE_GW].target_setting = CLOSED
            elif error > GW_BAND:
                links[DRAIN_GW].target_setting = setting
                links[RECHARGE_GW].target_setting = CLOSED
            elif error < -GW_BAND:
                links[DRAIN_GW].target_setting = CLOSED
                links[RECHARGE_GW].target_setting = setting
            else:
                links[DRAIN_GW].target_setting = CLOSED
                links[RECHARGE_GW].target_setting = CLOSED
            next_gw_update = now + GW_UPDATE

        # the underdrains, reading the groundwater and the sewer
        gw_far_high = gw_depth > GW_FAR_HIGH
        gw_far_low = gw_depth < GW_FAR_LOW

        for i in SWALES:
            depth = swale_depth[i]
            free_capacity = (SWALE_PONDING - depth) * SWALE_AREA[i]
            expected_inflow = (forecast[SWALE_GAGE[i]] / 1000.0
                               * SWALE_CATCH[i])

            if gw_far_high:
                release_at = RELEASE_WET_GW
            elif gw_far_low:
                release_at = RELEASE_DRY_GW
            else:
                release_at = RELEASE_NORMAL

            if depth >= SAFETY_LEVEL:
                links[f'Ud_{i}'].target_setting = OPEN
            elif sewer_spilling:
                links[f'Ud_{i}'].target_setting = CLOSED
            elif depth >= release_at:
                links[f'Ud_{i}'].target_setting = OPEN
            elif expected_inflow > free_capacity:
                links[f'Ud_{i}'].target_setting = OPEN
            else:
                links[f'Ud_{i}'].target_setting = CLOSED

print()
print()
print(f'Finished in {time.time() - start_clock:.0f} s')
print('  ' + os.path.basename(OUT_PATH))
print('  ' + os.path.basename(RPT_PATH))
print()
print('Score it with ../evaluation/paper_objectives.py --scenario 1')
