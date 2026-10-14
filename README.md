# Integrated real-time control of blue-green infrastructure

Model files, control scripts and results for the two Astlingen scenarios
reported in the paper. Everything needed to reproduce the integrated runs
is in this folder; no absolute paths and no files outside it.

## Layout

```
scenario1_swales/
    scenario1_swales.inp              SWMM model, swales + groundwater + river
    scenario1_integrated_control.py   the integrated controller
    SW_downstream_level.txt           river stage boundary

scenario2_blue_roofs/
    scenario2_blue_roofs.inp          SWMM model, blue roofs + sewer
    scenario2_integrated_control.py   the integrated controller

rainfall/
    1Astlingen_Erft1.txt              gauge RG1, 5 min, year 2000
    2Astlingen_Erft2.txt              gauge RG2
    3Astlingen_Erft3.txt              gauge RG3
    4Astlingen_Erft4.txt              gauge RG4
    rainfall_2000.csv                 the same four records, one csv

evaluation/
    paper_objectives.py               objective functions

results/
    objectives_scenario1.csv          objectives, all stages
    objectives_scenario2.csv
    sensitivity_scenario1.csv         set-point sensitivity analysis
    sensitivity_scenario2.csv
```

## Requirements

Python 3.

```
pip install -r requirements.txt
```

`pyswmm` is pinned to 2.1.0, the version the reported runs were made
with. It carries the SWMM engine itself, so the volumes below are a
property of that version and another one may not reproduce them exactly.

## Running

```
cd scenario1_swales
python scenario1_integrated_control.py

cd scenario2_blue_roofs
python scenario2_integrated_control.py
```

Each simulates one year at a 5 minute time step and takes several
minutes. The `.out` and `.rpt` are written next to the `.inp`.

## Scoring

```
cd evaluation
python paper_objectives.py --scenario 1 ../scenario1_swales/scenario1_swales.out
python paper_objectives.py --scenario 2 ../scenario2_blue_roofs/scenario2_blue_roofs.out
```

Expected values for the integrated configurations:

| | Obj A, m3/yr | Obj B, rms filling | Obj C |
|---|---|---|---|
| Scenario 1 | 188,370 | 0.1144 | 0.2221 |
| Scenario 2 | 239,581 | 0.0901 | irrigation 0.955 |

## The objectives

| | |
|---|---|
| **A** | minimise discharge to the receiving water |
| **B** | maximise BGI storage and its beneficial use, infiltration for the swales, irrigation for the roofs |
| **C** | stabilise the groundwater table, scenario 1 only |

Objective A is reported directly in m3 per year: with equal weights the
sewer objective reduces to the total volume discharged. B and C are
reported through the variables that constitute them. The values are
therefore the terms of the objective functions rather than the functions
themselves, which is sufficient here, since the comparison is between
configurations under a fixed control procedure rather than the
performance of an optimised one.

## Notes on the models

Neither `.inp` contains `[CONTROLS]` rules. In both scenarios the four
EFD orifices V2, V3, V4 and V6 are driven from the Python script, and
leaving the native rules in place would make SWMM and the script fight
over the same actuators.

The rainfall records are the Astlingen Erft gauges, cut to the simulated
year: 01/01/2000 00:00 to 12/31/2000 23:55 at a 5 minute step, 105,120
values per gauge. The source records run to 2009 and are not included
here, since neither `.inp` reads beyond 2000.

**29 February 2000 is absent from all four gauge records**, although 2000
is a leap year. This is a property of the source data, not of the model,
and SWMM reads the same files.

The two scenarios measure discharge to the receiving water differently,
because the models book it differently. Scenario 1 has overflow weirs
from the tanks and CSO structures into the surface water body, so the
discharge appears as weir flow and the nodes almost never report
flooding. Scenario 2 has no such weirs and the same quantity is reported
as node flooding. `paper_objectives.py` accounts for this; the two
numbers are comparable.

## Control structure in brief

**Scenario 1.** Each swale has three outlets. Infiltration to the
groundwater and the 0.30 m overflow weir to the sewer are never
controlled; the underdrain to the river is. A second actuator, two
orifices between groundwater and river, is reset every six hours in
proportion to how far the water table is from its target. The couplings:
EFD counts the expected swale spill, the swale release threshold moves
with the groundwater level, and both the underdrains and the exchange
orifices hold back while the sewer is already discharging into the river.

**Scenario 2.** Each roof has a controlled drain to the sewer and an
irrigation offtake. The roof follows a five-question cascade, first match
winning: release if nearly full; hold if its own downstream path is at
risk or EFD has flagged its tank as the fullest; release ahead of a storm
that will not fit; release in dry weather above the irrigation reserve;
otherwise hold. EFD counts the water stored on the roofs upstream of each
tank, and runs first in each step so the roofs read the current verdict.

In both scenarios the upstream storage is added to the **numerator** of
the filling degree only, never to the denominator. The structures are
unevenly distributed between tanks, so dividing by a combined capacity
would put each tank on a different scale and equal filling degree would
stop comparing like with like.

## License and citation

The code is MIT licensed, see `LICENSE`. The Astlingen benchmark model
and the Erft rainfall records come from the benchmark's own publication
and carry its terms, not this license.

`CITATION.cff` drives the "Cite this repository" button.
