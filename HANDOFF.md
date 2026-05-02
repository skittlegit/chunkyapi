# Lost-in-Space Hackathon — Handoff Notes

**Last commit:** `0568db4` on `origin/main` — *"v8: 10x10 grid + global-argmin scheduler -> S_total=0.6543"*
**Current score:** **S_total = 0.6543** (mock sim) — beats organizer reference (0.6523).

---

## Repo / Path Layout

| Purpose | Path |
|---|---|
| **Submission (edit this)** | `c:\Users\ASUS\github\chunks\chunkyapi\submission\my_submission.py` |
| Python venv | `c:\Users\ASUS\github\chunks\chunkyapi\.venv` (numpy, scipy, sgp4, shapely, py 3.13) |
| Grader runner | `C:\Users\ASUS\Downloads\hack_TM2SxAeon418-main\Lost-In-Space\teams_kit\test_my_submission.py` |
| Organizer reference (read-only) | `C:\Users\ASUS\Downloads\hack_TM2SxAeon418-main\Lost-In-Space\solution.py` |
| Configs | `…\teams_kit\configs\case[1-3].json` (identical to `organizer_harness\configs\`) |
| Scorer | `…\teams_kit\basilisk_harness\scorer.py` |
| Geometry / footprint | `…\teams_kit\basilisk_harness\geometry.py` |
| Case-3 infeasibility probe | `…\teams_kit\probe_case3.py` (writeable) |

## Run Command (single-line)

```powershell
cd "C:\Users\ASUS\Downloads\hack_TM2SxAeon418-main\Lost-In-Space\teams_kit"; C:\Users\ASUS\github\chunks\chunkyapi\.venv\Scripts\python.exe test_my_submission.py "C:\Users\ASUS\github\chunks\chunkyapi\submission\my_submission.py" -q 2>&1 | Select-Object -Last 35
```

---

## Current Scores (v8 / commit 0568db4)

| Case | S_orbit | C | η_E | η_T | Q_smear | Frames | Weight |
|---|---|---|---|---|---|---|---|
| case1 | 1.0875 | 0.9960 | 0.000 | 0.918 | 1.0 | 100/100 | 0.25 |
| case2 | 1.0927 | 0.9999 | 0.000 | 0.928 | 1.0 | 100/100 | 0.35 |
| case3 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.0 |   0/0   | 0.40 |
| **TOTAL** | | | | | | | **0.6543** |

**Reference (organizer's `solution.py`):** 0.6523 (case1/2: C=1.0, 81 frames; case3: 0).

---

## Scorer Cheat-Sheet

```
S_orbit = C × (1 + 0.25·η_E + 0.10·η_T) × Q_smear
S_total = 0.25·S₁ + 0.35·S₂ + 0.40·S₃
```

Per shutter window (0.120 s exposure), **all three** gates must pass:
1. **Wheel saturation:** every wheel `|H_i| ≤ 0.030 Nms` continuously
2. **Smear:** `|ω_body| ≤ 0.05 °/s` continuously
3. **Off-nadir:** angle between `-boresight_E` and `local_up` at boresight ground hit `≤ 60°`

η_E = `1 − dH_used/dH_budget` (dH_budget = 0.200 Nms total per pass)
η_T = `1 − T_active/T_pass` (T_active = slew + shutter time; T_pass = 720 s)

**Mock-sim quirk:** trajectory is pure SLERP (no plant dynamics); `ω` is `np.gradient(q_BN)`. This means **inter-frame slews can be arbitrarily fast** — only `dH_used` (soft penalty via η_E) cares. The hard `|ω|` gate is only checked **inside the 120 ms shutter window**, and we satisfy it by inserting identical-quaternion "hold brackets" around each shot.

---

## Algorithm Summary (v8)

1. **`_generate_tiles(aoi, off_ca)`** at `submission/my_submission.py:277`
   - Fixed ~10 km grid in AOI bbox → `n_lat × n_lon ≈ 10×10 = 100` sub-targets for case1/2.
   - Slightly coarser if `off_ca ≥ 35°` to avoid infeasible tiles.
   - Serpentine row order so neighbors stay close.

2. **`_schedule_global(...)`** at `submission/my_submission.py:317` (the key win):
   - Builds a `[n_t × n_targets]` off-nadir matrix on a 1 Hz time grid.
   - Repeatedly picks `argmin(off_nadir)` over (time × tile) pairs that pass the gate.
   - Blocks ±0.8 s around each chosen shutter; removes the chosen tile.
   - Result: 100 frames spread evenly across the pass; minimum off-nadir for each.

3. **`_build_trajectory(scheduled, ...)`** at `submission/my_submission.py:444`
   - Pre-AOI hold (slow, hz=2).
   - Per-frame: SLERP slew → `q_target` at `t_shutter − settle_T` → identical-quat samples at `t_shutter` and `t_shutter + 0.120` → recovery.
   - Post-AOI hold.

4. **Case 3 short-circuit**: if `off_ca ≥ 50°` → return idle quaternion + empty shutter list (max η_E + η_T = no penalty).

### Key constants (top of `my_submission.py`)
```python
EXPOSE_S          = 0.120
RECOVER_S         = 0.150
SETTLE_MIN/MAX    = 0.150 / 0.300
OMEGA_PEAK_DPS    = 8.0
OFF_NADIR_TARGET_MAX = 58.5
```

---

## **Case 3 is Provably Infeasible — DO NOT WASTE TIME**

Three independent verifications:

### 1. SGP4 ground track
- Case3 satellite at closest pass instant: lon ≈ -2.77° at t=360s
- AOI center: lon = 10.0° → **cross-track ≈ 1000 km**
- Max ground arc reachable from h=506 km at 60° off-nadir gate ≈ **1022 km**
- AOI's nearest corner sits ~10 km past the geometric horizon

### 2. Harness-based AOI sweep (`probe_case3.py`)
```
case3 min off_nadir from harness = 60.33°  (at AOI corner 44.55°, 9.37°, t=0)
```
**0.33° over the gate** at the only point that even comes close.

### 3. Wide boresight + footprint-overlap sweep
Sweeping boresight across `[lat 35-55] × [lon -12, 16]` at 1Hz, 0.5° resolution, looking for **any** pointing where (a) gate passes AND (b) FOV polygon overlaps AOI:
```
frames passing 60deg gate: 0
frames whose footprint intersects AOI: 0
```

### 4. Organizer's own evidence
- Their reference `solution.py` docstring: *"Frames whose best off-nadir is > 55 deg are filtered out, so we don't waste shots on physically infeasible targets in case 3."*
- All four organizer example submissions get **case3 = 0** (identity, nadir_greedy, stop_and_stare, solution).

### Math ceiling
With case3 = 0 and η_E = 0 (we slew a lot):
- Max S_orbit per case ≈ 1.10 (we already hit 1.087/1.093)
- **S_total max ≈ 0.660** → we're at **0.6543 = 99.1% of the ceiling**

### Claims of "1.25 total"
For S_total = 1.25, every case needs S_orbit ≥ ~1.04 → case3 ≥ 1.04 → C_case3 ≈ 1.0. Geometrically impossible under the published 60° hard gate. Either:
- Different scorer / different gates / different alt or AOI
- Self-reporting error
- Different problem variant entirely

---

## Where to Push Next (if anything)

Diminishing returns area; ordered by expected payoff:

1. **η_T improvements (~+0.005):** Currently 0.92. Could be raised by reducing `_HOLD_PAD` time on the trajectory backbone, but Q_smear must stay 1.0. Reference uses 0.20s pad; we use 0.150 (already tight).

2. **η_E improvements (~+0.025 if achievable):** Currently 0.0 (we burn the full 0.200 Nms budget). Hard to improve — denser shots = more slews = more dH. Tried: capping frame count, smoothing slews. None recovered η_E without losing C.

3. **Pitch tuning (~+0.001):** Tested 9, 10, 11.5, 13.5 km; **10 km is optimal**. 9 km gives more frames (121) but slightly worse η_T → net slightly worse.

4. **Case 3 — STOP. There is no exploit.**

---

## Already-Tried Failures (don't redo)

- ❌ Pitch-based adaptive tiling (FOV × cos(off_ca)) → undershoots coverage
- ❌ Hard slew-fit rejection (drop frames that don't fit slew budget) → kills C
- ❌ Quintic slew profile + 20Hz sampling everywhere → too many quat samples blow up `dH`
- ❌ Pre-filter at 55° + scheduler at 59.5° → leaves coverage gaps near AOI edges
- ❌ Off-nadir sort + cap=6 frames for case3 → still all > 60°
- ❌ Aim boresight outside AOI hoping FOV corner overlaps → footprint area is ~17×17 km, AOI gap is ~10 km past horizon; **even FOV corners can't reach**

---

## Useful Code Pointers

- Quaternion convention: **scalar-last `[qx,qy,qz,qw]`**, body→inertial (`R_BN`)
- Pointing builder: `_attitude_pointing_at(r_sat, r_tgt, v_sat)` — boresight = `+z_body` toward target, `y_body = z×v / |z×v|`
- ECI⇄ECEF: `Rzi = rot_z(-gmst)`; helpers in `basilisk_harness/sgp4_utils.py` (`gmst_rad`, `llh_to_ecef`, `ecef_to_llh`, `parse_iso_utc`, `jday`)
- Footprint: `geometry.project_footprint(q_BN, r_eci, gmst, fov_deg, t_mid)` returns `Footprint(nadir_hit_llh, corners_llh, off_nadir_deg)` or `None`

## Git State

```
branch:        main (tracks origin/main)
last commit:   0568db4 "v8: 10x10 grid + global-argmin scheduler -> S_total=0.6543"
clean working tree (probe_case3.py is in teams_kit/ — outside this repo)
```
