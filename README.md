# ChunkyAPI

FastAPI backend for the "Lost in Space" hackathon — satellite Earth observation
attitude planning and imaging scheduling.

## Scoring (current)

Against `app/data/test_cases.json` (contest weights `0.25 / 0.35 / 0.40`):

| Case | Coverage | η_E   | η_T   | Q_smear | S_orbit | Weight | Tiles imaged |
|------|----------|-------|-------|---------|---------|--------|--------------|
| 1    | 1.00     | 0.21  | 0.99  | 1.00    | 1.151   | 0.25   | 49 / 49      |
| 2    | 1.00     | 0.48  | 0.99  | 1.00    | 1.219   | 0.35   | 36 / 36      |
| 3    | 0.89     | 0.86  | 1.00  | 1.00    | 1.174   | 0.40   | 10 / 16      |

`S_total = 0.25·1.151 + 0.35·1.219 + 0.40·1.174 = 1.18`.

## Planner architecture

The planner (`app/core/planner.py`) is a two-pass scheduler designed around the
fact that the scoring `ΔH = I · ∫|ω| dt` charges for **every degree of slew**,
including a wind-up slew from nadir to the first tile and a return-to-nadir
slew at the end. Naïvely those two slews alone consume the entire 0.200 N·m·s
budget.

1. **Tile windows.** For each tile we compute its full *access interval*
   `[t_access_start, t_access_end]` — the time range during which the
   instantaneous off-nadir into that tile stays below the gate. Adaptive
   threshold ladder `(strict, +5°, +7°, +9°, +11°, hard_ceiling-1°)` lets a
   far-ground-track pass (case 3) image at, say, 55° even when the strict
   working limit is 50°.
2. **Pass A — pick shutter times.** Walk tiles in serpentine order; for each,
   pick the earliest legal shutter time inside its access window, accounting
   for slew time from the previous tile. Reject tiles that don't fit.
3. **Pass B — build attitude trajectory.** Hold the *first* tile's attitude
   from `t = 0` (no nadir-to-tile1 wind-up). Between consecutive tiles, hold
   the previous attitude until the latest moment, slew, settle, hold, expose.
   Hold the *last* tile's attitude until end of pass (no return-to-nadir).
4. **Footprints + scoring.** Per-shutter footprints feed the coverage
   integrator (sample-grid in equirect projection). `η_T` uses *active
   integration time* (sum of shutter durations), not the imaging window.
   `Q_smear` is sampled at five sub-intervals per shutter, not at a single
   midpoint, so a stray waypoint can't crater the score.

## Local development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs

## Endpoints

- `GET  /health`
- `GET  /api/cases`
- `GET  /api/cases/{case_id}`
- `GET  /api/cases/{case_id}/ephemeris`
- `POST /api/plan`
- `POST /api/simulate`
- `POST /api/validate`

## Deploy (Railway)

The repo includes `Dockerfile` and `railway.toml`. Set `ALLOWED_ORIGINS` env
var in the Railway dashboard if your frontend lives on a non-default origin.

## Tests

```powershell
pytest -q
```
