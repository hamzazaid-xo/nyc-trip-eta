# Project: NYC Trip Duration Prediction (classic ML portfolio project)

## Why this project exists
Public portfolio project for Hamza, a senior engineer whose production AI work (LLM agents, RAG,
self-hosted models) is under NDA. This repo must prove **classic ML judgement**: clean data handling,
no leakage, honest validation, strong baselines, explainability, and a production-style serving path.
The audience is hiring managers and AI/ML interviewers. Quality and clarity beat model complexity.

Framing: "Predict trip duration at booking time", the same ETA problem a dispatch system
(e.g. non-emergency medical transport) has to solve before a driver is assigned.

**Second goal, equally important: this is a learning project.** Hamza is learning classic ML concepts
by building this. Code that appears without explanation is a failure even if it runs.

---

## Teaching protocol (read this before every phase)

Work as a **train of thought**, not a code dump. Every phase follows the same five beats, and the
first three happen in chat *before any file is written*:

1. **Recap — where we are.** Two or three sentences: what the previous phase produced, what we now
   know that we did not know before, and what is still unanswered.
2. **Why this step now.** The specific question this phase answers. If the phase could be skipped,
   say what would break later.
3. **Options and the choice.** The two or three realistic ways to do this step, the tradeoff between
   them, and a recommendation with the reason. Wait for approval here.
4. **Build.** One module at a time, smallest useful piece first. After each module, a short note:
   what it does, the one line worth staring at, and how to see it work (a command to run).
5. **Close the loop.** What the output shows, what surprised us, and the sentence that sets up the
   next phase. Append this to `docs/notes/phase-N.md` — these notes become the raw material for the
   README in Phase 8.

Further rules for the teaching side:

- Introduce vocabulary explicitly the first time it appears (leakage, target encoding, early
  stopping, quantile metric, SHAP value). One or two sentences, in plain language, tied to this data.
- When a number comes out of a script, interpret it. "MAE 4.3 min" means little; "half our
  predictions are within ~3 min, but rush-hour Manhattan trips are off by 8" is the lesson.
- Prefer showing a wrong/naive version first when the contrast is the lesson (this is exactly why
  the leakage experiment in the README matters).
- Never jump ahead. If a later phase's idea comes up early, name it and park it.
- When Hamza asks "why", answer with the concept, not just the API.

### Concept arc (the curriculum hiding inside the phases)
| Phase | Core concepts |
|---|---|
| 0 | Reproducible environments, containers, dependency pinning, multi-stage builds |
| 1 | Data contracts, cleaning as documented decisions, train/val/test splits over **time** |
| 2 | Distributions, skew and log transforms, reading a feature's signal off a plot |
| 3 | Baselines, group statistics, fallback hierarchies, choosing metrics that match the cost of error |
| 4 | Gradient boosting, categorical handling, early stopping, overfitting, error slicing |
| 5 | Model explanation vs. feature importance, local vs. global attribution |
| 6 | Model serialization, versioning, input validation, serving latency |
| 7 | Testing ML code, guard tests, CI |
| 8 | Writing up results honestly; the limitations section as a signal of seniority |

---

## Working conventions for Claude Code
- Follow the teaching protocol above. Propose, wait for approval, then build.
- Work phase by phase. Small, focused commits with clear messages.
- **Everything runs in Docker.** Never install or run Python, uv, pip, jupyter, or mlflow on the
  host. Every command goes through `make`, which wraps `docker compose`. If a step seems to need a
  host tool, say so instead of silently installing it.
- Ask before downloading more than ~2 GB of data or adding a dependency not listed in the Stack table.
- Never commit data, model artifacts, or MLflow runs. Keep them in gitignored folders.
- Fix random seeds. Every result in the README must be reproducible from a single command.
- Host machine for this project: **32 GB RAM, ~39 GB free disk** — so use the full four months, no
  downsampling. If that ever changes, downsample (20% per month) and say so in the README.
- Prefer simple, readable code over clever code. Type hints everywhere. Docstrings on public functions.

---

## Environment: everything in Docker

One image, multiple stages. One compose file, four services.

```
Dockerfile          base -> deps -> dev -> runtime   (multi-stage)
compose.yaml        dev | mlflow | api | test
Makefile            the only interface Hamza types
```

- **base** — `python:3.14.7-slim-trixie`, with `uv` copied in from the official uv image
  (`COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/`). No compilers needed: every pinned
  dependency has a Python 3.14 wheel (verified, see Stack).
- **deps** — `uv sync --frozen` from `pyproject.toml` + `uv.lock`. Cached layer, so editing source
  never reinstalls packages.
- **dev** — adds dev dependencies (pytest, ruff, jupyterlab). Repo bind-mounted at `/app`, so edits
  on the host are live in the container. This is where data prep, training and the notebook run.
- **runtime** — production serving image. Only runtime deps, no notebook, no training code,
  non-root user, the trained model copied in from `models/`. This image is what Phase 6 ships.

Compose services:

| Service | Purpose | Port |
|---|---|---|
| `dev` | shell, data prep, training, jupyter | 8888 (jupyter) |
| `mlflow` | MLflow tracking UI, backed by `./mlruns` | 5000 |
| `api` | the `runtime` image serving FastAPI | 8000 |
| `test` | one-shot pytest + ruff, no data needed | — |

Makefile targets (the whole interface):

```
make build        # build dev + runtime images
make shell        # bash inside the dev container
make download     # fetch raw TLC data into data/raw/
make data         # clean + split -> data/processed/ + cleaning report
make notebook     # jupyterlab at localhost:8888
make mlflow       # MLflow UI at localhost:5000
make baselines    # Phase 3
make train        # Phase 4
make explain      # Phase 5 (SHAP images)
make api          # build + run runtime image at localhost:8000
make test         # pytest
make lint         # ruff check + format --check
make down         # stop everything
```

---

## Stack (pinned; versions verified against PyPI / Docker Hub on 2026-10-01)

| Thing | Version | Note |
|---|---|---|
| Python | **3.14.7** (`python:3.14.7-slim-trixie`) | 3.14 is the current stable line, security support to 2030-10. 3.14.8 is the newest source release; the slim image published is .7 — pin the image, bump when it lands. |
| uv | 0.12.21 | env + lockfile, inside the container only |
| polars | 1.44.2 | **primary dataframe** for loading, cleaning, feature building |
| pandas | 3.0.6 | only at the scikit-learn / SHAP / MLflow boundary, where pandas is the expected input |
| pyarrow | 25.0.1 | Parquet I/O |
| scikit-learn | 1.9.1 | baselines, metrics, preprocessing |
| lightgbm | 4.7.0 | the main model |
| shap | 0.52.0 | explainability (requires Python ≥3.12 — satisfied) |
| mlflow | 3.16.1 | local file-backed tracking in `./mlruns` |
| fastapi | 0.142.2 | serving |
| pydantic | 2.13.5 | request/response validation |
| uvicorn | 0.54.0 | ASGI server |
| pytest | 9.1.1 | tests |
| httpx | 0.28.1 | API tests against the FastAPI app |
| ruff | 0.16.9 | lint + format |
| geopandas | 1.2.0 | read the taxi zone shapefile once, compute zone centroids |
| pyogrio | 0.13.0 | geopandas' shapefile reader (no GDAL system install needed) |
| shapely | 2.1.2 | geometry for centroids |
| matplotlib | 3.11.2 | plots |

**Decision — polars for the pipeline, pandas at the edges.** Polars keeps the cleaning and feature
code explicit and fast; pandas appears only where a library demands it. Rationale goes in the README
(reviewers notice this kind of deliberate boundary). Say the word if you'd rather be pandas-only for
reviewer familiarity.

**Decision — geo work is a one-time, build-time step.** `geopandas`/`shapely` run once in Phase 1 to
turn the shapefile into `zone_centroids.parquet`. The `runtime` image never imports them, so the
serving image stays small. Nothing paid, nothing cloud.

---

## Data
- NYC TLC Yellow Taxi trip records (monthly Parquet files), from the TLC trip record data page:
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  (Verify the current download URLs from that page; do not guess.)
- Taxi zone lookup CSV and taxi zone shapefile (same page), used for borough names and zone centroids.
- Use four consecutive months of one recent year: months 1–2 train, month 3 validation, month 4 test.
- Expect ~50–80 MB per monthly file, so ~300 MB total — well under the ask-first threshold.
- Write a `scripts/download_data.py` that fetches exactly these files into `data/raw/`.

## The leakage rule (central to the project)
Only use information available **at booking time**. Allowed: pickup datetime, pickup zone, dropoff zone,
passenger count, vendor, and anything derived from them (hour, weekday, holiday flag, zone centroids,
straight-line distance between zone centroids, borough pair).
Forbidden as features: trip_distance (measured after the trip), fare, tips, tolls, total_amount,
payment type, dropoff time. Document this decision prominently in the README, including a short
experiment showing how much "better" the model looks when trip_distance leaks in, and why that
number is fake.

---

## Phases

### Phase 0 — The container
**Where we stand:** empty repo, a plan, nothing runnable.
**The question:** can anyone — including future-you on another machine — reproduce this project with
one command and zero host installs?
**Concepts:** why reproducibility is a correctness property, not a nicety; layer caching; why dev and
production images should differ; lockfiles vs. loose version ranges.
**Build:** `pyproject.toml` + `uv.lock` with the pins above, multi-stage `Dockerfile`, `compose.yaml`,
`Makefile`, `.gitignore` (`data/ models/ mlruns/ .venv/`), ruff config, repo skeleton:
```
src/tripeta/      data.py, features.py, train.py, evaluate.py, predict.py
api/              FastAPI app
scripts/          download_data.py
notebooks/        01_eda.ipynb only (exploration, not pipeline logic)
tests/
docs/notes/       phase-by-phase learning notes
data/ models/ mlruns/   (gitignored)
```
**Done when:** `make build && make shell` drops you into a container where `python -c "import
polars, lightgbm, shap, geopandas"` succeeds, and `make test` passes with one trivial test.
**Why Phase 1 follows:** with a trustworthy environment, any surprise from here on is about the data
or the model — never about the setup.

### Phase 1 — Data and cleaning
**Where we stand:** a working container, no data.
**The question:** what does one honest row of training data look like, and how many rows do we throw
away to get there?
**Concepts:** data contracts; cleaning as a series of *documented decisions* with a row count each;
why the split must be chronological (months 1–2 / 3 / 4) and not random.
**Build:** `scripts/download_data.py`; `src/tripeta/data.py` with cleaning rules, each reporting how
many rows it removes:
- duration between 1 and 180 minutes
- unknown zones dropped (**verify the actual unknown/N-A zone IDs from the lookup CSV — do not
  hardcode 264/265 on assumption; 263 is a real zone**)
- pickup datetime inside the file's own month
- implausible average speeds removed, using straight-line zone-centroid distance, **not** trip_distance
Plus the one-time shapefile → `zone_centroids.parquet` step.
**Done when:** `make data` produces cleaned train/val/test Parquet files and a cleaning report
showing the row count after each rule.
**Why Phase 2 follows:** we have rows, but no intuition. Next we look before we model.

### Phase 2 — EDA notebook
**Where we stand:** clean, split data; no understanding of its shape.
**The question:** what actually drives trip duration, and which features are worth engineering?
**Concepts:** right-skewed targets and why `log(duration)` shows up later; reading signal strength
off a plot; the difference between a pattern and a usable feature.
**Build:** `notebooks/01_eda.ipynb` — duration distribution (raw and log), duration by hour and
weekday, top zone pairs, borough patterns.
**Done when:** the notebook tells a short story in markdown cells and explicitly motivates each
feature Phase 3 will build.
**Why Phase 3 follows:** we now have a feature list and, more importantly, a sense of what "good"
should look like — which is only meaningful against a baseline.

### Phase 3 — Baselines
**Where we stand:** features chosen, no model, no idea what score is respectable.
**The question:** how well can we do with almost no machine learning? That number is the bar.
**Concepts:** why a baseline is the most important number in the project; group statistics and
fallback hierarchies; picking metrics that match the cost of being wrong (a dispatcher cares about
p90, not just the mean).
**Build:**
- Baseline A: global median duration
- Baseline B: median per (pickup zone, dropoff zone), falling back to borough pair, then global
- Baseline C: linear regression on the engineered features
- Validation metrics: MAE (minutes), RMSE, median absolute error, p90 absolute error
**Done when:** results logged to MLflow and a comparison table generated by a script, not by hand.
**Why Phase 4 follows:** now a gradient-boosted model has something to beat, and "beating it" has a
number attached.

### Phase 4 — LightGBM
**Where we stand:** a baseline bar and a metric set.
**The question:** does a real model earn its complexity, and where does it still fail?
**Concepts:** gradient boosting in one paragraph; training on `log(duration)` and reporting in
minutes; native categorical handling for zones; early stopping as overfitting control; error slicing
as the actual skill.
**Build:** training on log target, early stopping on the validation month, a modest and documented
hyperparameter search (not a giant sweep), then error analysis by hour, by borough pair, by trip
length bucket. Final numbers reported **once** on the untouched test month.
**Done when:** LightGBM clearly beats Baseline B — or the README honestly explains why it doesn't.
**Why Phase 5 follows:** a model that wins but can't be explained is unusable in dispatch.

### Phase 5 — Explainability
**Where we stand:** a model with known strengths and known failure slices.
**The question:** what is the model actually using, and would a dispatcher believe it?
**Concepts:** SHAP values vs. built-in feature importance; global summary vs. a single prediction's
explanation.
**Build:** SHAP summary plot plus two or three single-prediction explanations, saved as images for
the README.
**Done when:** the plots are saved and you can narrate one prediction out loud, end to end.
**Why Phase 6 follows:** it's a model you trust — time to make it callable.

### Phase 6 — Serving
**Where we stand:** a trusted, explained model sitting in `models/`.
**The question:** what does it take to answer a dispatcher's request in milliseconds?
**Concepts:** model serialization and versioning; validating input at the boundary; loading once at
startup instead of per request; why the production image is not the dev image.
**Build:** FastAPI `POST /predict` — input pickup datetime, pickup zone, dropoff zone, passenger
count; output predicted minutes plus model version. Pydantic validation with clear 422s for unknown
zones. `GET /health`. Model loaded once at startup. The `runtime` Docker stage plus a `make api`
target.
**Done when:** `make api` serves predictions and a `curl` example in the README works.
**Why Phase 7 follows:** it works on your machine; now make it stay working.

### Phase 7 — Quality and CI
**Where we stand:** a working service, no safety net.
**The question:** what would catch it if a future change silently reintroduced leakage?
**Concepts:** what is worth testing in ML code (the deterministic parts); the guard test as encoded
judgement; CI as the thing that runs your standards when you forget to.
**Build:** pytest for feature functions, cleaning rules, a **leakage guard** test asserting forbidden
columns never reach the model, and API tests against a tiny fixture model. Must run in under a minute
with no real data. ruff lint + format. GitHub Actions running lint and tests on every push, inside
the same Docker image.
**Done when:** `make test` and CI both pass from a clean clone with no data present.
**Why Phase 8 follows:** the work is done; the work is not yet *legible*.

### Phase 8 — README as a design doc
**Where we stand:** a complete, tested, served, explained model — and eight phase notes in `docs/notes/`.
**The question:** can a hiring manager understand the judgement in this repo in two minutes?
**Concepts:** writing results honestly; why the limitations section is the strongest seniority signal
in a portfolio repo.
**Build:** sections — problem and framing, data and cleaning, the leakage decision (with the fake-win
experiment), validation strategy, results table (all baselines vs. LightGBM on the test month), error
analysis, SHAP findings, serving architecture diagram (Mermaid), how to reproduce, limitations and
next steps (live traffic signals, demand forecasting per zone, drift monitoring).
**Done when:** it's scannable, every number traces to a command, and nothing is oversold.

---

## Out of scope (for now)
Deep learning, external weather/traffic APIs, cloud deployment. Mention as next steps only.
