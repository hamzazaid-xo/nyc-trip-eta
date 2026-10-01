# Phase 0 — The container

## Where we stood
One commit (CLAUDE.md), a remote, nothing runnable. Host: macOS on arm64 (aarch64), 32 GB RAM,
Docker Desktop with 10 CPUs and **16 GB allocated to the Docker VM** — worth remembering, because the
container, not the host, is what the data pipeline has to fit inside.

## The question this phase answered
Can this project be reproduced on another machine with one command and zero host installs?

## Concepts
**Reproducibility is a correctness property.** With the environment pinned first, any later surprise
is about the data or the model — never about the laptop. That removes a whole category of debugging.

**A lockfile is not the same as pinned versions.** `pyproject.toml` pins the 20 packages we chose;
`uv.lock` pins all 185 that actually get installed, including transitive ones, per platform. Only the
lockfile makes a build byte-identical later.

**Dev and production are different artifacts.** The dev image needs jupyter, SHAP, geopandas,
matplotlib. The serving image needs a model and FastAPI. Separating them now means Phase 6 has nothing
to retrofit — and the production image never carries a notebook.

**Layer caching rewards ordering.** The dependency stages copy *only* `pyproject.toml` and `uv.lock`
before installing. Editing `src/` therefore never reinstalls a package.

## Decisions and why
| Decision | Why |
|---|---|
| Multi-stage `Dockerfile` (base → deps → dev → runtime) | Gives the dev/prod split for free and makes caching explicit |
| venv at `/opt/venv`, **outside** the `/app` bind mount | A host `.venv` can never shadow it, and a macOS/arm venv can never leak into Linux. This is the single most common Docker+Python bug |
| Dependency groups: `[project]` = runtime, `train` = heavy analysis, `dev` = tooling | The group split *is* the image split. `--no-default-groups` builds the lean serving image |
| `PYTHONPATH=/app/src`, project not installed | Imports work identically in every stage with no editable-install/bind-mount interaction to reason about |
| Tag pin `python:3.14.7-slim-trixie`, digest pin deferred | Readable now; Phase 7 CI is where a digest actually pays off |
| `runtime` stage non-root (uid 10001); `dev` stage root | Non-root is a real boundary in production. In dev it buys nothing, because macOS virtiofs handles mount ownership |
| Native `arm64` build | Fast locally. A `linux/amd64` buildx pass is a Phase 6 concern if this ever deploys |
| `models/` bind-mounted read-only into `api` | `models/` is gitignored and dockerignored, so it cannot be baked in yet. Phase 6 decides how the model actually ships |

## The bugs we hit (worth remembering)
`make build` builds `dev` and `api` concurrently. Both stages mount the same BuildKit cache at
`/root/.cache/uv`, so both tried to download `pyarrow` at once and one died:

```
error: Failed to download `pyarrow==25.0.1`
  cause: Could not acquire lock
  cause: Timeout (300s) when waiting for lock on .../pyarrow/...lock
```

BuildKit cache mounts default to `sharing=shared`, which permits concurrent access — and then uv's own
per-wheel file lock is what breaks. The fix is `sharing=locked`, which makes BuildKit serialize access
to the cache mount across parallel builds, keeping the cache shared without the race.

### 2. `ModuleNotFoundError: No module named 'api'`
`PYTHONPATH` was `/app/src`, but `api/` sits at the repo root, not under `src/`. Fixed to
`/app/src:/app`. Lesson: a src-layout project with a sibling package has *two* import roots.

### 3. ruff flagged `from api.main import app` as misordered
ruff's `src` setting lists import **roots**, not first-party directories. Listing `"api"` told ruff
that `main` is first-party — which left `api.main` looking like a third-party package. The root is the
repo itself, so `src = ["src", "."]`.

### 4. `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead`
Starlette moved to the httpx 2.x line. Swapped `httpx==0.28.1` for `httpx2==2.13.1`. This is the
argument for building immediately after pinning: a version table alone cannot catch a compatibility
message that only appears at import time.

### 5. `OSError: libgomp.so.1: cannot open shared object file`
`import lightgbm` failed. LightGBM publishes a `py3`-universal wheel that *dynamically links* the
system OpenMP runtime instead of bundling it, and `slim` images do not ship `libgomp1`. One
`apt-get install --no-install-recommends libgomp1` in the **base** stage, so dev and runtime both
inherit it. Lesson: a wheel existing for your Python version does not mean it is self-contained.

## Verified
```
make build      tripeta-dev 2.43 GB, tripeta-api 1.06 GB
make test       16 passed
make lint       All checks passed! / 12 files already formatted
production image:
  GET /health -> {"status":"ok","model_version":null}   (ready in 2s)
  running as   -> uid=10001(appuser)                    (non-root confirmed)
  import geopandas -> ModuleNotFoundError               (dev/prod split is real, 1.37 GB lighter)
```

## What exists now
```
Dockerfile      4 stages
compose.yaml    dev | notebook | mlflow | api | test
Makefile        the only interface
pyproject.toml  20 direct pins + ruff/pytest config
uv.lock         185 packages, resolved for Linux/cp314
src/tripeta/    data, features, train, evaluate, predict  (docstring stubs)
api/main.py     GET /health only — enough to prove the production image runs
tests/          dependency-import smoke tests + a /health test
```

## What this sets up
Phase 1 can now assume a trustworthy environment and spend all its attention on the data: what one
honest row looks like, and how many rows we throw away to get there.
