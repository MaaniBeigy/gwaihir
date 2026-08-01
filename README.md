# gamebus-coach

`coach-api` is the GameBus 2026 Coach service: a single FastAPI app
(`src.agents.app:app`) on port `8003` that runs a habit interview,
standardizes routines, schedules campaign challenges around the user's
calendar, and reports phase events to the gateway over mTLS.

Two compose files ship with the repo:

- [`docker-compose.yml`](docker-compose.yml) — local dev. Defines two
  services: `coach-api` (plain HTTP on `:8003`, embedded Chroma, mTLS off)
  and `coach-dev` (under the `dev` profile — builds the `dev` stage of
  the Dockerfile and adds pytest/black/isort/ruff on top).
- [`deploy/docker-compose.yml`](deploy/docker-compose.yml) — production.
  nginx mTLS terminator + GPU `coach-api` + separate `chroma-db`.

[`Dockerfile.coach`](Dockerfile.coach) is multi-stage; the dev image
shares the runtime layer cache, so you never install Python tooling on
the host.

## Setup

```sh
# 1. Configure env:
cp .env.example .env
# edit .env — set OPENROUTER_API_KEY and COACH_INTERNAL_TOKEN at minimum

# 2. Build both images (runtime and dev). First time is ~5 minutes:
docker compose build coach-api
docker compose --profile dev build coach-dev

# 2b. Force a clean rebuild after large changes, file renames, or dep bumps.
#     Run this whenever you suspect a stale layer (e.g. logs only show the
#     CUDA banner, container restart-loops, ImportError on a renamed module):
docker compose down -v
docker compose build --no-cache coach-api
docker compose --profile dev build --no-cache coach-dev

# 3. Run the service in the background:
docker compose up -d coach-api
# → http://localhost:8003/health
# Tail logs in a separate terminal (or skip):
docker compose logs -f coach-api

# 4. Run the test suite (any of these):
docker compose run --rm coach-dev pytest
docker compose run --rm coach-dev pytest -m unit
docker compose run --rm coach-dev pytest -m integration
docker compose run --rm coach-dev pytest tests/unit/test_collections.py

# 5. Coverage:
docker compose run --rm coach-dev pytest --cov --cov-report=term-missing --cov-report=html
# HTML report lands at ./htmlcov/index.html (bind-mounted from the container).
docker compose run --rm coach-dev pytest --cov --cov-fail-under=90

# 6. Format and lint:
docker compose run --rm coach-dev sh -c "isort src tests && black src tests"
docker compose run --rm coach-dev ruff check src tests

# 7. Smoke-test /chat:
./scripts/smoke_chat.sh

# 8. Tear down:
docker compose down -v

# 9. Production deploy on the GPU host:
docker volume create coach-chroma
docker volume create coach-models
docker compose -f deploy/docker-compose.yml up -d
```

## GPU support

`coach-api` runs on `nvidia/cuda:12.4.0-runtime-ubuntu22.04` and the embedding
model loads on CUDA. To actually expose your GPU to the container you need
two things:

1. **NVIDIA driver on the host** (already true if you have an NVIDIA GPU
   on Windows + WSL2).
2. **NVIDIA Container Toolkit inside your WSL2 distro.** One-time install:
   ```sh
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
       | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
       | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo service docker restart
   ```

Verify the GPU is visible from inside Docker:

```sh
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# expected: a table with your GPU name and driver version
```

`docker-compose.yml` already declares the `nvidia` device reservation on
`coach-api` and `coach-dev`, so once the toolkit is installed the GPU is
attached automatically. If you don't have an NVIDIA GPU, comment out the
`deploy.resources.reservations.devices` block on those services — the
service will fall back to CPU embeddings (slow but functional).

## HTTP contract

```
POST /chat
  body: {
    playerId, campaignId, sessionId, message,
    llmKey?,                            # per-request override, optional
    context: {
      challenges:          [...],       # campaign GameBus tasks
      scheduledActivities: [...],       # all SCHEDULE_ACTIVITY rows in window
      schedulingWindow: {
        scheduleStartDate: "YYYY-MM-DD",
        scheduleEndDate:   "YYYY-MM-DD",
        schedulingPeriodWeeks: int,
        currentDate: "YYYY-MM-DD",
        currentDayname: "mon|tue|...|sun"
      }
    }
  }
  resp: {
    output: "...",
    interviewComplete: bool,
    plannedActivities: [{
      gameDescriptorTK: "SCHEDULE_ACTIVITY",
      activityType: "...",
      startDate: "ISO",
      endDate: "ISO",
      allDay: false,
      isHealthActivity: bool,
      challengeRuleId: int|null
    }]            # only when interviewComplete=true
  }

POST /reset       body: {playerId, campaignId, sessionId}
GET  /health                             # 503 while embedding model loads, 200 once ready
GET  /metrics                            # Prometheus exposition
POST /admin/flush?older_than_days=N      # internal; same auth as /chat
```

In production every request arrives via the nginx mTLS terminator and
coach-api validates `X-Client-CN == COACH_EXPECTED_PEER_CN` and
`X-Coach-Internal-Token == COACH_INTERNAL_TOKEN`. In dev
(`COACH_REQUIRE_MTLS=false`) those checks are skipped.

## Environment variables

See [`.env.example`](.env.example) for the canonical list.

| Name | Default | Purpose |
| ---- | ------- | ------- |
| `COACH_REQUIRE_MTLS` | `false` | Toggle X-Client-CN / token enforcement. Production: `true`. |
| `COACH_INTERNAL_TOKEN` | `dev-only-change-me` | Shared secret for inbound and outbound coach internal traffic. |
| `COACH_EXPECTED_PEER_CN` | `gamebus-api-v2` | CN nginx is required to set on `X-Client-CN`. |
| `OPENROUTER_API_KEY` | _(required)_ | LLM key for all four agents. Per-request `llmKey` overrides this. |
| `INTERVIEWER_MODEL` / `STANDARDIZER_MODEL` / `SCHEDULER_MODEL` / `CONTROLLER_MODEL` | _(per-agent defaults)_ | Optional per-agent model overrides. |
| `SCHEDULING_PERIOD_WEEKS` | `1` | Default scheduling horizon. |
| `COACH_MAX_SCHEDULE_ITERATIONS` | `3` | Max scheduler-controller rounds before the latest plan is accepted. |
| `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_PATH` | _(see env example)_ | Empty `CHROMA_HOST` → embedded persistent client at `CHROMA_PATH`. Production sets `CHROMA_HOST=chroma-db`. |
| `COACH_MEMORY_RETENTION_DAYS` | `7` | Periodic flush horizon. |
| `HF_HOME` | `/models` | HuggingFace cache directory; backed by the `coach-models` volume. |
| `COACH_WEBHOOK_URL` | _(unset in dev)_ | Gateway phase-event endpoint. |
| `COACH_WEBHOOK_CLIENT_CERT` / `_KEY` / `COACH_WEBHOOK_SERVER_CA` | _(production)_ | mTLS material for the outbound webhook. |

## Internals

- **Per-(playerId, campaignId) memory.** Every Chroma read/write goes
  through [`collection_for(player_id, campaign_id)`](src/memory/collections.py),
  which returns `coach_p{playerId}_c{campaignId}`. There is no global
  collection.
- **Periodic flush.** [`src/memory/flush.py`](src/memory/flush.py) deletes
  vectors older than `COACH_MEMORY_RETENTION_DAYS` every 24 h and exposes
  `coach_memory_flush_vectors_total{status}` on `/metrics`.
- **Embedding model.** [`src/memory/embeddings.py`](src/memory/embeddings.py)
  loads `BAAI/bge-large-en-v1.5` on CUDA at startup (CPU fallback). The
  `coach-models` volume keeps the weights warm across restarts.
- **Phase webhook.** [`src/webhook/client.py`](src/webhook/client.py) posts
  `INTERVIEWING` / `SCHEDULING` / `COMPLETE` / `ERROR` events to
  `${COACH_WEBHOOK_URL}` over mTLS, with three retries at 1 s / 4 s / 15 s.
  In dev the client is a no-op.

## Layout

```
gamebus-coach/
├── app/
│   ├── requirements.txt          # runtime deps
│   └── requirements-dev.txt      # tests + lint + format (built into the dev stage)
├── Dockerfile.coach              # multi-stage: runtime (default) and dev
├── src/
│   ├── Agents/
│   │   ├── app.py                # FastAPI: /chat /reset /health /admin/flush /metrics
│   │   ├── coach_graph.py         # LangGraph flow: interview + scheduler-controller loop
│   │   ├── interviewer_core.py
│   │   ├── standardizer_core.py
│   │   ├── controller_core.py
│   │   ├── scheduler_core.py
│   │   ├── ChromaDB/chroma_db.py
│   │   └── Tools/
│   │       ├── conversation_parser.py
│   │       ├── gamebus_task_extractor.py
│   │       └── task_normalizer.py
│   ├── Memory/
│   │   ├── collections.py        # collection_for(player_id, campaign_id)
│   │   ├── embeddings.py         # BGE GPU embedding function
│   │   ├── flush.py              # APScheduler + Prometheus counter
│   │   └── Templates/            # agent prompt templates
│   └── Webhook/client.py         # mTLS phase-event client
├── deploy/
│   ├── docker-compose.yml        # production stack (nginx + coach-api + chroma-db)
│   ├── nginx/nginx.conf          # mTLS terminator
│   ├── certs/                    # gitignored
│   └── coach-certs/              # gitignored
├── docker-compose.yml            # local dev: coach-api + coach-dev (under `dev` profile)
├── pyproject.toml                # black / isort / coverage config
├── pytest.ini                    # test discovery + markers
├── scripts/smoke_chat.sh         # one-shot /chat smoke test
├── tests/
│   ├── unit/                     # offline; mocked LLM, Chroma, embeddings, webhook
│   └── integration/              # FastAPI end-to-end via TestClient
└── .env.example
```
