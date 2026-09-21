# Medical Appointment Agent

[![CI](https://github.com/italoo97/medical_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/italoo97/medical_agent/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C.svg)](https://www.langchain.com/langgraph)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#development)

A conversational agent that schedules, cancels, and looks up medical appointments on behalf of a patient, backed by **real Google Calendars** (one calendar per professional) instead of an in-memory or hardcoded roster.

The agent understands free-form natural language ("I want to see a cardiologist next Monday at 4pm"), but every decision that actually touches a calendar — availability, booking, cancellation — is made by plain, testable Python code. The LLM's only job is to translate between natural language and structured data; it never decides anything on its own.

## Contents

- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the agent](#running-the-agent)
- [API reference](#api-reference)
- [Development](#development)
- [Design principles](#design-principles)
- [Known limitations](#known-limitations)

## How it works

Every message is processed by a small [LangGraph](https://www.langchain.com/langgraph) state machine:

```mermaid
graph LR
    A[identify_intent] -->|schedule| B[scheduler]
    A -->|cancel| C[canceller]
    A -->|check| D[checker]
    A -->|unknown| E[message_generator]
    B --> E
    C --> E
    D --> E
    E --> F[END]
```

1. **`identify_intent`** — an LLM call (via OpenRouter) extracts structured data from the patient's message: intent (`schedule`, `cancel`, `check`, or `unknown`), professional name/specialty, patient name, date, time, and reason. The current professional roster is fed into this prompt so the model can resolve nicknames, typos, or partial names to the exact registered professional, instead of the application guessing.
2. **`scheduler` / `canceller` / `checker`** — plain Python nodes that validate the extracted data and talk to Google Calendar through `AppointmentService`. They never call the LLM. Cancelling doesn't require an exact date/time: if the patient doesn't remember it, the agent looks up their next upcoming appointment (scoped to the named professional, when given, so it never surfaces or touches another professional's appointment).
3. **`message_generator`** — a second LLM call turns the outcome (success, or a specific failure reason) into a short, friendly reply in the same language the patient used.

Professionals are **discovered dynamically**: `AppointmentService.list_professionals()` reads whichever Google Calendars have been shared with the service account, so adding a professional is an operational action (share a calendar + one API call — see [API reference](#api-reference)), never a code change or a redeploy.

## Tech stack

| Concern | Choice |
|---|---|
| Agent orchestration | [LangGraph](https://www.langchain.com/langgraph) |
| LLM provider | [OpenRouter](https://openrouter.ai/) (free-tier models by default) |
| LLM gateway | [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/) — optional, off by default |
| Calendar backend | [Google Calendar API](https://developers.google.com/calendar/api) via a service account |
| HTTP server | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Validation | [Pydantic](https://docs.pydantic.dev/) / pydantic-settings |
| Observability | [LangSmith](https://www.langchain.com/langsmith) tracing (optional) |
| Testing | pytest, with a fully in-memory `FakeCalendarClient` — no real network calls in the test suite |
| Linting / types | Ruff + mypy (strict) |

## Project structure

```
src/medical_agent/
├── main.py                    # entry point: FastAPI app (/chat, /admin) and CLI mode
├── core/config.py              # Settings (pydantic-settings), read from .env
├── exceptions.py                # GatewayError / UpstreamProviderError
├── schemas/chat.py              # ChatRequest / ChatResponse (HTTP contracts)
├── graph/
│   ├── factory.py               # builds and compiles the StateGraph
│   ├── entry.py                  # factory used by `langgraph dev` (see langgraph.json)
│   ├── routing.py                 # routes on the extracted intent
│   ├── state.py                   # AppointmentState (TypedDict)
│   └── nodes/                      # identify_intent, scheduler, canceller, checker, message_generator
├── prompts/v1/                   # prompt-building code
│   ├── templates/*.md             # the actual prompt text lives here, not in .py files
│   └── _loader.py                  # parses `### section` headers out of the .md files
└── services/
    ├── calendar_client.py           # CalendarClient Protocol + Professional
    ├── google_calendar_client.py     # real implementation (Google Calendar API)
    ├── fake_calendar_client.py        # in-memory implementation, used only in tests
    ├── appointment_service.py         # business logic: booking, cancelling, lookups
    └── llm.py                          # OpenRouterService (LangSmith-traced)
tests/                            # unit tests, one FakeCalendarClient, no real API calls
```

Every service is injected as a dependency (`AppointmentService` takes a `CalendarClient`, node factories take an `AppointmentService`/`BaseLLMService`), so the graph, the HTTP app, and the tests can all be assembled with different implementations without touching the business logic.

## Setup

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/)
- An [OpenRouter](https://openrouter.ai/) API key
- A Google Cloud project with the Calendar API enabled

### 1. Install dependencies

```bash
poetry install
```

### 2. Create a Google Calendar service account

1. In the [Google Cloud Console](https://console.cloud.google.com/), create (or reuse) a project and enable the **Google Calendar API** for it.
2. Create a **service account** and generate a JSON key for it.
3. For each professional, share their Google Calendar with the service account's email (found in the JSON key as `client_email`), with **"Make changes to events"** permission or higher. Viewer-only access is not enough — the agent needs to create and delete events.
4. Sharing alone does **not** make the calendar show up for the service account — see the note in [API reference](#api-reference) about registering it.

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Authenticates calls to OpenRouter |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | yes | The full service account key, as one line of JSON |
| `ADMIN_API_KEY` | yes | Protects the `/admin/professionals/*` endpoints (generate with `openssl rand -hex 32`) |
| `LANGSMITH_TRACING` | no | Set to `true` to enable tracing |
| `LANGSMITH_API_KEY` | no | Required only if tracing is enabled |
| `LANGSMITH_PROJECT` | no | LangSmith project name for traces |
| `OPENROUTER_BASE_URL` | no | Base URL of the LLM provider. Defaults to OpenRouter directly; see [Routing LLM traffic through a gateway](#routing-llm-traffic-through-a-gateway-optional) |
| `CF_AIG_TOKEN` | no | Cloudflare AI Gateway token. Only needed when that gateway has Authenticated Gateway enabled |

The model list, temperature, timezone (default `America/Sao_Paulo`) and a few other defaults live in `src/medical_agent/core/config.py` rather than in `.env` — adjust them there if needed.

## Running the agent

Short commands are wired up via [taskipy](https://github.com/taskipy/taskipy):

```bash
poetry run task server   # FastAPI server on http://0.0.0.0:8000, exposing POST /chat
poetry run task cli      # talk to the agent directly in the terminal
poetry run task studio   # langgraph dev -- visual debugger (see below)
```

### LangGraph Studio

`poetry run task studio` runs [`langgraph dev`](https://docs.langchain.com/oss/python/langgraph/local-server), which starts an in-memory API server (`http://127.0.0.1:2024`) for the compiled graph and prints a LangSmith Studio link (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`). Studio runs in the browser but talks to that local server, so you can see the node graph, step through a run turn by turn, edit the state, and replay from any checkpoint -- without touching `/chat` or the CLI.

It's driven by `langgraph.json` at the repo root, which points at `src/medical_agent/graph/entry.py:graph` -- a factory that builds the same graph as `main.build_context()`, minus the `async with` cleanup (the dev server owns the process for the whole session, so nothing needs to be closed early). `langgraph dev` also instruments known blocking calls (sqlite3, file reads) on its event loop, so `entry.py` builds `ConversationStateStore` and `GoogleCalendarClient` inside `asyncio.to_thread` -- neither is truly async, but this keeps the dev server's watchdog happy without rewriting them. `LANGSMITH_API_KEY` in `.env` is optional for the server to run, but required for traces to show up in LangSmith; in-Studio tracing itself needs `langgraph-api` 0.11.0+, which is why it's pinned directly in `[tool.poetry.group.dev.dependencies]` instead of relying on whatever `langgraph-cli[inmem]` resolves to.

## API reference

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "I am Maria Santos, I want to see a cardiologist next Monday at 4pm",
    "session_id": "any-string-that-identifies-this-conversation"
  }'
```

`session_id` is required -- it's what ties together turns of the same conversation (so a follow-up message can fill in a detail you forgot the first time, or cancel an appointment without repeating your name). Use a stable value per conversation (e.g. a UUID your client generates once per chat session), not a new one on every request.

Returns the natural-language reply alongside structured fields describing what the agent actually did, useful for a UI or an automated client that shouldn't have to parse prose:

```json
{
  "model": "medical-agent",
  "content": "Your appointment with Dr. John Doe is confirmed for Monday at 16:00. See you then!",
  "intent": "schedule",
  "error": null,
  "calendar_id": "c_xxxx@group.calendar.google.com",
  "appointment_datetime": "2026-09-07T16:00:00-03:00"
}
```

### `POST /admin/professionals/{calendar_id}` and `DELETE /admin/professionals/{calendar_id}`

Both require an `x-admin-key` header matching `ADMIN_API_KEY`.

Sharing a calendar with the service account only grants it *permission* — it does not add the calendar to the service account's own calendar list, so it wouldn't show up in `list_professionals()` on its own. This endpoint performs that one-time registration:

```bash
curl -X POST http://localhost:8000/admin/professionals/<calendar_id> \
  -H "x-admin-key: $ADMIN_API_KEY"
```

Both calls are idempotent — registering an already-registered calendar, or removing one that isn't registered, returns a `2xx` with a `status` field (`registered` / `already_registered` / `removed` / `not_registered`) rather than an error. Removing a professional (e.g. they've left) is the same call with `DELETE` instead of `POST`.

## Routing LLM traffic through a gateway (optional)

`OPENROUTER_BASE_URL` sets the base URL the LLM service talks to. It defaults to OpenRouter itself, so leaving it unset produces a request byte for byte identical to one made with no gateway at all — the test suite never learns a gateway exists, local development can bypass it, and backing out during an incident means deleting one line from `.env` rather than shipping a deploy.

Pointing it at a [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/) provider-native endpoint routes every call through it:

```bash
OPENROUTER_BASE_URL=https://gateway.ai.cloudflare.com/v1/<ACCOUNT_ID>/<GATEWAY>/openrouter
CF_AIG_TOKEN=<token>   # only when Authenticated Gateway is enabled
```

The provider-native endpoint is used rather than Cloudflare's unified REST API because it forwards the request body untouched, preserving OpenRouter's own `models` fallback array and `provider` routing — the unified endpoint normalises both away.

What it buys, none of which changes the agent's behaviour:

| Capability | Why it matters here |
|---|---|
| Cost and latency per call | Tagged per graph node, so it is visible which of the two LLM calls is the expensive one |
| Which model actually answered | `models` is a fallback list; the gateway log names the one that served each request |
| Response caching | Identical requests return from cache, with no tokens spent |
| Rate limiting | Enforced at Cloudflare's edge, before the request reaches the provider |
| DLP scanning | Flags identifiers — national ID numbers among them — travelling inside prompts |

This complements LangSmith rather than replacing it. LangSmith traces what happened *inside* the graph: the prompt, the extracted `Intent`, which branch ran. The gateway records what happened *on the wire*: status code, cost, cache hit, latency. When a date is extracted wrong, the answer is in LangSmith; when a call takes nine seconds, it is in the gateway.

## Development

```bash
poetry run task lint         # ruff check
poetry run task type_check   # mypy --strict
poetry run task test         # pytest, with an HTML coverage report in htmlcov/ (fails under 85% coverage)
poetry run task gate         # all three of the above, in sequence
```

CI runs the same three checks on every push and pull request to `main` (see `.github/workflows/ci.yml`).

Tests never hit a real API: `FakeCalendarClient` is a fully in-memory stand-in for `CalendarClient`, so the business logic (booking, cancelling, conflict detection, cross-patient isolation) is verified deterministically and for free.

## Design principles

- **The LLM only translates, never decides.** It turns a patient's message into structured data, and turns a decided outcome back into prose. Every actual decision — is this slot free, does this professional exist, is this the right patient's appointment — is made by ordinary, unit-tested Python.
- **Trust, but verify.** Every boundary — the LLM's structured output, an HTTP response body, a value pulled from another service — is re-validated in code rather than trusted blindly.
- **Prompts are data, not code.** Prompt text lives in `.md` files under `prompts/v1/templates/`, not in Python string literals, so editing prose never triggers a linter and prompt changes are reviewable as plain text diffs.
- **No hardcoded roster.** Professionals are whoever currently has a calendar shared (and registered) with the service account — adding or removing one is an operational action, not a deploy.
- **Patient isolation by construction.** `patient_name` is mandatory for every scheduling and cancellation request; searches for "my appointment" without an exact date/time are always scoped to the named professional when one is given, so one patient's request can never surface or act on another patient's appointment.

## Known limitations

- **Single calendar per professional, single timezone.** All calendars are assumed to be in the timezone configured in `core/config.py`.
