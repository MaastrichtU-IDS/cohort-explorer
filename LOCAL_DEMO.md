# Local Cohort Explorer ↔ AADCR v2 demo

This lane connects Cohort Explorer to the real Advanced Analytics Data Clean Room
v2 interface from Delta. It is for deterministic local demonstrations and
integration development. It is **not** a confidential-computing environment or a
production Data Clean Room security boundary.

The product boundary is deliberate:

1. Cohort Explorer owns metadata workbook and dictionary upload, exploration,
   validation, concept mapping, cross-study mapping selection, and room planning.
2. Cohort Explorer creates one AADCR v2 room plus metadata-derived data-node slots
   as unmerged DEV changes.
3. The browser follows the returned room link into the real AADCR v2 UI.
4. Synthetic CSV upload/provisioning, participants, permissions, computation nodes,
   merge requests, execution, results, and audit happen in AADCR v2.

The two products keep their own visual identity: Cohort Explorer owns the metadata
experience on port 3001, and the handoff on port 3002 uses Delta's unchanged
Decentriq `StylesWrapper` and production themes.

No Cohort Explorer-local “My DCRs” page is part of this lane. External concept and
mapping services are fixture-backed, so no model, ontology service, Decentriq token,
OAuth account, external LLM, embedding service, Athena service, or live SPARQL source
is required.

The deterministic synthetic pack contains plausible cardiovascular TIME-CHF and
GISSI-HF rows derived from the repository’s metadata dictionaries and mapping
semantics. It is synthetic-only and must never be presented as real cohort data.

## Pinned source boundary

Launch scripts fail closed unless the adjacent Delta checkout exactly matches this
clean reviewed integration:

- `davstur/aadcrv2` baseline `f13ef54fc3f0f56dae185d4aa35c6dff01ee8839`;
- local integration `0ebd0b94366c38bab442bcd69db568feede040f8`.

That integration commit moved forward on 28 Jul 2026 to carry the work behind the
WP7/WP8 report figures. Nothing in this lane's contract changed, but three things are
worth knowing:

- The AI explanation and merge-request assessment no longer invent content. The merge
  view is a change set, and it used to be handed to the model labelled as a complete
  state, so an assessment would report that a request adding one computation removed
  every participant, dataset and permission in the room.
- The explanation provider is chosen by whichever API key is configured
  (`OPENAI_API_KEY` or `GOOGLE_AI_API_KEY`). With neither, the endpoints keep the
  existing 503 "provider is disabled" contract, which is what this lane relies on.
- There are now two opt-in demo rooms in Delta, both off by default, so this lane
  still creates its own room from cohort metadata. `ensure_sample_room` seeds the
  single-owner "Cohort analysis (local sample)"; `ensure_demo_clean_room` seeds a
  two-owner "Cohort analysis" whose second dataset forces a real approval instead of
  self-approving.

Default adjacent checkouts:

```text
$HOME/projects/cohort-explorer-aadcrv2
$HOME/projects/delta-aadcrv2
```

Set `AADCRV2_REPO_DIR` before the first command to use another Delta checkout. Each
Compose namespace gets independent 256-bit session and service-JWT secrets under
`.demo-state/<namespace>/runtime.env` with mode `0600`.

## Prerequisites

- current Docker Desktop with Docker Compose
- Git 2.39+
- Python 3.11 and `uv`
- Node.js 20 or 22 LTS with npm 10+
- both repositories at the pinned source boundary

Initial setup may need network access for package and image downloads. Runtime
application traffic is loopback-only and the application containers have no public
egress.

## Start and verify

From the Cohort Explorer repository:

```bash
make demo-generate
make demo-up
make demo-seed
make demo-smoke
```

`demo-generate` creates or validates `data/synthetic-demo-pack`. Generation is
deterministic and refuses to overwrite an unmarked directory. Regenerate a pack
created by this repository with:

```bash
DEMO_FORCE=true make demo-generate
```

Services bind only to loopback:

| Service | URL |
|---|---|
| Cohort Explorer UI | `http://localhost:3001` |
| AADCR v2 UI | `http://localhost:3002` |
| Cohort Explorer API | `http://127.0.0.1:3000` |
| AADCR v2 API | `http://127.0.0.1:18000` |

Local login uses `nikolas.molyndris@decentriq.ch`. The AADCR UI obtains a short-lived
local-only token through its reverse proxy; the token is kept in memory and the
endpoint is disabled by default outside this explicit local configuration.

The API smoke verifies an empty baseline, guarded login, metadata/dictionary upload,
fixture mapping, deterministic definition archives, one room, exactly eight
metadata-derived DEV data nodes, and **zero** participants, permissions, computation
nodes, merge requests, PROD nodes, or provisioned datasets before handoff. It also
checks replay idempotency, invalid/unauthorized tokens, upload limits, gateway origin
pinning, Host allowlisting, loopback publication, internal networking, and blocked
AADCR public egress.

Safe evidence is written to
`.demo-state/cohort-explorer-aadcr-demo/smoke-evidence.json`.

## Browser acceptance

For the full user-owned journey from Cohort Explorer into the real AADCR v2 UI:

```bash
make demo-down
make demo-browser-install
make demo-browser-test
```

Set `DEMO_BROWSER_HEADED=true` for visible Chromium. The test leaves the stack
running for inspection. See [LOCAL_DEMO_BROWSER_CHECKLIST.md](LOCAL_DEMO_BROWSER_CHECKLIST.md).

`make demo-browser-ready` is a useful manual checkpoint: it creates a fresh
namespace, seeds only the central workbook, and prints both UI URLs, the admin email,
pack path, namespace, and evidence directory.

## Isolation and cleanup

The CE backend/frontend, Oxigraph, AADCR API, and AADCR UI join one internal Docker
network and publish no direct host ports. A read-only unprivileged nginx gateway
binds four loopback ports, rejects non-local Host headers, and strips CE cookies
before forwarding cross-service traffic. The immutable synthetic pack is mounted
read-only; mutable data and secrets live in namespace-specific state/volumes. No
container mounts the Docker socket.

Stop while retaining volumes:

```bash
make demo-down
```

Remove the namespace’s containers and mutable volumes:

```bash
DEMO_PURGE=true make demo-down
```

If smoke reports a non-fresh room or mapping baseline, purge and restart. If checkout
validation fails, select the pinned clean Delta revision. Inspect service failures
with the exact namespace and env file printed by the launch command.
