# Local browser acceptance

The browser lane starts from a fresh namespace containing only the central cohort
workbook. It proves the user-owned path without inheriting dictionaries, mappings,
rooms, or cookies from an API seed. The runner refuses a dirty Cohort Explorer tree,
and the AADCR launcher independently requires its exact clean reviewed revision, so
the revisions in passing evidence identify the code that actually ran.

Use Node.js 20 or 22 LTS with npm 10+. The first install needs access to the npm
registry and Playwright's Chromium download; the acceptance run itself is
loopback-only.

Run from the repository root after stopping any other process on ports 3000, 3001,
or 18000. If a prior browser lane is still open, first use the namespace printed by
that run and saved in its `run-summary.json`:

```bash
COMPOSE_PROJECT_NAME=<previous-namespace> DEMO_PURGE=true make demo-down
make demo-browser-install
make demo-browser-test
```

Set `DEMO_BROWSER_HEADED=true` for a visible Chromium run:

```bash
DEMO_BROWSER_HEADED=true make demo-browser-test
```

The one-worker test verifies:

- guarded local login as `nikolas.molyndris@decentriq.ch` and administrator access;
- zero runtime dictionaries, mappings, and rooms before browser actions;
- real cohort cards, counts, EDA JSON, variable images, and metadata download;
- validation, upload, and replacement of both synthetic dictionaries;
- OR, AND, and exact search; study/institute, OMOP-domain, data-type,
  category-count, visit, outcome, and source filters; equivalent-variable grouping;
  and persistent variable and category concept mappings across a changed-label
  dictionary replacement plus rejected malformed replacement;
- fixture-backed mapping generation, cache activity, table, graph, and download;
- all six wizard steps, explicit participants/research/samples/mapping/upload-slot
  choices, deterministic definition inspection with every archived asset hash
  checked against its source, and provider-accurate security copy;
- one matching AADCR room, participants, nodes, seven provisioned datasets, refresh,
  persistence after reload, audit events, and the exact aggregate-result SHA-256;
- named screenshots at every material checkpoint; and
- no page errors, unapproved console errors, unexpected failed local responses,
  HTTP(S) requests, or WebSocket connections to a non-loopback host.

The stack is deliberately left running. Named evidence is written below the selected
namespace at `.demo-state/<namespace>/browser-evidence/`; Playwright traces and
failure screenshots are ignored under `artifacts/browser-demo/`.

The passing evidence package contains `01-login-admin.png` through
`11-aggregate-result.png`, `dcr-definition.zip`, `dcr-definition.sha256`, and a
sanitized `run-summary.json`. It contains no cookies, tokens, secrets, real
participant or patient rows, or live-service URLs. The definition ZIP intentionally
includes the selected synthetic fixture samples.
