# Local browser acceptance

The browser lane begins with only the central metadata workbook seeded. It proves
the user-owned handoff from Cohort Explorer into the real AADCR v2 interface without
inheriting dictionaries, mappings, rooms, or cookies from an API-only full flow.

Run from the Cohort Explorer repository after stopping anything using ports 3000,
3001, 3002, or 18000:

```bash
make demo-browser-install
make demo-browser-test
```

For visible Chromium:

```bash
DEMO_BROWSER_HEADED=true make demo-browser-test
```

The runner requires clean pinned source revisions and one Playwright worker. It
verifies:

- guarded local login as `nikolas.molyndris@decentriq.ch`;
- real cohort cards, metadata counts, EDA, downloads, and variable imagery;
- validation, upload, and replacement of both synthetic dictionaries;
- search/filter modes and persistent variable/category concept mappings;
- fixture-backed cross-study mapping generation, table, graph, and download;
- the three ownership-specific handoff steps: room/cohorts, mapping slots, and
  review/handoff;
- CE creates exactly one AADCR room with eight unmerged metadata-derived DEV data
  slots and does not add participants, permissions, computations, merges, or data;
- the CE success action opens `/aadcrv2/dcr/<room-id>` on the separate AADCR v2 UI;
- the AADCR surface retains Delta's original Decentriq theme and does not inherit
  Cohort Explorer colours or layout;
- in AADCR v2, the local admin uploads and provisions a synthetic CSV in DEV, opens
  the Python computation editor, adds a participant, creates a change request, and
  verifies the corresponding audit trail;
- no page errors, unapproved console errors, failed local responses, or network
  requests/WebSockets to non-loopback hosts.

Passing evidence lives under
`.demo-state/<namespace>/browser-evidence/`. It contains named screenshots,
a sanitized `run-summary.json`; it must contain no cookies, tokens, secrets, or
synthetic row contents. Playwright traces and failure screenshots remain under
ignored `artifacts/browser-demo/`.

The stack stays running after acceptance so the final browser can remain on the real
AADCR UI for manual inspection. Tear it down with the namespace recorded in the run
summary:

```bash
COMPOSE_PROJECT_NAME=<namespace> DEMO_PURGE=true make demo-down
```
