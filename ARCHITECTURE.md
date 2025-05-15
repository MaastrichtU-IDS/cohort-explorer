# Cohort Explorer Architecture

This document provides a detailed overview of the Cohort Explorer's technical architecture, components, data flow, and key technical decisions.

## 1. System Overview

The Cohort Explorer is a web application designed to manage and explore metadata associated with medical research cohorts within the iCARE4CVD project. It facilitates the connection between data custodians (who provide metadata) and data scientists (who want to analyze data) by providing a centralized platform for metadata discovery and enabling the setup of secure analysis environments (Data Clean Rooms - DCRs) on the Decentriq platform.

**Core Components:**

*   **Frontend:** A Next.js single-page application providing the user interface.
*   **Backend:** A Python FastAPI server acting as the API and business logic layer.
*   **Database:** An Oxigraph RDF triplestore for storing and querying metadata.
*   **Deployment:** Docker Compose for containerization and orchestration.
*   **External Integration:** Decentriq platform (for authentication and DCR creation).

## 2. Component Breakdown

### 2.1. Frontend (`frontend/`)

*   **Framework:** Next.js 14+ (React 18+)
*   **Language:** TypeScript
*   **UI:** TailwindCSS, DaisyUI (Tailwind component library), React Feather (Icons)
*   **State Management:** Primarily React Context API and component state. For server state (data fetching), it relies on direct API calls (e.g., using `fetch` within React components or Server Components if applicable).
*   **Authentication Client:** NextAuth.js (configured for OAuth with JWT session strategy).
*   **Key Responsibilities:**
    *   Rendering the user interface for browsing cohorts and variables.
    *   Handling user login/logout flow via NextAuth.js, interacting with the backend's auth endpoints.
    *   Providing forms for uploading cohort data dictionaries (CSV files).
    *   Displaying variable details and providing an interface for mapping variables/categories to ontology terms.
    *   Implementing client-side search and filtering on the fetched metadata.
    *   Managing the DCR "basket" and triggering DCR creation via backend API calls.
    *   Communicating with the Backend API (`/api/*` proxied by Next.js or directly via `NEXT_PUBLIC_API_URL`).

### 2.2. Backend (`backend/`)

*   **Framework:** FastAPI
*   **Language:** Python 3.11+
*   **Key Libraries:**
    *   `uvicorn`: ASGI server.
    *   `python-multipart`: Form data parsing.
    *   `rdflib`: Creating, parsing, and manipulating RDF graphs.
    *   `SPARQLWrapper`: Executing SPARQL queries/updates against the Oxigraph endpoint.
    *   `pandas`: Parsing CSV and Excel (`.xlsx`) files.
    *   `python-jose`: JWT creation and validation for session management.
    *   `httpx` (or `requests`): Making requests to Decentriq API and potentially other external services (e.g., OHDSI Athena, Bioregistry).
    *   `python-dotenv`: Loading environment variables.
*   **Key Responsibilities:**
    *   Providing RESTful API endpoints for the frontend.
    *   Handling OAuth callbacks from Decentriq, validating tokens, fetching user info.
    *   Managing user sessions using JWT stored in HTTP-only cookies.
    *   Authorizing requests based on user roles (e.g., admin checks, cohort ownership checks).
    *   Receiving uploaded metadata files (CSV, Excel), validating their structure and content.
    *   Parsing metadata files and transforming the data into RDF triples based on the `cohort-explorer-ontology.ttl`.
    *   Storing/updating RDF triples in the Oxigraph triplestore via SPARQL 1.1 Update.
    *   Executing SPARQL queries to retrieve cohort/variable metadata for the frontend.
    *   Handling requests to create/update mappings.
    *   Interacting with the Decentriq API to create and configure Data Clean Rooms (DCRs).
    *   Managing the persistence of uploaded files (`data/cohorts/`, `data/iCARE4CVD_Cohorts.xlsx`).
    *   Initializing the triplestore on startup (`init_triplestore`).

### 2.3. Database (`db` service in Docker Compose)

*   **Type:** RDF Triplestore
*   **Engine:** Oxigraph
*   **Interface:** SPARQL 1.1 Query and Update endpoint (HTTP).
*   **Persistence:** Data is stored on the filesystem within the container, mounted to `./data/db` on the host via Docker volumes.
*   **Data Model:** Uses named graphs to segregate data:
    *   `icare:graph/metadata`: General cohort information loaded from the main Excel file.
    *   `icare:cohort/{cohort_id}`: Variable definitions and metadata extracted from the uploaded CSV data dictionary for a specific cohort.
    *   `icare:cohort/{cohort_id}/mappings`: Mappings (variable/category -> standard concept) created via the UI for a specific cohort.
    *   `icare:ontology`: (Potentially, or loaded directly) The iCARE4CVD and potentially base ontologies (like OMOP CDM). *(Self-correction: Initialization loads the ontology triples directly into a graph, likely `https://w3id.org/icare4cvd/omop-cdm-v6` based on `upload.py`)*
*   **Key Responsibilities:**
    *   Storing all cohort and variable metadata according to the RDF model.
    *   Executing SPARQL queries received from the backend.
    *   Executing SPARQL updates (INSERT DATA, DELETE WHERE) received from the backend.
    *   Ensuring data persistence through volume mounting.

## 3. Data Flow and Interactions

**(See Data Flow diagram in README.md for a visual representation)**

1.  **Metadata Fetch:** Frontend requests cohort data -> Backend executes a large SPARQL query against Oxigraph -> Oxigraph returns results -> Backend formats as JSON -> Frontend receives JSON and renders UI, enabling client-side filtering/search.
2.  **Upload:** Frontend uploads CSV -> Backend receives file -> Backend validates, parses with Pandas -> Backend generates RDF triples using RDFLib -> Backend sends SPARQL INSERT/DELETE updates to Oxigraph.
3.  **Mapping:** Frontend sends mapping details (variable URI, property, concept URI/CURIE, label) -> Backend validates -> Backend generates SPARQL INSERT/DELETE updates for the mapping graph -> Backend sends updates to Oxigraph.
4.  **Authentication:** Frontend redirects to Decentriq -> User logs in -> Decentriq redirects to Backend (`/auth/callback`) with code -> Backend exchanges code for token, validates, gets user info -> Backend creates JWT -> Backend sets HTTP-only cookie on response -> Frontend stores cookie.
5.  **Authenticated Request:** Frontend sends request with JWT cookie -> Backend middleware verifies JWT -> Backend processes request.
6.  **DCR Creation:** Frontend sends selected cohort IDs -> Backend retrieves necessary metadata for these cohorts via SPARQL query -> Backend constructs payload for Decentriq API -> Backend calls Decentriq API (`create_provision_dcr`) -> Backend returns result/status to Frontend.

## 4. Key Design Decisions

*   **Metadata Storage (RDF Triplestore):**
    *   **Pros:** Flexible schema, semantic querying capabilities (SPARQL), well-suited for linked data and ontology integration, potential for future semantic reasoning.
    *   **Cons:** Steeper learning curve than relational databases, SPARQL performance can be complex to optimize, Oxigraph is relatively new compared to some other triplestores (though stable for current use).
    *   **Rationale:** Aligns well with the project's goal of mapping variables to standard ontologies and representing heterogeneous metadata structures.
*   **Client-Side Filtering/Search:**
    *   **Pros:** Reduces load on the backend/database for simple filtering operations, potentially faster UI interactions after initial load.
    *   **Cons:** Requires transferring all metadata initially (can be slow/memory-intensive if data grows significantly), search capabilities limited by client-side implementation.
    *   **Rationale:** Considered acceptable given the expected data volume for iCARE4CVD. Can be revisited if performance degrades.
*   **Technology Choices:**
    *   **FastAPI (Python Backend):** Modern, high-performance, good tooling, strong Python ecosystem for data handling (Pandas) and RDF (RDFLib).
    *   **Next.js (TypeScript Frontend):** Robust React framework, good developer experience, TypeScript provides type safety.
    *   **Oxigraph:** Open-source, performant, standards-compliant RDF store.
    *   **Docker Compose:** Standard and straightforward for managing multi-container local development and deployment.
*   **Authentication (Decentriq OAuth + JWT Sessions):** Leverages existing Decentriq accounts, standard OAuth flow, JWT provides stateless session management suitable for APIs.
*   **Named Graphs:** Used effectively to separate general metadata, cohort-specific definitions, and user-generated mappings, allowing for easier management and targeted updates/deletions.

## 5. Deployment Architecture

**(See Deployment section in README.md for setup details)**

*   **Containerization:** All components (Frontend, Backend, DB) run in Docker containers.
*   **Orchestration:** Docker Compose manages the lifecycle and networking of containers.
*   **Environment Configuration:** Uses `.env` files for secrets and environment-specific settings.
*   **Persistence:** Docker volumes are used to persist Oxigraph data (`./data/db`) and uploaded cohort files (`./data/cohorts`, `./data/iCARE4CVD_Cohorts.xlsx`).
*   **Production:** Assumes deployment behind a reverse proxy (like Nginx) responsible for:
    *   Handling incoming HTTPS traffic.
    *   Terminating SSL.
    *   Routing requests to the appropriate containers based on hostname (e.g., `explorer.yourdomain.com` -> frontend, `api.explorer.yourdomain.com` -> backend).
    *   Optionally serving static assets built by Next.js.

## 6. Future Considerations

*   **Scalability:** If metadata volume increases significantly, transition filtering/searching logic to the backend using targeted SPARQL queries.
*   **Mapping Engine:** Enhance the mapping capabilities, potentially integrating more sophisticated mapping tools or suggestions.
*   **Statistics/Visualizations:** Implement backend logic to compute and store variable statistics, and corresponding frontend components to display them.
*   **Blockchain Integration:** Integrate with LUCE blockchain for consent management (requires defining API interactions).
*   **Monitoring/Logging:** Enhance monitoring and structured logging for better observability in production.
*   **Testing:** Implement more comprehensive automated tests (unit, integration, end-to-end). 