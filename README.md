# 🫀 iCARE4CVD Cohort Explorer

Webapp enabling the upload, exploration, and mapping of cohort metadata, built for the [iCARE4CVD project](https://icare4cvd.eu).


It interacts with a privacy computing platform ([Decentriq](https://www.decentriq.com/)) to create secure Data Clean Rooms (DCRs) where data scientists can run analyses on selected cohorts. The Cohort Explorer handles **only cohort metadata**; the actual sensitive cohort data is uploaded directly to Decentriq DCRs by data custodians after the DCR is configured via the Explorer.

## ✨ Key Features

This platform enables *data custodians* and *data scientists* to:

*   🔐 **Login:** Authenticate using their [Decentriq](https://www.decentriq.com/) account via OAuth. Access is restricted based on permissions configured within Decentriq.
    *   *Contact [Decentriq](https://www.decentriq.com/) to request an account if you are part of the iCARE4CVD project.*
*   ⬆️ **Upload Metadata:** Data custodians can upload CSV cohort metadata files (data dictionaries) describing the variables of their study cohort.
*   🔄 **Re-upload/Update:** Existing data dictionaries can be re-uploaded to correct or update information. Mappings defined via the Explorer are preserved if variable names remain unchanged.
*   🔍 **Explore & Search:** Data scientists can explore available cohorts and variables:
    *   Full-text search across all cohort names, descriptions, and variable details.
    *   Filter cohorts by type and provider institution.
    *   Filter variables by OMOP domain, data type (String, Integer, Float, Datetime), and whether they are categorical.
*   🔗 **Map Variables:** Data custodians can map variables and their categories within their cohorts to standard ontology concepts (e.g., SNOMED CT, LOINC) sourced via the [OHDSI Athena API](https://athena.ohdsi.org/search-terms/terms?query=).
    *   Mappings improve data harmonization and analysis capabilities (⚠️ Currently work in progress for full utilization).
    *   Uses namespaces from the [Bioregistry](https://bioregistry.io) to convert concept CURIEs (e.g., `snomedct:12345`) to URIs.
*   🛒 **Select & Configure DCR:** Data scientists can select cohorts for analysis and add them to a "basket".
*   🚀 **Publish DCR:** With one click, data scientists can trigger the creation of a [Data Clean Room](https://www.decentriq.com/) (DCR) on the Decentriq platform.
    *   The DCR is automatically configured with a data schema derived from the metadata of the selected cohorts.
    *   Data scientists can then access the DCR in Decentriq, write analysis code, and request computation on the provisioned cohorts (data provided separately by custodians).

> [!IMPORTANT]
> Only the designated owner of a cohort (as specified in the central `iCARE4CVD_Cohorts.xlsx` spreadsheet) and platform administrators can upload or modify the data dictionary and mappings for that cohort.

## 🏗️ Architecture Overview

The Cohort Explorer follows a standard web application architecture with three main components, orchestrated using Docker Compose:

1.  **Frontend (`frontend/`)**: A client-side web application built with **Next.js (React)** and **TypeScript**. It provides the user interface for exploring cohorts, managing uploads, defining mappings, and initiating DCR creation. It interacts with the Backend API. Styling is done with **TailwindCSS** and **DaisyUI**.
2.  **Backend (`backend/`)**: A **Python** API server built with **FastAPI**. It handles:
    *   Authentication (via Decentriq OAuth and JWT session tokens).
    *   Parsing uploaded metadata files (CSVs, Excel).
    *   Interacting with the Triplestore (SPARQL queries/updates).
    *   Managing data files on the server.
    *   Communicating with the Decentriq API for DCR creation.
    *   Serving metadata to the frontend.
3.  **Database (`db`)**: An **Oxigraph** triplestore instance. It stores all cohort and variable metadata as RDF triples, conforming to the custom iCARE4CVD ontology. It exposes a SPARQL endpoint accessible only by the Backend.

**Data Flow:**

1.  **Initial Metadata Load:** An administrator uploads the central `iCARE4CVD_Cohorts.xlsx` file containing general information about all participating cohorts. This data is parsed by the backend and stored in a dedicated graph (`icare:graph/metadata`) in the triplestore.
2.  **Data Dictionary Upload:** A data custodian uploads a CSV data dictionary for their specific cohort.
3.  **Backend Processing:** The backend validates the CSV, parses its content, and transforms it into RDF triples according to the iCARE4CVD ontology (`cohort-explorer-ontology.ttl`). These triples are stored in a named graph specific to the cohort (e.g., `icare:cohort/{cohort_id}`).
4.  **Mapping:** Custodians use the frontend UI to map variables/categories to standard concepts. The frontend sends requests to the backend (`/insert-triples` endpoint), which stores these mappings as RDF triples in a separate named graph for that cohort (e.g., `icare:cohort/{cohort_id}/mappings`).
5.  **Exploration:** The frontend fetches *all* cohort and variable metadata (including mappings) from the backend via a single comprehensive SPARQL query executed by the backend against the triplestore. Filtering and searching are performed client-side in TypeScript.
6.  **DCR Creation:** A data scientist selects cohorts and triggers DCR creation. The frontend sends the list of selected cohort IDs to the backend (`/create-provision-dcr` endpoint).
7.  **Decentriq Interaction:** The backend uses the cohort metadata (retrieved from the triplestore) to generate the necessary configuration and interacts with the Decentriq API to create and publish the DCR.

**Authentication:**

*   The frontend redirects the user to Decentriq for OAuth login.
*   Upon successful authentication, Decentriq redirects back to the backend (`/auth/callback`).
*   The backend verifies the OAuth token, fetches user details, and checks permissions.
*   The backend generates a JWT session token containing user information (email, roles).
*   This JWT is sent back to the frontend via a secure, HTTP-only cookie.
*   Subsequent requests from the frontend include this cookie, which the backend validates to authorize actions.

> [!NOTE]
> The current approach of fetching all metadata to the frontend works well for the expected scale of the iCARE4CVD project. If the number of cohorts/variables grows significantly, this could be optimized by performing filtering and searching directly via SPARQL queries on the backend, returning only the relevant subset of data to the frontend.

## 💾 Data Model

Metadata is stored in an Oxigraph triplestore using RDF. The structure is defined by a custom OWL ontology (`cohort-explorer-ontology.ttl`) with the namespace `icare: <https://w3id.org/icare4cvd/>`.

**Key Classes:**

*   `icare:Cohort`: Represents a study cohort. Linked to its variables via `icare:hasVariable`. Has properties like `dc:identifier`, `icare:institution`, `icare:cohortType`, `icare:studyParticipants`, etc. General cohort metadata resides in the `icare:graph/metadata` named graph.
*   `icare:Variable`: Represents a variable (column) within a cohort's data dictionary. Has properties like `dc:identifier`, `rdfs:label`, `icare:varType`, `icare:units`, `icare:omop` domain, etc. Linked to categories via `icare:categories`. Can be mapped using `icare:conceptId` (original) and `icare:mappedId` (user-defined). Variable definitions reside in the cohort's named graph (e.g., `icare:cohort/{cohort_id}`).
*   `icare:VariableCategory`: Represents a specific category for a categorical variable. Has `rdf:value`, `rdfs:label`, and can be mapped using `icare:mappedId`.

Mappings defined via the UI are stored in a separate named graph (e.g., `icare:cohort/{cohort_id}/mappings`) to distinguish them from the base metadata derived directly from the uploaded file.

## 🧑‍💻 Development

[![Update ontology documentation](https://github.com/MaastrichtU-IDS/cohort-explorer/actions/workflows/docs.yml/badge.svg)](https://github.com/MaastrichtU-IDS/cohort-explorer/actions/workflows/docs.yml)

> [!WARNING]
> Running the Cohort Explorer requires the spreadsheet containing general cohort information. It must be an Excel file named `iCARE4CVD_Cohorts.xlsx` located in the `data/` directory (create this directory if it doesn't exist). The spreadsheet needs a sheet named `Descriptions` with specific columns (see original README section for details).

### Prerequisites

*   [Docker](https://docs.docker.com/engine/install/) and Docker Compose
*   Optionally [uv](https://docs.astral.sh/uv/) (for Python environment management and better IDE integration)
*   Optionally [pnpm](https://pnpm.io/installation) (for frontend package management)

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cohort-explorer
    ```
2.  **Create `.env` file:** Copy the required environment variables (OAuth Client ID/Secret, Decentriq API token, JWT Secret, Admin emails, etc.) into a `.env` file at the root of the repository. Ask project administrators for necessary secrets.
    ```bash
    # Example .env structure
    AUTH_ENDPOINT=https://auth0.com # Or your Decentriq auth endpoint
    CLIENT_ID=YOUR_DECENTRIQ_CLIENT_ID
    CLIENT_SECRET=YOUR_DECENTRIQ_CLIENT_SECRET
    DECENTRIQ_EMAIL=YOUR_DECENTRIQ_ADMIN_EMAIL # Email linked to the token below
    DECENTRIQ_TOKEN=YOUR_DECENTRIQ_API_TOKEN # Get from Decentriq platform
    JWT_SECRET=GENERATE_A_STRONG_SECRET # e.g., using python -c "import secrets; print(secrets.token_urlsafe(32))"
    ADMINS=admin1@example.com,admin2@example.com # Comma-separated list of admin emails
    TOGATHER_API_KEY=YOUR_TOGATHER_API_KEY # If using Togather integration

    # Frontend needs this at build time if not using default localhost:3000
    # NEXT_PUBLIC_API_URL=http://localhost:3000
    ```
    > [!WARNING]
    > *   Ensure the user email you intend to log in with during development is included in the `ADMINS` list if you need administrative privileges (like uploading any cohort).
    > *   The `DECENTRIQ_TOKEN` is required for the backend to interact with the Decentriq API (e.g., for DCR creation).

3.  **Prepare Data Directory:**
    *   Create a `data/` directory at the root of the project if it doesn't exist.
    *   Place the `iCARE4CVD_Cohorts.xlsx` file inside the `data/` directory.
    *   Uploaded cohort dictionaries will be stored in `data/cohorts/`.
    *   The Oxigraph database files will be persisted in `data/db/`.

4.  **Start Services (Docker - Recommended):**
    ```bash
    docker compose up --build
    ```
    This command builds the images (if necessary) and starts the `frontend`, `backend`, and `db` services defined in `docker-compose.yml`. Hot-reloading is enabled for both the frontend and backend.
    *   Frontend will be accessible at `http://localhost:3001`.
    *   Backend API will be accessible at `http://localhost:3000`.
    *   Oxigraph SPARQL endpoint (for debugging, if needed) at `http://localhost:7878`.

    > [!IMPORTANT]
    > For Decentriq OAuth to function correctly during local development, the provider must be configured to allow redirects to `http://localhost:3000/auth/callback`.

    > [!TIP]
    > For optimal IDE support (autocompletion, type checking) within the `backend` code, install dependencies locally using `uv`:
    > ```bash
    > uv sync --system # Or use a virtual environment
    > ```
    > Similarly, for the frontend:
    > ```bash
    > cd frontend
    > pnpm install # or npm install / yarn install
    > cd ..
    > ```

5.  **Start Services (Manual - Alternative):**
    You can run each component manually outside Docker. Ensure the `.env` file is present in the `backend/` directory for this method.

    *   **Start Database:**
        ```bash
        docker compose up db
        ```
    *   **Start Backend (Terminal 2):**
        ```bash
        cd backend
        # Assumes .env file is here
        DEV_MODE=true uv run uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload
        cd ..
        ```
    *   **Start Frontend (Terminal 3):**
        ```bash
        cd frontend
        pnpm install # If not already done
        pnpm dev # Starts on http://localhost:3001 by default
        cd ..
        ```

### Code Formatting and Linting

Run the formatting script to ensure code consistency:

```bash
./scripts/fmt.sh
```
This uses Ruff/Black for Python and Prettier for TypeScript/JSON/etc.

### Upgrading Dependencies

*   **Python Backend:**
    ```bash
    # Make sure uv is installed and you are in the project root
    cd backend
    # Update dependencies in pyproject.toml if needed
    uv pip compile pyproject.toml -o requirements.txt --upgrade
    # Update locked dev dependencies (if using hatch)
    # hatch run compile:dev
    cd ..
    # Rebuild docker image if running in docker
    docker compose build backend
    ```
*   **TypeScript Frontend:**
    ```bash
    cd frontend
    pnpm up --latest # Or use npm/yarn equivalent
    cd ..
    # Rebuild docker image if running in docker
    docker compose build frontend
    ```

## 🐳 Deploy in Production

Deployment uses Docker Compose with the `docker-compose.prod.yml` configuration file, which builds optimized production images and typically runs behind a reverse proxy like Nginx.

1.  **Prerequisites:**
    *   Server with Docker and Docker Compose installed.
    *   Reverse proxy (e.g., Nginx Proxy Manager, Traefik, or vanilla Nginx) configured for handling SSL and routing requests to the containers. The example `docker-compose.prod.yml` includes labels for `nginx-proxy/nginx-proxy`.
    *   Ensure the `data/` directory exists on the host and contains the `iCARE4CVD_Cohorts.xlsx` file.

2.  **Configuration:**
    *   Create a `.env` file on the server with the production configuration (different hostnames, potentially different secrets, ensure `JWT_SECRET` is strong and persistent).
        ```bash
        # Example Production .env
        AUTH_ENDPOINT=https://auth0.com # Production Decentriq endpoint
        CLIENT_ID=PROD_DECENTRIQ_CLIENT_ID
        CLIENT_SECRET=PROD_DECENTRIQ_CLIENT_SECRET
        DECENTRIQ_EMAIL=PROD_DECENTRIQ_ADMIN_EMAIL
        DECENTRIQ_TOKEN=PROD_DECENTRIQ_API_TOKEN
        JWT_SECRET=YOUR_VERY_STRONG_PRODUCTION_SECRET
        ADMINS=prod_admin1@example.com,prod_admin2@example.com
        TOGATHER_API_KEY=PROD_TOGATHER_API_KEY

        # These are used by nginx-proxy in the example prod compose file
        # Ensure these match your DNS and proxy setup
        FRONTEND_HOSTNAME=explorer.yourdomain.com
        BACKEND_HOSTNAME=api.explorer.yourdomain.com
        ```
    *   Modify `docker-compose.prod.yml` if necessary:
        *   Adjust `NEXT_PUBLIC_API_URL` build argument for the frontend.
        *   Adjust `FRONTEND_URL` environment variable for the backend.
        *   Update `VIRTUAL_HOST`, `LETSENCRYPT_HOST`, etc., environment variables to match your domain and reverse proxy setup. Remove them if using a different proxy solution.

3.  **Deployment:**
    ```bash
    # Pull latest code
    git pull origin main # Or your production branch

    # Create/Update data directory and .env file on the server

    # Build and deploy
    docker compose -f docker-compose.prod.yml up --build -d
    ```
    The `--build` flag ensures images are rebuilt if the code has changed. `-d` runs the containers in detached mode.

4.  **Configure Reverse Proxy:** Ensure your reverse proxy correctly routes traffic for your chosen hostnames (e.g., `explorer.yourdomain.com`, `api.explorer.yourdomain.com`) to the appropriate container ports (`frontend:3000`, `backend:80`) and handles SSL termination.

## 🪄 Database Administration

The Oxigraph database persists its data in the `data/db` directory (mounted as a volume).

### 🗑️ Reset Database

**Use with caution!** This deletes all stored metadata and mappings.

1.  Stop the application: `docker compose down`
2.  Delete the database directory: `rm -rf data/db`
3.  Restart the application: `docker compose up -d` (or `docker compose -f docker-compose.prod.yml up -d` for production)

Upon restart, the `init_triplestore` function in the backend will run:
*   It will load the base ontology.
*   It will re-process the main `iCARE4CVD_Cohorts.xlsx` file.
*   It will re-process any existing `*_datadictionary.csv` files found in `data/cohorts/*/`.

> [!WARNING]
> Resetting the database **permanently deletes**:
> *   All concept mappings added manually via the Cohort Explorer UI.
> *   Information about the Decentriq "Airlock" data preview setting for cohorts (it defaults to `false` upon reloading from files; Admins must re-upload affected dictionaries with the correct setting).

### 💾 Backup Database

Dump the entire triplestore content (all named graphs) into a single N-Quads file.

*   **Development:**
    ```bash
    # Run from the host machine
    curl -X GET -H 'Accept: application/n-quads' http://localhost:7878/store > data/triplestore_dump_$(date +%Y%m%d).nq
    ```
*   **Production:** Execute the `curl` command *inside* the `backend` container, ensuring the output path is accessible from the host (e.g., mounted `/data` volume).
    ```bash
    # Run from the host machine
    docker compose -f docker-compose.prod.yml exec backend curl -X GET -H 'Accept: application/n-quads' http://db:7878/store > data/triplestore_dump_$(date +%Y%m%d).nq
    ```

### ♻️ Reload Database from Backup

Reload a dump into an *empty* triplestore.

1.  Reset the database first (see above).
2.  Ensure the application (specifically the `db` service) is running.
3.  Change the backup filename (`triplestore_dump_YYYYMMDD.nq`) accordingly.

*   **Development:**
    ```bash
    # Run from the host machine
    curl -X POST -T data/triplestore_dump_YYYYMMDD.nq -H 'Content-Type: application/n-quads' http://localhost:7878/store
    ```
*   **Production:** Execute the `curl` command *inside* the `backend` container, ensuring the input dump file is accessible *within* the container (e.g., in the mounted `/data` volume).
    ```bash
    # Run from the host machine
    # Note: The path /data/triplestore_dump... is INSIDE the container
    docker compose -f docker-compose.prod.yml exec backend curl -X POST -T /data/triplestore_dump_YYYYMMDD.nq -H 'Content-Type: application/n-quads' http://db:7878/store
    ```

### 🚚 Move the App

To move the application to a different server:

1.  Stop the application on the old server.
2.  Copy the entire `data/` directory (including `iCARE4CVD_Cohorts.xlsx`, `data/cohorts/`, and `data/db/`) to the new server.
3.  Set up the `.env` file on the new server.
4.  Deploy the application on the new server using Docker Compose.

## ⚠️ Known Issues / TODO

*   [ ] After a period of inactivity, the frontend might show a black screen or error; reloading the page usually fixes it (likely related to session expiry or WebSocket issues if any are used).
*   [ ] Integrate the LUCE blockchain component for data sharing consent (requires defining interaction points and potentially a dedicated API for the blockchain).
*   [ ] Add UI elements (e.g., buttons/popups) to display statistical summaries or visualizations for variables (requires backend changes to generate/store stats and frontend components).
*   [ ] Full utilization of variable mappings in analysis/downstream processes is still under development.
*   [ ] Automatic mapping generation (`generate_mappings` function in `upload.py`) is experimental and currently disabled.

## License

[LICENSE.txt](LICENSE.txt) (Please review the license file for details).
