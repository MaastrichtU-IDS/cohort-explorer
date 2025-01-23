# 🫀 iCARE4CVD Cohort Explorer

Webapp enabling to upload and explore cohorts metadata, built for the [iCARE4CVD project](https://icare4cvd.eu).

It interacts with a privacy computing platform ([Decentriq](https://www.decentriq.com/)) to create secure workspace where data scientists can run analysis on the selected cohorts. The cohorts data is uploaded only to Decentriq, the explorer only uses cohorts metadata.

It aims to enable *data custodians* and *data scientists* to:

*   🔐 Login with their [Decentriq](https://www.decentriq.com/) account (OAuth based authentication, can be easily switch to other providers). Only accounts with the required permissions will be able to access the webapp.
    *   ✉️ Contact [Decentriq](https://www.decentriq.com/) to request an account if you are part of the iCARE4CVD project
*   📤 Data custodians upload CSV cohort metadata files describing the variables of a study cohort
*   🔎 Data scientists explore available cohorts and their variables through a web app:
    *   Full text search across all cohorts and variables
    *   Filter cohorts per types and providers
    *   Filter variables per OMOP domain, data type, categorical or not
*   🔗 Data custodians can map each variable of their cohorts to standard concepts, sourced from [OHDSI Athena](https://athena.ohdsi.org/search-terms/terms?query=) API (SNOMEDCT, LOINC...) through the web app.
    *   Mapping variables will help with data processing and exploration (⚠️ work in progress)
    *   We use namespaces from the [Bioregistry](https://bioregistry.io) to convert concepts CURIEs to URIs.
*   🛒 Data scientists can add the cohorts they need to perform their analysis to a [Data Clean Room](https://www.decentriq.com/) (DCR) on the Decentriq platform.
    *   Once complete, the data scientists can publish their DCR to Decentriq in one click.
    *   The DCR will be automatically created with a data schema corresponding to the selected cohorts, generated from the metadata provided by the data custodians.
    *   The data scientist can then access their DCR in Decentriq, write the code for their analysis, and request computation of this code on the provisioned cohorts.

> [!IMPORTANT]
>
> Only the owner of the cohort (as described in the spreadsheet holding all cohorts generic metadata), and the platform admins,  can upload the data dictionary or edit mappings for a cohort.

> [!NOTE]
>
> You can reupload a cohort dictionary that have been already uploaded (in case you need to fix something). The mappings defined via the Cohort Explorer will be kept, as long as the variables names do not change.

## ⚠️ Known issues

Here are a known "issues" with the Cohort Explorer, and how to fix them:

- [ ] After a period of inactivity you might see a black screen with an error message, in this case just reload the page

## 🗺️ Technical overview

This platform is composed of 3 main components:

*   **[Oxigraph](https://github.com/oxigraph/oxigraph) triplestore database** containing the cohorts and their variables metadata, exposing a SPARQL endpoint only available to the backend API.
    *   The data stored in the triplestore complies with the custom **[iCARE4CVD OWL ontology](https://maastrichtu-ids.github.io/cohort-explorer/)**. It contains 3 classes: Cohort, Variable, and Variable category. You can explore the ontology classes and properties [here](https://maastrichtu-ids.github.io/cohort-explorer/browse).
    *   Oxigraph has not yet reached release 1.0, but it is already stable enough for our currently expected use. It has the advantages of being open source (important for accountability and trust), and developed in Europe. If missing features appears to be blocking, consider migrating to [OpenLink Virtuoso](https://github.com/openlink/virtuoso-opensource), you'll only need to update the function that upload a RDFLib graph as file.

*   **`backend/` server**, built with python, FastAPI and RDFLib.
*   **`frontend/` web app** running in the client browser, built with TypeScript, NextJS, ReactJS, TailwindCSS, and DaisyUI.

🐳 Everything is deployed in docker containers define in the `docker-compose.yml` files.

🔐 Authentication is done through the Decentriq OAuth provider, but it could be replaced by any other OAuth provider easily. Once the user logged in through the external OAuth provider, the backend generates an encrypted JWT token, which is passed to the frontend using HTTP-only cookies.

> [!NOTE]
>
> All metadata about cohorts and variables are retrieved by one mighty SPARQL query, and passed to the frontend as one big dictionary. Filtering and searching is then done in TypeScript on this cohorts dictionary.
>
> We expect the amount of metadata for all cohorts will stay small enough to be handled directly on the client. If it becomes too big, it can be replaced by performing search and applying filters using SPARQL queries, to only retrieve metadata about relevant cohorts.

## ☑️ To do

*   [ ] Integrate the LUCE blockchain component for data sharing consent:
    *   [ ] We will store blockchain addresses, handle authentication, and add the UI elements directly in the Cohort Explorer (we can even store private keys or do wallet stuff there too if needed)
    *   [ ] But we need to be able to query the blockchain easily through an API from our system (a basic HTTP OpenAPI would suffice, e.g. built with [FastAPI](https://fastapi.tiangolo.com))

## 🧑‍💻 Development

[![Update ontology documentation](https://github.com/MaastrichtU-IDS/cohort-explorer/actions/workflows/docs.yml/badge.svg)](https://github.com/MaastrichtU-IDS/cohort-explorer/actions/workflows/docs.yml)

> [!WARNING]
>
> For deploying the Cohort Explorer in development or production you will **need the spreadsheet containing the general cohorts informations**. It needs to be an excel spreadsheet named `iCARE4CVD_Cohorts.xlsx` with a sheet named `Descriptions` with the following columns:
>
> | **Name of Study** | **Type** | **Institution** | Contact partner | Email | **Study type** | **N** | **Start date** | **End date** | **Study duration** | **Ongoing** | **Patient population** | **Primary objective** | **Primary endpoints** | **Secondary endpoints** |
> | ----------------- | -------- | --------------- | --------------- | ----- | -------------- | ----- | -------------- | ------------ | ------------------ | ----------- | ---------------------- | --------------------- | --------------------- | ----------------------- |
> |                   |          |                 |                 |       |                |       |                |              |                    |             |                        |                       |                       |                         |

### ⚡ Start for development

Requirements: [Docker](https://docs.docker.com/engine/install/)

1. Create a `.env` file with the secret configuration at the root of the repository (ask admins for the client ID and secret, and retrieve you Decentriq token from decentriq):

   ```bash
   AUTH_ENDPOINT=https://auth0.com
   CLIENT_ID=AAA
   CLIENT_SECRET=BBB
   DECENTRIQ_EMAIL=admin1@email.com
   DECENTRIQ_TOKEN=EEE
   JWT_SECRET=vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM
   ADMINS=admin1@email.com,admin2@email.com
   TOGATHER_API_KEY=AAA
   ```

   > [!WARNING]
   >
   > * Make sure to add the user you will login with to the list of `ADMINS` if you want to be able to upload any cohort.
   > * Get a token from Decentriq for your login email (the explorer needs a token to access Decentriq API)

2. Put the excel spreadsheet with all cohorts metadata in `data/iCARE4CVD_Cohorts.xlsx`. Uploaded cohorts will go to separated folders in `data/cohorts/`

3. Start the whole stack in development with docker compose, with hot reload for the frontend and backend:

    ```bash
    docker compose up
    ```

> [!IMPORTANT]
>
> For the authentication to the Decentriq OAuth provider to work you need to deploy the backend on <http://localhost:3000>

> [!NOTE]
>
> In development the user requesting a DCR will be added to as data owner of all cohorts dataset requested for development purpose (so they can provision the data themselves, and to avoid spamming emails owners when developing)

> [!TIP]
>
> Alternatively you can start the different component in in different terminals outside of docker. For this, put the `.env` file in the `backend` folder
>
> Start the database with docker:
>
> ```bash
> docker compose up db
> ```
>
> In a different terminal, start the backend with [hatch](https://hatch.pypa.io/latest/install/):
>
> ```bash
> cd backend
> DEV_MODE=true uv run uvicorn src.main:app --port 3000 --reload
> hatch run dev
> ```
>
> In another terminal, start the frontend with [pnpm](https://pnpm.io/installation):
>
> ```bash
> cd frontend
> pnpm install
> pnpm dev
> ```
>

### 🧹 Code formatting and linting

Automatically format Python code with ruff and black, and TypeScript code with prettier:

```bash
./scripts/fmt.sh
```

### 🆙 Upgrading dependencies

For the python backend, update the locked dependencies in `backend/requirements.txt` to the latest available (accepted ranges can be updated inside the `backend/pyproject.toml` file):

```bash
cd backend
hatch run compile
```

For the NextJS frontend:

```bash
cd frontend
pnpm up --latest
```

## 🐳 Deploy

Deploy on a server in production with docker compose.

Put the excel spreadsheet with all cohorts metadata in `data/iCARE4CVD_Cohorts.xlsx`. Uploaded cohorts will go to separated folders in `data/cohorts/`

1. Generate a secret key used to encode/decode JWT token for a secure authentication system:

    ```bash
    python -c "import secrets ; print(secrets.token_urlsafe(32))"
    ```

2. Create a `.env` file with secret configuration:

    ```bash
    AUTH_ENDPOINT=https://auth0.com
    CLIENT_ID=AAA
    CLIENT_SECRET=BBB
    DECENTRIQ_EMAIL=admin1@email.com
    DECENTRIQ_TOKEN=EEE
    JWT_SECRET=vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM
    ADMINS=admin1@email.com,admin2@email.com
    TOGATHER_API_KEY=AAA
    ```

3. Deploy the stack for production:

    ```bash
    docker compose -f docker-compose.prod.yml up -d
    ```

We currently use [nginx-proxy](https://github.com/nginx-proxy/nginx-proxy) for routing through environment variables in the `docker-compose.yml` file, you can change for the proxy of your liking.

## 🪄 Database administration

### 🗑️ Reset database

Reset the database by deleting the `data/db` folder:

```bash
rm -rf data/db
```

Next restart of the application the database will be re-populated using the data dictionaries CSV files stored on the server.

> [!WARNING]
>
> Resetting the database only if really necessary, it will cause to lose:
>
> - All concept mappings added from the Cohort Explorer
> - The info about Decentriq airlock data preview for cohorts that have been uploaded (it will default to false when recreating the database, admins can update them by downloading and reuploading the cohorts with the right airlock setting)

### 💾 Backup database

It can be convenient to dump the content of the triplestore database to create a backup.

To backup the triplestore in development:

```bash
curl -X GET -H 'Accept: application/n-quads' http://localhost:7878/store > data/triplestore_dump_$(date +%Y%m%d).nq
```

In production you will need to run it in the `backend` docker container, e.g. with:

```bash
docker compose exec backend curl -X GET -H 'Accept: application/n-quads' http://db:7878/store > data/triplestore_dump_$(date +%Y%m%d).nq
```

> \[!CAUTION]
>
> The path given for `triplestore_dump.nq` is **outside** the docker container

### ♻️ Reload database from backup

You can easily reload the dump in an empty triplestore.

Change the name of the backup file to yours, with the right date:

```bash
curl -X POST -T data/triplestore_dump_20240225.nq -H 'Content-Type: application/n-quads' http://localhost:7878/store
```

In production you will need to run it in the `backend` docker container, e.g. with:

```bash
docker compose exec backend curl -X POST -T /data/triplestore_dump_20240225.nq -H 'Content-Type: application/n-quads' http://localhost:7878/store
```

> \[!CAUTION]
>
> The path given for `triplestore_dump.nq` is **inside** the docker container

### 🚚 Move the app

If you need to move the app to a different server, just copy the whole `data/` folder.

## ✨ Automatically generate variables metadata

Experimental: you can use the [`csvw-ontomap`](https://github.com/vemonet/csvw-ontomap) python package to automatically generate a CSV metadata file for your data file, with the format expected by iCARE4CVD. It will automatically fill the following columns: var name, var type, categorical, min, max. But it does not properly extract datetime data types.

Install the package:

```bash
pip install git+https://github.com/vemonet/csvw-ontomap.git
```

Run profiling, supports `.csv`, `.xlsx`, `.sav`:

```bash
csvw-ontomap data/COHORT_data.sav -o data/COHORT_datadictionary.csv
```

## TODO

- [ ] For each variable add button to show popup of image with stats
