# 🫀 iCare4CVD Cohort Explorer

Webapp built for the [iCare4CVD project](https://icare4cvd.eu).

It aims to enable data owners and data scientists to:

*   🔐 Login with their [Decentriq](https://www.decentriq.com/) account (OAuth based authentication, can be easily switch to other providers). Only accounts with the required permissions will be able to access the webapp.
    *   ✉️ Contact [Decentriq](https://www.decentriq.com/) to request an account if you are part of the iCare4CVD project
*   📤 Data owners upload CSV cohort metadata files describing the variables of a study cohort
*   🔎 Data scientists explore available cohorts and their variables through a web app:
    *   Full text search across all cohorts and variables
    *   Filter cohorts per types and providers
    *   Filter variables per OMOP domain, data type, categorical or not
*   🔗 Data owners can map each variable of their cohorts to standard concepts, sourced from [OHDSI Athena](https://athena.ohdsi.org/search-terms/terms?query=) API (SNOMEDCT, LOINC...) through the web app.
    *   Mapping variables will help with data processing and exploration (⚠️ work in progress)
    *   We use namespaces from the [Bioregistry](https://bioregistry.io) to convert concepts CURIEs to URIs.
*   🛒 Data scientists can add the cohorts they need to perform their analysis to a Data Clean Room (DCR)
    *   Once complete, the data scientists can publish their DCR to Decentriq in one click.
    *   The DCR will be automatically created with a data schema corresponding to the selected cohorts, generated from the metadata provided by the data owners.
    *   The data scientist can then access their DCR in Decentriq, write the code for their analysis, and request computation of this code on the provisioned cohorts.

> [!WARNING]
>
> If you logged in with a Decentriq user that does not have access to the Cohort Explorer, and need to re-login with another user: you will need to clear cache and cookies. Because Auth0 will keep your login in mind for some time, and it can be quite tricky to reset (they don't give the tools for managing that properly).

## 🗺️ Technical overview

This platform is composed of 3 main components:

*   **[Oxigraph](https://github.com/oxigraph/oxigraph) triplestore** containing the cohorts and their variables metadata, exposing a SPARQL endpoint only available to the backend API.
    *   The data stored in the triplestore complies with the custom **[iCARE4CVD OWL ontology](https://maastrichtu-ids.github.io/cohort-explorer/)**. It contains 3 classes: Cohort, Variable, and Variable category.

*   **`backend/` server**, built with python, FastAPI and RDFLib.
*   **`frontend/` web app** running on the client, built with TypeScript, NextJS, ReactJS, TailwindCSS, and DaisyUI.

🐳 Everything is deployed in docker containers using docker compose.

🔐 Authentication is done through the Decentriq OAuth provider, but it could be replaced by any other OAuth provider easily. Once the user logged in through the external OAuth provider, the backend generates an encrypted JWT token, which is passed to the frontend using HTTP-only cookies.

> \[!NOTE]
>
> All metadata about cohorts and variables are retrieved by one mighty SPARQL query, and passed to the frontend as one big dictionary. Filtering and searching is then done in TypeScript on this cohorts dictionary.
>
> We expect the amount of metadata for all cohorts will stay small enough to be handled directly on the client. If it becomes too big, it can be replaced by performing search and applying filters using SPARQL queries, to only retrieve metadata about relevant cohorts.

## ☑️ To do

*   [ ] Integrate LUCE blockchain component. Should it be deployed separately, or as a service in the `docker-compose.yml`?

## 🧑‍💻 Development

### 📥 Install dependencies

1.  Install [hatch](https://hatch.pypa.io/latest/) for managing python projects, and [pnpm](https://pnpm.io/installation) for TS/JS projects

    ```bash
    pip install hatch
    ```

2.  Create a `backend/.env` file with secret configuration:

    ```bash
    AUTH_ENDPOINT=https://auth0.com
    CLIENT_ID=AAA
    CLIENT_SECRET=BBB
    DECENTRIQ_EMAIL=ccc@ddd.com
    DECENTRIQ_TOKEN=EEE
    JWT_SECRET=vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM
    ADMINS=admin1@email.com,admin2@email.com
    ```

3.  Put the excel spreadsheet with all cohorts metadata in `data/iCARE4CVD_Cohorts.xlsx`. Uploaded cohorts will go to separated folders in `data/cohorts/`

> \[!WARNING]
>
> There is a bug with pandas when conditional cells are used in the excel spreadsheet. To remove conditional cells copy the whole sheet content, delete the current content, and paste the original sheet content without formatting (ctrl+shift+v)

> \[!IMPORTANT]
>
> For the authentication to the Decentriq OAuth provider to work you need to deploy the backend on <http://localhost:3000>

### ⚡ Start for development

In development it is more convenient to start all components like the database with docker, and the backend/frontend outside of docker.

Start the database with docker:

```bash
docker compose up db
```

In a different terminal, start the backend:

```bash
cd backend
hatch run dev
```

In another terminal, start the frontend:

```bash
cd frontend
pnpm dev
```

> \[!TIP]
>
> Alternatively you can start the whole stack in development with docker compose in 1 command, but you won't get hot reload for the frontend (you will need to rebuild the frontend image for changes to be taken into account):
>
> ```bash
> docker compose up
> ```

### 🧹 Code formatting and linting

Automatically format Python code with ruff and black, and TypeScript code with prettier.

```bash
./scripts/fmt.sh
```

## 🐳 Deploy

Deploy on a server in production with docker compose.

Put the excel spreadsheet with all cohorts metadata in `data/iCARE4CVD_Cohorts.xlsx`. Uploaded cohorts will go to separated folders in `data/cohorts/`

Generate a secret key used to encode/decode JWT token for a secure authentication system:

```bash
python -c "import secrets ; print(secrets.token_urlsafe(32))"
```

Create a `.env` file with secret configuration:

```bash
AUTH_ENDPOINT=https://auth0.com
CLIENT_ID=AAA
CLIENT_SECRET=BBB
DECENTRIQ_EMAIL=ccc@ddd.com
DECENTRIQ_TOKEN=EEE
JWT_SECRET=vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM
ADMINS=admin1@email.com,admin2@email.com
```

Deploy:

```bash
docker compose -f docker-compose.prod.yml up -d
```

## 🪄 Administration

### ✨ Automatically generate variables metadata

You can use the [`csvw-ontomap`](https://github.com/vemonet/csvw-ontomap) python package to automatically generate a CSV metadata file for your data file, with the format expected by iCARE4CVD. It will automatically fill the following columns: var name, var type, categorical, min, max

Install the package:

```bash
pip install git+https://github.com/vemonet/csvw-ontomap.git
```

Run profiling, supports `.csv`, `.xlsx`, `.sav`:

```bash
csvw-ontomap data/COHORT_data.sav -o data/COHORT_datadictionary.csv
```

### 🗑️ Reset database

Reset the database by deleting the `data/db` folder:

```bash
rm -rf data/db
```

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
