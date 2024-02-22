# 🫀 iCare4CVD Cohort Explorer

Webapp built for the [iCare4CVD project](https://icare4cvd.eu).

It aims to enable data owners and data scientists to:

* Login with their Decentriq account (OAuth based authentication, can be easily switch to other providers)
* Upload CSV or excel cohort metadata files describing the variables of a study cohort (tabular data file)
* Explore metadata about cohorts
* Map each variable of each cohorts to standard concepts, sourced from OHDSI Athena API (SNOMEDCT, LOINC...)
* Add cohorts to a "cart"
* Create a Data Clean Room in [Decentriq](https://www.decentriq.com/) with the cohorts added to the cart

## 🗺️ Architecture

* `backend/` built with python and FastAPI
* `frontend/` built with TypeScript, NextJS, ReactJS, TailwindCSS, and DaisyUI
* Triplestore providing a SPARQL endpoint: Oxigraph

Everything deployed in docker containers using docker compose.

## ☑️ To do

- [ ] Improve validation of uploaded data dictionaries
- [ ] Use a persistent triplestore
- [ ] Save mappings to triplestore
- [ ] Improve classes and predicates currently used to describe cohorts metadata files
- [ ] Add faceted search with filters for various fields of the cohorts metadata, and search for cohorts including searched variables
- [ ] Add LUCE blockchain component (as a service in the `docker-compose.yml`)

## 🧑‍💻 Development

### 📥 Install dependencies

1. Install [hatch](https://hatch.pypa.io/latest/) for managing python projects, and [pnpm](https://pnpm.io/installation) for TS/JS projects

   ```bash
   pip install hatch
   ```

2. Create a `backend/.env` file with secret configuration:

   ```bash
   AUTH_ENDPOINT=https://auth0.com
   CLIENT_ID=AAA
   CLIENT_SECRET=BBB
   DECENTRIQ_EMAIL=ccc@ddd.com
   DECENTRIQ_TOKEN=EEE
   JWT_SECRET=vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM
   ```

3. Put the spreadsheet with all cohorts metadata in `data/iCARE4CVD_Cohorts.csv`. Uploaded cohorts will go to `data/cohorts`

### ⚡ Start for development

Start the backend:

```bash
cd backend
hatch run dev
```

In another terminal, start the frontend:

```bash
cd frontend
pnpm dev
```

### 🧹 Code formatting and linting

```bash
cd backend
hatch run fmt

cd ../frontend
pnpm fmt
```

## 🐳 Deploy

Deploy on a server in production with docker compose

Generate a secret key used to encode/decode JWT token:

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
```

Deploy:

```bash
docker compose up -d
```
