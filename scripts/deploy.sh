#!/bin/bash

if [ "$1" = "--no-cache" ]; then
    echo "📦️ Building without cache"
    ssh ids1 'cd /data/deploy-services/cohort-explorer ; git pull ; docker compose -f docker-compose.prod.yml build --no-cache ; docker-compose -f docker-compose.prod.yml down ; docker compose -f docker-compose.prod.yml up --force-recreate -d'
else
    echo "♻️ Building with cache"
    ssh ids1 'cd /data/deploy-services/cohort-explorer ; git pull ; docker compose -f docker-compose.prod.yml up --force-recreate --build -d'
fi
