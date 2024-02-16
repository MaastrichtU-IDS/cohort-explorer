#!/bin/bash

if [ "$1" = "--no-cache" ]; then
    echo "📦️ Building without cache"
    ssh ids2 'cd /data/deploy-services/cohort-explorer ; git pull ; docker-compose build --no-cache ; docker-compose down ; docker-compose up --force-recreate -d'
else
    echo "♻️ Building with cache"
    ssh ids2 'cd /data/deploy-services/cohort-explorer ; git pull ; docker-compose up --force-recreate --build -d'
fi
