#!/bin/bash

# Dump triplestore RDF data to a file

docker compose exec backend curl -f -X GET -H 'Accept: application/n-quads' http://db:7878/store > data/triplestore_dump_$(date +%Y%m%d).nq
