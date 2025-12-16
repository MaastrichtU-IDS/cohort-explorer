#!/bin/bash
# Simple query to get ALL ViennaHF-Register triples

echo "=== All ViennaHF-Register Metadata ==="
echo ""

curl -s -X POST \
  -H "Content-Type: application/sparql-query" \
  -H "Accept: application/sparql-results+json" \
  --data "
SELECT ?subject ?predicate ?object
WHERE {
  GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {
    ?subject ?predicate ?object .
    FILTER(CONTAINS(LCASE(STR(?subject)), 'viennahf'))
  }
}
ORDER BY ?subject ?predicate
" \
  http://localhost:7878/query | \
  jq -r '.results.bindings[] | 
    "\(.subject.value | split("/")[-1] | .[0:40]) | \(.predicate.value | split("/")[-1] | split("#")[-1] | .[0:30]) | \(.object.value | .[0:100])"' | \
  column -t -s '|'
