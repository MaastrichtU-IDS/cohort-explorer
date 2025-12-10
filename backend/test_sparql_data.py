#!/usr/bin/env python3
"""
Quick diagnostic script to check what's in the triplestore
"""

from SPARQLWrapper import SPARQLWrapper, JSON
from src.config import settings

query_endpoint = SPARQLWrapper(settings.query_endpoint)
query_endpoint.setReturnFormat(JSON)
query_endpoint.setMethod('POST')

# Check what graphs exist
graphs_query = """
SELECT DISTINCT ?g
WHERE {
  GRAPH ?g { ?s ?p ?o }
}
"""

print("=== Checking available graphs ===")
query_endpoint.setQuery(graphs_query)
result = query_endpoint.query().convert()
graphs = [row["g"]["value"] for row in result["results"]["bindings"]]
print(f"Found {len(graphs)} graphs:")
for g in graphs:
    print(f"  - {g}")

# Check if studies_metadata graph has data
studies_check = """
PREFIX dc: <http://purl.org/dc/elements/1.1/>
SELECT (COUNT(*) as ?count)
WHERE {
  GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {
    ?s dc:identifier ?id
  }
}
"""

print("\n=== Checking studies_metadata graph ===")
query_endpoint.setQuery(studies_check)
result = query_endpoint.query().convert()
count = result["results"]["bindings"][0]["count"]["value"]
print(f"Found {count} studies with dc:identifier in studies_metadata graph")

# Check outcome specifications
outcome_check = """
PREFIX cmeo: <https://w3id.org/CMEO/>
PREFIX ro: <http://purl.obolibrary.org/obo/ro.owl/>
PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>

SELECT ?study ?outcome_type (COUNT(*) as ?count)
WHERE {
  GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {
    ?study ro:has_part ?p .
    ?p a obi:protocol ;
       ro:has_part ?outcome_spec .
    ?outcome_spec a cmeo:outcome_specification ;
                  ro:has_part ?spec .
    ?spec a ?outcome_type .
    FILTER(?outcome_type IN (cmeo:primary_outcome_specification, cmeo:secondary_outcome_specification))
  }
}
GROUP BY ?study ?outcome_type
"""

print("\n=== Checking outcome specifications structure ===")
query_endpoint.setQuery(outcome_check)
result = query_endpoint.query().convert()
if result["results"]["bindings"]:
    for row in result["results"]["bindings"]:
        print(f"  Study: {row['study']['value']}")
        print(f"  Type: {row['outcome_type']['value']}")
        print(f"  Count: {row['count']['value']}")
else:
    print("  No outcome specifications found with nested structure")

print("\n=== Done ===")
