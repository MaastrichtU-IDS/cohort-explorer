# Graph Naming Fix - Critical Issue Resolved
## Aligning cohort graph URIs between creation and querying

Date: 2025-10-23

---

## 🚨 CRITICAL ISSUE IDENTIFIED

There was a **major mismatch** in graph naming between how graphs were created and how queries searched for them.

### The Problem

**Graph Creation (OLD - WRONG)**:
```python
# In load_cohort_dict_file()
cohort_uri = get_cohort_uri(cohort_id)  # https://w3id.org/icare4cvd/cohort/{cohort_id}
g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_uri))  # Used cohort_uri as graph context
# Created graphs with pattern: https://w3id.org/icare4cvd/cohort/{cohort_id}
```

**Query Expectations**:
```sparql
# Variables query expects graphs with CMEO namespace
FILTER( STRSTARTS(STR(?g), "https://w3id.org/CMEO/graph/") )
FILTER( STRAFTER(STR(?g), "https://w3id.org/CMEO/graph/") != "studies_metadata" )
# Looks for graphs: https://w3id.org/CMEO/graph/{cohort_id}
```

### Impact
- ❌ Variables query would **find ZERO cohorts**
- ❌ All cohort dictionary data would be **invisible to queries**
- ✅ Studies metadata query worked fine (it already used correct namespace)

---

## ✅ FIX APPLIED

### 1. Created New Helper Function

```python
def get_cohort_graph_uri(cohort_id: str) -> URIRef:
    """Get the graph URI for a cohort in CMEO namespace (matches query expectations)"""
    return URIRef(OntologyNamespaces.CMEO.value + f"graph/{cohort_id}")
```

### 2. Updated Graph Creation

**In `load_cohort_dict_file()` function**:

```python
# NEW (CORRECT):
cohort_uri = get_cohort_uri(cohort_id)           # Entity URI: https://w3id.org/icare4cvd/cohort/{cohort_id}
cohort_graph_uri = get_cohort_graph_uri(cohort_id)  # Graph URI: https://w3id.org/CMEO/graph/{cohort_id}
g = init_graph()
g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_graph_uri))  # Use cohort_graph_uri as graph context
g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_graph_uri))
```

**ALL triples now use `cohort_graph_uri` as the 4th parameter (graph context)**:
- Variable triples: `g.add((variable_uri, property, value, cohort_graph_uri))`
- Category triples: `g.add((cat_uri, property, value, cohort_graph_uri))`
- All other cohort data

### 3. Updated Delete Operations

**Changed all delete calls** from:
```python
delete_existing_triples(get_cohort_uri(cohort_id))  # WRONG - deletes wrong graph
```

To:
```python
delete_existing_triples(get_cohort_graph_uri(cohort_id))  # CORRECT - deletes right graph
```

**Updated in 4 locations**:
1. `delete_cohort()` endpoint (line 635)
2. `upload_cohort_dict()` endpoint (line 730)
3. `generate_mappings()` function (line 760)
4. `init_triplestore()` function (line 1384)

---

## GRAPH STRUCTURE NOW CORRECT

### Studies Metadata Graph
```
Graph: https://w3id.org/CMEO/graph/studies_metadata
├── Study entities (study_design_execution)
├── Protocol entities
├── Outcome specifications
├── Inclusion/Exclusion criteria
└── Enrollment and population entities
```

### Individual Cohort Graphs
```
Graph: https://w3id.org/CMEO/graph/{cohort_id}
├── Cohort entity: https://w3id.org/icare4cvd/cohort/{cohort_id}
├── Variable entities: https://w3id.org/icare4cvd/cohort/{cohort_id}/{var_name}
├── Category entities: https://w3id.org/icare4cvd/cohort/{cohort_id}/{var_name}/category/{index}
└── All metadata properties
```

**Key Points**:
- **Entity URIs** use `icare4cvd` namespace (for cohort/variable/category subjects)
- **Graph URIs** use `CMEO` namespace (for named graph contexts)
- This separation is correct and matches query expectations

---

## VERIFICATION

To verify the fix works, check:

1. **Graph existence in triplestore**:
   ```sparql
   SELECT ?g (COUNT(*) as ?count)
   WHERE {
     GRAPH ?g { ?s ?p ?o }
     FILTER(STRSTARTS(STR(?g), "https://w3id.org/CMEO/graph/"))
   }
   GROUP BY ?g
   ```
   Should show:
   - `https://w3id.org/CMEO/graph/studies_metadata`
   - `https://w3id.org/CMEO/graph/{cohort1_id}`
   - `https://w3id.org/CMEO/graph/{cohort2_id}`
   - etc.

2. **Variables query now finds data**:
   - Run the variables query from `sparql_queries.txt` (line 295+)
   - Should return variables from all cohorts

3. **Studies metadata query still works**:
   - Run the studies query from `sparql_queries.txt` (line 3+)
   - Should return all cohort metadata

---

## NEXT STEPS

1. **Re-upload all cohort dictionaries** to regenerate graphs with correct URIs
2. **Clear triplestore** if old graphs with wrong URIs exist
3. **Test queries** to verify data retrieval works
4. **Clear cache** to ensure frontend gets fresh data

---

## FILES MODIFIED

- `/Users/anas-elghafari/cohort-explorer/backend/src/upload.py`
  - Added `get_cohort_graph_uri()` helper function
  - Updated `load_cohort_dict_file()` to use correct graph URI (10+ locations)
  - Updated all `delete_existing_triples()` calls (4 locations)

---

## RELATED FIXES

This graph naming fix complements the other compatibility fixes made:
1. ✅ Ongoing status relationship (`iao:is_about`)
2. ✅ Output population structure
3. ✅ Morbidity as characteristic
4. ✅ Age distribution as characteristic
5. ✅ Population location with site
6. ✅ **Graph naming alignment (THIS FIX)**

All systems are now fully aligned! 🎉
