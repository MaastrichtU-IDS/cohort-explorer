# SPARQL Query Compatibility Analysis
## Comparing queries in sparql_queries.txt with graph construction in upload.py

Date: 2025-10-22

---

## 1. ✅ Study Type / Descriptor - COMPATIBLE

### Query Pattern (lines 45-49):
```sparql
?study dc:identifier ?cohortId ;
       iao:is_about ?descriptor ;
       ro:has_part ?protocol .
?descriptor a sio:descriptor;
       rdfs:label ?study_type.
```

### Graph Construction (upload.py line 919, 990-1004):
```python
"study type": {"ns": "SIO", "type": "descriptor", "target": "study_design", "rel": "is_about", "rel_ns": "IAO", "use_rdfs_label": True}

# Creates:
# study_design_uri iao:is_about descriptor_uri
# descriptor_uri rdf:type sio:descriptor
# descriptor_uri rdfs:label "study type value"
```

**Status**: ✅ COMPATIBLE
- Uses `sio:descriptor` type
- Uses `iao:is_about` relationship
- Adds `rdfs:label` with the value

---

## 2. ⚠️ ISSUE: Ongoing Status - QUERY EXPECTS DIFFERENT RELATIONSHIP

### Query Pattern (lines 107-109):
```sparql
?completion_status a sio:ongoing ;
             iao:is_about ?sde ;
              cmeo:has_value ?study_ongoing .
```
**Query expects**: `iao:is_about` connecting ongoing status to study design execution

### Graph Construction (upload.py line 924):
```python
"ongoing": {"ns": "SIO", "type": "ongoing", "target": "execution", "datatype": "boolean"}

# Creates:
# study_design_execution_uri ro:has_part ongoing_uri
# ongoing_uri rdf:type sio:ongoing
# ongoing_uri cmeo:has_value "true/false"
```
**Graph creates**: `ro:has_part` relationship from execution to ongoing status

**Status**: ❌ INCOMPATIBLE
**Issue**: Query expects `iao:is_about` but graph uses `ro:has_part`

### Fix Required:
```python
"ongoing": {"ns": "SIO", "type": "ongoing", "target": "execution", "rel": "is_about", "rel_ns": "IAO", "datatype": "boolean"}
```

---

## 3. ⚠️ ISSUE: Population Location - MISSING RELATIONSHIP

### Query Pattern (lines 233-234):
```sparql
?site iao:is_about ?output_population ; 
      cmeo:has_value ?population_location .
```
**Query expects**: `site` entity with `iao:is_about` pointing to output population

### Graph Construction (upload.py line 926):
```python
"population location": {"ns": "BFO", "type": "site", "target": "execution"}

# Creates:
# study_design_execution_uri ro:has_part site_uri
# site_uri rdf:type bfo:site
# site_uri cmeo:has_value "location"
```
**Graph creates**: `ro:has_part` from execution to site, NO `iao:is_about` to population

**Status**: ❌ INCOMPATIBLE
**Issue**: Query expects site to point to `output_population`, but graph doesn't create this relationship

### Fix Required:
This field needs special handling in `handle_special_fields()` to:
1. Create eligibility criterion
2. Create output population entity
3. Link site to output population with `iao:is_about`

---

## 4. ✅ Timeline Specification - COMPATIBLE

### Query Pattern (lines 70-73):
```sparql
?protocol ro:has_part ?dcfrequency . 
?dcfrequency a cmeo:timeline_specification ; 
            cmeo:has_value ?data_collection_frequency .
```

### Graph Construction (upload.py line 933):
```python
"frequency of data collection": {"ns": "CMEO", "type": "timeline_specification", "target": "protocol"}

# Creates:
# protocol_uri ro:has_part timeline_spec_uri
# timeline_spec_uri rdf:type cmeo:timeline_specification
# timeline_spec_uri cmeo:has_value "frequency value"
```

**Status**: ✅ COMPATIBLE

---

## 5. ⚠️ ISSUE: Interventions - MISSING rdfs:label in Query Pattern

### Query Pattern (lines 77-79):
```sparql
?protocol ro:has_part ?intervention_spec.
?intervention_spec a cmeo:intervention_specification ; 
                   cmeo:has_value ?interventions .
```
**Query does NOT retrieve**: `rdfs:label`

### Graph Construction (upload.py line 934):
```python
"interventions": {"ns": "CMEO", "type": "intervention_specification", "target": "protocol", "label": True}

# Creates:
# protocol_uri ro:has_part intervention_uri
# intervention_uri rdf:type cmeo:intervention_specification
# intervention_uri rdfs:label "interventions"  # <-- ADDED but query doesn't use it
# intervention_uri cmeo:has_value "intervention value"
```

**Status**: ⚠️ COMPATIBLE but label is unused
**Note**: Adding `rdfs:label` is fine (doesn't break query), but query doesn't retrieve it

---

## 6. ✅ Contact Person / Administrator - COMPATIBLE

### Query Pattern (lines 120-138):
```sparql
?admin a ncbi:homo_sapiens ;
       obi:member_of ?organization;
       cmeo:has_value ?administrator.

?contactperson a ncbi:homo_sapiens ;
               obi:member_of ?organization;
               cmeo:has_value ?study_contact_person.
```

### Graph Construction (upload.py lines 930-931):
```python
"administrator": {"ns": "NCBI", "type": "homo_sapiens", "target": "execution", "rel": "has_participant"}
"study contact person": {"ns": "NCBI", "type": "homo_sapiens", "target": "execution", "rel": "has_participant"}
```

**Status**: ⚠️ NEEDS VERIFICATION
**Issue**: Graph construction adds `ro:has_participant` from execution to person, but query expects:
- Person has type `ncbi:homo_sapiens` ✅
- Person has `obi:member_of` organization ❌ (not created in simple field config)

**Note**: These fields may need special handling to create the organization membership relationship.

---

## 7. ✅ Inclusion/Exclusion Criteria - COMPATIBLE

### Query Pattern (lines 243-279):
```sparql
?criterion a obi:inclusion_criterion;
    ro:has_part ?inclusion_criterion_sub.

?inclusion_criterion_sub rdfs:label ?inc_label .
?inclusion_criterion_sub cmeo:has_value ?val_raw .

?exclusion_criterion a obi:exclusion_criterion;
    ro:has_part ?exclusion_criterion_sub.

?exclusion_criterion_sub rdfs:label ?exc_label .
?exclusion_criterion_sub cmeo:has_value ?exc_val_raw .
```

### Graph Construction (upload.py lines 1104-1161):
```python
# Creates inclusion criterion container
inclusion_criterion_uri = URIRef(study_uri + "/inclusion_criterion")
g.add((inclusion_criterion_uri, RDF.type, OBI.inclusion_criterion))
g.add((inclusion_criterion_uri, RO.part_of, eligibility_criterion_uri))
g.add((eligibility_criterion_uri, RO.has_part, inclusion_criterion_uri))

# For each specific inclusion criterion:
g.add((col_inclusion_criteria_uri, RDF.type, dynamic_inclusion_criterion_type))
g.add((col_inclusion_criteria_uri, RO.part_of, inclusion_criterion_uri))
g.add((inclusion_criterion_uri, RO.has_part, col_inclusion_criteria_uri))
g.add((col_inclusion_criteria_uri, RDFS.label, Literal(col)))
g.add((col_inclusion_criteria_uri, CMEO.has_value, Literal(value)))
```

**Status**: ✅ COMPATIBLE
- Uses correct types: `obi:inclusion_criterion`, `obi:exclusion_criterion`
- Uses correct relationship: `ro:has_part`
- Includes `rdfs:label` and `cmeo:has_value`

---

## 8. ✅ Outcome Specifications - COMPATIBLE

### Query Pattern (lines 147-184):
```sparql
?outcome_spec a cmeo:outcome_specification ;
              ro:has_part ?prim_spec .
?prim_spec a cmeo:primary_outcome_specification ;
           cmeo:has_value ?raw .

?outcome_spec a cmeo:outcome_specification ;
              ro:has_part ?sec_spec .
?sec_spec a cmeo:secondary_outcome_specification ;
          cmeo:has_value ?raw .
```

### Graph Construction (upload.py lines 1067-1104):
```python
outcome_spec_uri = URIRef(study_uri + "/outcome_specification")
g.add((outcome_spec_uri, RDF.type, CMEO.outcome_specification))
g.add((protocol_uri, RO.has_part, outcome_spec_uri))

primary_uri = URIRef(study_uri + "/primary_outcome_specification")
g.add((primary_uri, RDF.type, CMEO.primary_outcome_specification))
g.add((outcome_spec_uri, RO.has_part, primary_uri))
g.add((primary_uri, CMEO.has_value, Literal(primary_value)))
```

**Status**: ✅ COMPATIBLE
- Correct nested structure: protocol → outcome_spec → primary/secondary_spec
- Uses correct types and relationships

---

## 9. ⚠️ ISSUE: Morbidity - MISSING in Graph Construction

### Query Pattern (lines 236-239):
```sparql
?morbidity_spec a obi:morbidity ;
      ro:is_characteristic_of ?output_population ;
      cmeo:has_value ?mv .
```
**Query expects**: Morbidity linked to output_population with `ro:is_characteristic_of`

### Graph Construction (upload.py line 929):
```python
"morbidity": {"ns": "OBI", "type": "morbidity", "target": "protocol"}

# Creates:
# protocol_uri ro:has_part morbidity_uri
# morbidity_uri rdf:type obi:morbidity
# morbidity_uri cmeo:has_value "morbidity value"
```
**Graph creates**: Morbidity linked to protocol, NOT to output_population

**Status**: ❌ INCOMPATIBLE
**Issue**: Query expects `ro:is_characteristic_of ?output_population` relationship

### Fix Required:
This needs special handling to:
1. Create output_population entity
2. Link morbidity with `ro:is_characteristic_of` instead of `ro:has_part`

---

## Summary of Issues

| Field | Issue | Severity | Fix Required |
|-------|-------|----------|--------------|
| **Ongoing Status** | Uses `ro:has_part` instead of `iao:is_about` | HIGH | Change relationship type |
| **Population Location** | Missing `output_population` entity and `iao:is_about` relationship | HIGH | Special handling needed |
| **Morbidity** | Linked to protocol instead of output_population | HIGH | Special handling needed |
| **Contact Person/Admin** | Missing `obi:member_of` relationship to organization | MEDIUM | Special handling needed |
| **Interventions** | Label added but not used by query | LOW | Optional - no fix needed |

---

## Recommended Fixes

### 1. Fix Ongoing Status Relationship
```python
"ongoing": {"ns": "SIO", "type": "ongoing", "target": "execution", "rel": "is_about", "rel_ns": "IAO", "datatype": "boolean"}
```

### 2. Add Special Handling for Eligibility-Related Fields
Create a dedicated function to handle:
- Population location (with output_population entity)
- Morbidity (with characteristic_of relationship)
- Mixed sex (linked to output_population)
- Age distribution (linked to output_population)

These all need to be part of the eligibility criterion → enrollment → output_population structure.

### 3. Enhance Contact Person/Administrator Handling
Ensure organization membership relationships are created properly.
