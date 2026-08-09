**🧬 **CohortVarLinker**:** A Hybrid Semantic Matching Engine for Cross-Study Variable Harmonization

CohortVarLinker is a semantic matching tool designed to align and harmonize variables across heterogeneous clinical cohort studies. It leverages a hybrid approach that combines ontology-based reasoning with text-based semantic similarity (e.g., embeddings) to identify equivalent or related data elements between studies, even when they differ in naming conventions, granularity, or coding systems.

**🔧 Key Features:**

- Integration of domain ontologies (e.g., SNOMED CT, LOINC, RxNorm, ATC, CDISC, OMOP) for controlled vocabulary alignment
- Embedding-based semantic similarity to detect textual and contextual matches
- Support for categorical value normalization and unit mapping
- Compare Matched Variables across timelines (e.g. Visit numbers)
- Identification of partial, exact, and hierarchical variable matches
- Compatible with cohort metadata dictionaries and study-level documentation


**📌 Use Cases:**
- Harmonizing variable definitions across cardiovascular cohort studies
- Preparing study data for federated analysis or joint modeling
- Enabling semantic interoperability in cohort exploration tools
