
import pandas as pd
# import numpy as np
import os
from CohortVarLinker.src.utils import load_dictionary
import csv

CASE_SENSITIVE_COLS = {'concept_code', 'concept_code_1', 'concept_code_2'}

def lower_except_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # A unit's name IS its symbol, so keep UCUM names too - captured before the
    # vocabulary column itself gets flattened.
    ucum_names = None
    if {'vocabulary_id', 'concept_name'} <= set(df.columns):
        ucum_names = df.loc[
            df['vocabulary_id'].astype(str).str.lower() == 'ucum', 'concept_name'
        ]
    lower = [c for c in df.columns if c not in CASE_SENSITIVE_COLS]
    if lower:
        df[lower] = df[lower].apply(lambda x: x.astype(str).str.lower())
    for c in set(df.columns) & CASE_SENSITIVE_COLS:
        df[c] = df[c].astype(str)
    if ucum_names is not None and len(ucum_names):
        df.loc[ucum_names.index, 'concept_name'] = ucum_names.astype(str)
    return df

def ucum_hierarchy(concept_df: pd.DataFrame, relationship_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds parent SNOMED unit concept relationships to UCUM unit concepts.
    """
    # Create a clean copy to avoid SettingWithCopyWarning
    concept_df = concept_df.copy()
    # Ensure types for merging
    concept_df["concept_id"] = concept_df["concept_id"].astype(str)
    concept_df["vocabulary_id"] = concept_df["vocabulary_id"].astype(str).str.lower()
    parent_id = "35624217"  # SNOMED 'unit of measure' concept_id
    parent_rows = concept_df[concept_df["concept_id"] == parent_id]
    if parent_rows.empty:
        print(f"[WARNING] Parent concept_id {parent_id} (SNOMED Unit) not found. Skipping UCUM hierarchy.")
        return relationship_df
    
    parent_row = parent_rows.iloc[0]
    parent_name = parent_row["concept_name"]
    parent_code = parent_row["concept_code"]
    parent_vocab = parent_row["vocabulary_id"]
    parent_domain = parent_row["domain_id"]
    parent_class = parent_row["concept_class_id"]

    # --- All UCUM concepts ---
    ucum_df = concept_df[concept_df["vocabulary_id"] == "ucum"].copy()

    if ucum_df.empty:
        print("[WARNING] No UCUM concepts found. Skipping UCUM hierarchy.")
        return relationship_df

    # --- Build the new relationship rows ---
    new_rel = pd.DataFrame({
        "concept_id_1": ucum_df["concept_id"],
        "concept_id_2": parent_id,
        "relationship_id": "is a",
        "concept_name_1": ucum_df["concept_name"],
        "concept_name_2": parent_name,
        "concept_vocabulary_1": ucum_df["vocabulary_id"],
        "concept_vocabulary_2": parent_vocab,
        "concept_domain_1": ucum_df["domain_id"],
        "concept_domain_2": parent_domain,
        "concept_class_1": ucum_df["concept_class_id"],
        "concept_class_2": parent_class,
        "concept_code_1": ucum_df["concept_code"],
        "concept_code_2": parent_code,
        # Initialize synonyms as empty if not present, will be filled later or handled by caller
        "concept_synonym_1": ucum_df["synonyms"] if "synonyms" in ucum_df.columns else "",

        "concept_synonym_2": parent_row.get("synonyms", "")
    })
    

    if relationship_df is not None:
        # Align columns
        combined = pd.concat([relationship_df, new_rel], axis=0, ignore_index=True)
        return combined
    return new_rel

def enrich_concept_synonyms(concept_df, concept_syn_file):
    print("Loading synonyms...")
    concept_syn = pd.read_csv(concept_syn_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE)
    concept_syn = concept_syn.apply(lambda x: x.astype(str).str.lower())
    concept_syn.columns = concept_syn.columns.str.lower()
    # Filter for English (4180186)
    eng_syns = concept_syn[concept_syn['language_concept_id'] == '4180186']

    # Group synonyms by concept_id
    grouped_syns = eng_syns.groupby('concept_id')['concept_synonym_name'].apply(
        lambda x: ";".join(pd.unique(x).astype(str))
    ).reset_index()
    grouped_syns.rename(columns={'concept_synonym_name': 'synonyms'}, inplace=True)

    # Merge back to concept_df
    concept_df = concept_df.merge(grouped_syns, on="concept_id", how="left")
    
    # Fill NaN synonyms with empty string
    concept_df['synonyms'] = concept_df['synonyms'].fillna("")
    
    return concept_df

def add_snomed_atc_equivalence_viarxnorm_v1(concept_file, concept_syn_file, relationship_file, output_file):
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE,
        usecols=['concept_id', 'concept_name', 'vocabulary_id', 'domain_id',
                 'concept_class_id', 'concept_code', 'invalid_reason'],)
    concept_df = lower_except_codes(concept_df)
    concept_df = concept_df[concept_df['invalid_reason'].isin(['nan', ''])]
    concept_df.drop(columns=['invalid_reason'], inplace=True)
    concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)

    relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE)
    relationship_df.columns = relationship_df.columns.str.lower()
    relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())
    # Filter out deprecated relationships
    if 'invalid_reason' in relationship_df.columns:
        relationship_df = relationship_df[relationship_df['invalid_reason'].isin(['nan',''])]

    # Also filter: both concept_ids must be valid concepts
    valid_ids = set(concept_df['concept_id'])
    snomed_rxnorm_df = relationship_df[
        (relationship_df['relationship_id'] == 'snomed - rxnorm eq') &
        (relationship_df['concept_id_1'].isin(valid_ids)) &
        (relationship_df['concept_id_2'].isin(valid_ids))
    ]
    rxnorm_atc_df = relationship_df[
        (relationship_df['relationship_id'] == 'rxnorm - atc pr lat') &
        (relationship_df['concept_id_1'].isin(valid_ids)) &
        (relationship_df['concept_id_2'].isin(valid_ids))
    ]

    merged_df = snomed_rxnorm_df.merge(
        rxnorm_atc_df, left_on="concept_id_2", right_on="concept_id_1",
        how="inner", suffixes=('_snomed_rxnorm', '_rxnorm_atc'))

    rel_1 = pd.DataFrame({
        'concept_id_1': merged_df['concept_id_1_snomed_rxnorm'],
        'concept_id_2': merged_df['concept_id_2_rxnorm_atc'],
        'relationship_id': 'snomed - atc eq'
    })
    rel_2 = pd.DataFrame({
        'concept_id_1': rel_1['concept_id_2'].values,
        'concept_id_2': rel_1['concept_id_1'].values,
        'relationship_id': 'atc - snomed eq'
    })
    final_rels = pd.concat([rel_1, rel_2], ignore_index=True)

    # Attach metadata (same as original, but concept_df is now filtered)
    final_rels = final_rels.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
    final_rels.drop(columns=["concept_id"], inplace=True)
    final_rels.rename(columns={
        "concept_name": "concept_name_1", "vocabulary_id": "concept_vocabulary_1",
        "domain_id": "concept_domain_1", "concept_class_id": "concept_class_1",
        "concept_code": "concept_code_1", "synonyms": "concept_synonym_1"
    }, inplace=True)
    final_rels = final_rels.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
    final_rels.drop(columns=["concept_id"], inplace=True)
    final_rels.rename(columns={
        "concept_name": "concept_name_2", "vocabulary_id": "concept_vocabulary_2",
        "domain_id": "concept_domain_2", "concept_class_id": "concept_class_2",
        "concept_code": "concept_code_2", "synonyms": "concept_synonym_2"
    }, inplace=True)
    final_rels.to_csv(output_file, index=False)

# def add_snomed_loinc_equivalence_via_cpt4(concept_file, concept_syn_file, relationship_file, output_file):
#     """Derive LOINC <-> SNOMED equivalence by composing through CPT-4.

#     `SNOMED - LOINC eq` and `LOINC - SNOMED eq` are declared in RELATIONSHIP.csv
#     (concept ids 45956696 / 45956695) but carry ZERO instances in the 2026-08-05
#     Athena release, so a LOINC lab concept and its SNOMED counterpart have no
#     direct bridge. Both legs of a CPT-4 detour do exist -- `LOINC - CPT4 eq`
#     (2,019 rows) and `CPT4 - SNOMED eq` (15,055) -- and composing two asserted
#     equivalences is itself an equivalence, so the detour is sound where it lands.

#     Mirrors add_snomed_atc_equivalence_viarxnorm_v1, which composes
#     `snomed - rxnorm eq` with `rxnorm - atc pr lat` for the drug side.

#     Note the ceiling: only 2,019 LOINC concepts have a CPT-4 equivalent at all,
#     so this recovers a small slice of LOINC rather than the full bridge the
#     missing relation would have given.
#     """
#     concept_df = pd.read_csv(concept_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE,
#         usecols=['concept_id', 'concept_name', 'vocabulary_id', 'domain_id',
#                  'concept_class_id', 'concept_code', 'invalid_reason'],)
#     concept_df = concept_df.apply(lambda x: x.astype(str).str.lower())
#     concept_df = concept_df[concept_df['invalid_reason'].isin(['nan', ''])]
#     concept_df.drop(columns=['invalid_reason'], inplace=True)
#     concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)

#     relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE)
#     relationship_df.columns = relationship_df.columns.str.lower()
#     relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())
#     if 'invalid_reason' in relationship_df.columns:
#         relationship_df = relationship_df[relationship_df['invalid_reason'].isin(['nan', ''])]

#     valid_ids = set(concept_df['concept_id'])
#     loinc_cpt4_df = relationship_df[
#         (relationship_df['relationship_id'] == 'loinc - cpt4 eq') &
#         (relationship_df['concept_id_1'].isin(valid_ids)) &
#         (relationship_df['concept_id_2'].isin(valid_ids))
#     ]
#     cpt4_snomed_df = relationship_df[
#         (relationship_df['relationship_id'] == 'cpt4 - snomed eq') &
#         (relationship_df['concept_id_1'].isin(valid_ids)) &
#         (relationship_df['concept_id_2'].isin(valid_ids))
#     ]
#     print(f"LOINC->CPT4 legs: {len(loinc_cpt4_df):,}, CPT4->SNOMED legs: {len(cpt4_snomed_df):,}")

#     merged_df = loinc_cpt4_df.merge(
#         cpt4_snomed_df, left_on="concept_id_2", right_on="concept_id_1",
#         how="inner", suffixes=('_loinc_cpt4', '_cpt4_snomed'))

#     rel_1 = pd.DataFrame({
#         'concept_id_1': merged_df['concept_id_1_loinc_cpt4'],
#         'concept_id_2': merged_df['concept_id_2_cpt4_snomed'],
#         'relationship_id': 'loinc - snomed eq'
#     })
#     rel_2 = pd.DataFrame({
#         'concept_id_1': rel_1['concept_id_2'].values,
#         'concept_id_2': rel_1['concept_id_1'].values,
#         'relationship_id': 'snomed - loinc eq'
#     })
#     final_rels = pd.concat([rel_1, rel_2], ignore_index=True)
#     # One CPT-4 concept can bridge several LOINC/SNOMED concepts, so the compose
#     # can repeat a pair; a duplicated edge is harmless but inflates the file.
#     final_rels.drop_duplicates(inplace=True)
#     print(f"Derived {len(final_rels):,} LOINC<->SNOMED equivalence rows "
#           f"({final_rels['concept_id_1'].nunique():,} distinct sources)")

#     final_rels = final_rels.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
#     final_rels.drop(columns=["concept_id"], inplace=True)
#     final_rels.rename(columns={
#         "concept_name": "concept_name_1", "vocabulary_id": "concept_vocabulary_1",
#         "domain_id": "concept_domain_1", "concept_class_id": "concept_class_1",
#         "concept_code": "concept_code_1", "synonyms": "concept_synonym_1"
#     }, inplace=True)
#     final_rels = final_rels.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
#     final_rels.drop(columns=["concept_id"], inplace=True)
#     final_rels.rename(columns={
#         "concept_name": "concept_name_2", "vocabulary_id": "concept_vocabulary_2",
#         "domain_id": "concept_domain_2", "concept_class_id": "concept_class_2",
#         "concept_code": "concept_code_2", "synonyms": "concept_synonym_2"
#     }, inplace=True)
#     final_rels.to_csv(output_file, index=False)
#     print(f"Saved LOINC<->SNOMED equivalence to: {output_file}")


def enrich_relationships_final(concept_file,cpt4_concepts, concept_syn_file, relationship_file, icarecvd_vocab_df, output_file):
    print("Enriching Relationships...")
    
    include_relationships = [
     
        'is a','subsumes',
        'rxnorm - atc pr lat', 'atc - rxnorm pr lat',
        # 'rxnorm - atc','atc - rxnorm', (too broad and include combinational drugs which can cause false positive mappings)
        'atc - snomed eq', 'snomed - atc eq',
        'disposition of' ,'has disposition',
        "has answer", "answer of", 
        'cpt4 - snomed eq', 'snomed - cpt4 eq', 
        'cpt4 - loinc eq' , 'loinc - cpt4 eq',
        'snomed - rxnorm eq','rxnorm - snomed eq',
        'snomed - loinc eq', 'loinc - snomed eq', # Added missing ones from your list
        'mapped from', 'maps to', # Important for standard maps
        # 'value mapped from', 'maps to value', 
        # 'brand name of', 'has brand name', 
        'finding site of', 'has finding site',
        # 'has associated morphology (snomed)', 'associated morphology of (snomed)',
        # 'has causative agent (snomed)', 'causative agent of (snomed)',
        # 'has etiology (snomed)', 'etiology of (snomed)',
        'has ingredient (snomed)', 'ingredient of (snomed)',

       
         
    ]

    loin_specific_relationships = [
         'has component', 'component of',
         'has method', 'method of',  #these two for LOINC hierarchy
         'has specimen', 'specimen of',  #these two for LOINC hierarchy
         'has property', 'property of',  #these two for LOINC hierarchy
         'has scale type', 'scale type of',  #these two for LOINC hierarchy
         'has system', 'system of',  #these two for LOINC hierarchy
         'has time aspect', 'time aspect of'  #these two for LOINC hierarchy
    ] # add these where concept_class_id.lower() is 'lab test'
    
    # Load Concept and Synonyms
    # Athena ships tab-separated files whose concept_name may contain a bare
    # double quote. With pandas' default quoting that quote opens a quoted field
    # and everything up to the next one is swallowed: on the 2026-08-05 release
    # this silently dropped 178,377 of 4,477,269 concepts, taking their
    # relationship rows with them via the inner joins below. The loss is
    # invisible -- no warning, no error, just a shorter file -- so QUOTE_NONE is
    # required on every Athena read.
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE,
                             usecols=['concept_id', 'concept_name', 'vocabulary_id', 'domain_id', 'concept_class_id', 'concept_code', 'invalid_reason'])
    concept_df = lower_except_codes(concept_df)
    cpt_concept_df = pd.read_csv(cpt4_concepts, sep='\t', dtype=str, quoting=csv.QUOTE_NONE,
                                 usecols=['concept_id', 'concept_name', 'vocabulary_id', 'domain_id', 'concept_class_id', 'concept_code', 'invalid_reason'])
    cpt_concept_df = lower_except_codes(cpt_concept_df)
    # Filter valid concepts only (or Nan)
    concept_df = concept_df[concept_df['invalid_reason'].isin(['nan', ''])] 
    cpt_concept_df = cpt_concept_df[cpt_concept_df['invalid_reason'].isin(['nan', ''])] 

    concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)
    concept_df.drop(columns=['invalid_reason'], inplace=True)

    cpt_concept_df = enrich_concept_synonyms(cpt_concept_df, concept_syn_file)
    cpt_concept_df.drop(columns=['invalid_reason'], inplace=True)

    # merge concept_df and cpt_concept_df
    concept_df = pd.concat([concept_df,cpt_concept_df],axis=0, ignore_index=True)
    concept_df.drop_duplicates(subset='concept_id', keep='first', inplace=True)


    # Load Relationships
    rel_df = pd.read_csv(relationship_file, sep='\t', dtype=str, quoting=csv.QUOTE_NONE)
    rel_df.columns = rel_df.columns.str.lower()
    rel_df = rel_df.apply(lambda x: x.astype(str).str.lower())
    
    lab_test_ids = set(concept_df[concept_df['concept_class_id'] == 'lab test']['concept_id'])
    print(f"Total lab test concepts: {len(lab_test_ids)}")
    # Filter Relationships
    # rel_df_wi = rel_df[rel_df['relationship_id'].isin([r.lower() for r in include_relationships])]
    # print(f"Relationships after initial filter: {len(rel_df_wi)}")
    rel_df = rel_df[
    (rel_df['relationship_id'].isin([r.lower() for r in include_relationships])) |
    (rel_df['relationship_id'].isin([r.lower() for r in loin_specific_relationships]) &
     rel_df['concept_id_1'].isin(lab_test_ids))
]
    
    print(f"Relationships after LOINC-specific filter: {len(rel_df)}")
    rel_df = rel_df[rel_df['concept_id_1'] != rel_df['concept_id_2']] # Remove self-loops in raw data

    # Merge Metadata 1
    rel_df = rel_df.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="inner")
    rel_df.drop(columns=["concept_id", "invalid_reason", "valid_start_date", "valid_end_date"], inplace=True, errors='ignore')
    rel_df.rename(columns={
        "concept_name": "concept_name_1", "vocabulary_id": "concept_vocabulary_1",
        "domain_id": "concept_domain_1", "concept_class_id": "concept_class_1",
        "concept_code": "concept_code_1", "synonyms": "concept_synonym_1"
    }, inplace=True)

    # Merge Metadata 2
    rel_df = rel_df.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="inner")
    rel_df.drop(columns=["concept_id", "invalid_reason", "valid_start_date", "valid_end_date"], inplace=True, errors='ignore')
    rel_df.rename(columns={
        "concept_name": "concept_name_2", "vocabulary_id": "concept_vocabulary_2",
        "domain_id": "concept_domain_2", "concept_class_id": "concept_class_2",
        "concept_code": "concept_code_2", "synonyms": "concept_synonym_2"
    }, inplace=True)

 
    if icarecvd_vocab_df is not None and not icarecvd_vocab_df.empty:
        # Ensure icarecvd_vocab_df has compatible columns or clean them
        print("Merging iCare4CVD vocabulary...")
        icarecvd_vocab_df = lower_except_codes(icarecvd_vocab_df)
        rel_df = pd.concat([rel_df, icarecvd_vocab_df], axis=0, ignore_index=True)

    # Add UCUM Hierarchy
    rel_df = ucum_hierarchy(concept_df, rel_df)

    # Final Cleanup: Remove rows with missing vocabularies
    initial_len = len(rel_df)
    # rel_df = rel_df.dropna(subset=['concept_vocabulary_1', 'concept_vocabulary_2'])
    rel_df = rel_df[
    rel_df['concept_vocabulary_1'].notna() & (rel_df['concept_vocabulary_1'] != 'nan') & (rel_df['concept_vocabulary_1'] != '') &
    rel_df['concept_vocabulary_2'].notna() & (rel_df['concept_vocabulary_2'] != 'nan') & (rel_df['concept_vocabulary_2'] != '')
]
    
    print(f"Dropped {initial_len - len(rel_df)} rows due to missing vocabularies.")
    
    rel_df.to_csv(output_file, index=False)
    print(f"Enriched file saved as: {output_file}")
 

if __name__ == "__main__":
    
    # Update these paths as needed
    athena_vocab_dir = "/Users/komalgilani/Downloads/vocabulary_05082026"
    output_dir = "/Users/komalgilani/phd_projects/CohortVarLinker/data"
    
    # 1. Load iCare4CVD
    icare_file = f"{output_dir}/cvd_vocab.csv"
    if os.path.exists(icare_file):
        df_icare = load_dictionary(icare_file)
        print(df_icare.head(2))
    else:
        print("Warning: iCare4CVD vocabulary file not found. Proceeding without it.")
        df_icare = pd.DataFrame() # Empty DF
    
   
    # # # 2. Generate SNOMED-ATC Equivalence
    add_snomed_atc_equivalence_viarxnorm_v1(
        concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
       
        concept_syn_file=f"{athena_vocab_dir}/CONCEPT_SYNONYM.csv",
        relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
        output_file=f"{output_dir}/concept_relationship_snomed_atc_equivalence_only.csv",
    )
    
 
    
    # 2b. Generate LOINC-SNOMED equivalence by composing through CPT-4.
    # SNOMED - LOINC eq / LOINC - SNOMED eq are declared in RELATIONSHIP.csv but
    # have zero instances in this release, so the direct bridge is unavailable.
    # add_snomed_loinc_equivalence_via_cpt4(
    #     concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
    #     concept_syn_file=f"{athena_vocab_dir}/CONCEPT_SYNONYM.csv",
    #     relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
    #     output_file=f"{output_dir}/concept_relationship_loinc_snomed_equivalence_only.csv",
    # )

    # 3. Enrich Relationships
    enrich_relationships_final(
        concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
        cpt4_concepts= f"{athena_vocab_dir}/CONCEPT_CPT4.csv",
        concept_syn_file=f"{athena_vocab_dir}/CONCEPT_SYNONYM.csv",
        relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
        icarecvd_vocab_df=df_icare,
        output_file=f"{output_dir}/concept_relationship_enriched.csv",
    )

    # 4. Final Combination Step (Merging the generated equivalence with the enriched file)
    print("Combining Equivalence and Enriched files...")
    snomed_atc_df = pd.read_csv(f"{output_dir}/concept_relationship_snomed_atc_equivalence_only.csv", dtype=str)
    # loinc_snomed_df = pd.read_csv(f"{output_dir}/concept_relationship_loinc_snomed_equivalence_only.csv", dtype=str)
    enrich_df = pd.read_csv(f"{output_dir}/concept_relationship_enriched.csv", dtype=str)
    # print(f"enriched: {len(enrich_df):,}, snomed-atc: {len(snomed_atc_df):,}, loinc-snomed: {len(loinc_snomed_df):,}")

    combined_df = pd.concat([enrich_df, snomed_atc_df], axis=0, ignore_index=True)
    # combined_df = enrich_df
    # Deduplicate
    before_dedup = len(enrich_df)
    enrich_df.drop_duplicates(inplace=True)
    print(f"Removed {before_dedup - len(combined_df)} duplicates.")

    # Remove rows with missing IDs or Vocabs
    # REPLACE the dropna + notna block with a single filter:
    combined_df = combined_df[
        combined_df['concept_id_1'].notna() & combined_df['concept_id_2'].notna() &
        combined_df['concept_vocabulary_1'].notna() & (combined_df['concept_vocabulary_1'] != '') &
        combined_df['concept_vocabulary_2'].notna() & (combined_df['concept_vocabulary_2'] != '')
    ]
    
    # Final Save
   
    # combined_df = combined_df[
    #     combined_df['concept_vocabulary_1'].notna() & (combined_df['concept_vocabulary_1'] != '') &
    #     combined_df['concept_vocabulary_2'].notna() & (combined_df['concept_vocabulary_2'] != '')
    # ]
    print("Total rows in final enriched file after removing missing vocabularies")
    print(f"{combined_df.head(5)}") 
    unique_vocab_1 = combined_df['concept_vocabulary_1'].nunique()
    unique_vocab_2 = combined_df['concept_vocabulary_2'].nunique()
    print(f"Unique vocabularies in concept_vocabulary_1: {unique_vocab_1}")
    print(f"Unique vocabularies in concept_vocabulary_2: {unique_vocab_2}")
    combined_df.to_csv(f"{output_dir}/concept_relationship_enriched.csv", index=False)
    print(f"After removing missing vocabularies, final enriched file saved as: {output_dir}/concept_relationship_enriched.csv")
   
