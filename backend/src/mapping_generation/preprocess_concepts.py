import pandas as pd

import multiprocessing as mp
import numpy as np

from tqdm import tqdm

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # Ensures reproducibility

DATA_DIR = "/workspace/snomed_vocab"

STOP_WORDS = ['stop','start','combinations','combination','various combinations','various','left','right','blood','finding','finding status',
              'status','extra','point in time','pnt','oral','product','oral product','several','types','several types','random','nominal',
              'p time','quan','qual','quantitative','qualitative','ql','qn','quant','anti','antibodies','wb','whole blood','serum','plasma','diseases',
              'disorders','disorder','disease','lab test','measurements','lab tests','meas value','measurement','procedure','procedures',
              'panel','ordinal','after','before','survey','level','levels','others','other','p dose','dose','dosage','frequency','calc','calculation','qnt','quan','quant','point in time','quantitative',
              'cell','level','avg','average','qnt','qal','qul','plt','pln','test','tests','thrb','combinations combinations'
,'function','fn','fnc','pst','panl','ob','gyn','p birth','birth','pan','panel.','proc','procedure','device','stnd','standard','hydrate','mouse','ql','human','f','m','female','male','rat'
]

def check_concept_name_duplicates(concept_df):
    # Step 1: Identify duplicate concept_names that have both 'S' and Null
    duplicates = concept_df[concept_df.duplicated('concept_name', keep=False)]
    
    # Step 2: Separate rows with 'S', 'C', and Null standard_concept
    duplicates_s = duplicates[duplicates['standard_concept'] == 'S']
    duplicates_c = duplicates[duplicates['standard_concept'] == 'C']
    duplicates_null = duplicates[duplicates['standard_concept'].isnull()]
    
    # Step 3: Keep only the 'S' rows from duplicates that have Null counterparts
    duplicates_s_only = duplicates_s[duplicates_s['concept_name'].isin(duplicates_null['concept_name'])]
    
    # Step 4: Combine the filtered data with the original data excluding the duplicates and keep 'C' type values
    result_df = pd.concat([
        concept_df[~concept_df['concept_name'].isin(duplicates['concept_name'])],
        duplicates_s_only,
        duplicates_c,
        concept_df[~concept_df['concept_name'].isin(duplicates['concept_name']) & (concept_df['standard_concept'] == 'C')]
    ])
    
    # Remove any duplicates again to ensure clean data
    result_df = result_df.drop_duplicates(subset='concept_name', keep='first')
    
    return result_df
def is_mostly_english(text, threshold=0.8):
    """Check if a given text is mostly English based on language detection."""
    
    if pd.isnull(text):
        return False
    
    # Handle multiple synonyms separated by ;;
    synonyms = text.split(';;')
    total_synonyms = len(synonyms)
    english_synonyms = 0

    for synonym in synonyms:
        synonym = synonym.strip()
        if not synonym:
            continue
        try:
            lang = detect(synonym)
            if lang == 'en':
                english_synonyms += 1
        except LangDetectException:
            # If language detection fails (e.g., empty string), consider it non-English
            continue

    # Calculate the proportion of English synonyms
    if total_synonyms == 0:
        return False
    
    if (english_synonyms / total_synonyms) >= threshold:
        return True
    return False


def process_chunk(chunk):
    """Process a chunk of the synonyms DataFrame."""
    chunk['is_english'] = chunk['concept_synonym_name'].apply(is_mostly_english)
    return chunk[chunk['is_english']]

def fetch_concept(data_directory, only_standard=True, vocabulary_list=['Korean Revenue Code','Concept Class','Domain','Metadata','Vocabulary','Notes'],sep='\t',output_dir=DATA_DIR):
    concept_file = os.path.join(data_directory, 'CONCEPT.csv')
    relationship_file = os.path.join(data_directory, 'CONCEPT_RELATIONSHIP.csv')
    concept_df = update_non_standard_mapping(relationship_file,concept_file)
    print(f"UC ={concept_df.columns}")
    print(concept_df['vocabulary_id'].unique().tolist())
    print(f"after check_concept_name_duplicates ={concept_df.shape}")
    if only_standard:
        concept_df = concept_df[concept_df['standard_concept'].isin(['S', 'C'])]
    print(f"Total Standard concept = {concept_df.shape}")
    if vocabulary_list:
        concept_df = concept_df[~concept_df['vocabulary_id'].isin(vocabulary_list)]
    print(concept_df['vocabulary_id'].unique().tolist())
    print(f"unique standard_concept={concept_df['standard_concept'].unique().tolist()}")    
    # concept_df = concept_df[concept_df['vocabulary_id'].isin(vocabulary_list)]
    concept_df = concept_df[concept_df['invalid_reason'].isnull() | (concept_df['invalid_reason'] == '')]
    print(f"Valid_Concepts ={concept_df.shape}")
    print(concept_df['vocabulary_id'].unique().tolist())
    concept_df = concept_df.drop(['valid_start_date','valid_end_date','invalid_reason'],axis=1)
    print(f"Removed un_ncesssary columns ={concept_df.shape}")
    print(f"after remove some domains:{concept_df.shape}")
    output_file = os.path.join(output_dir, 'output/concepts_0.csv')
    concept_df.to_csv(output_file, index=False)
    print(f"Synonyms appended and saved to {output_file}")
    return concept_df

import csv
def fetch_chvsynonyms(file, concept_df, output_dir):
    # Convert unique concepts to a set for O(1) lookup time
    unique_concepts_set = set(concept_df['concept_name'].str.lower().unique())

    # Load synonyms and preprocess
    synonyms_list = []

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        print("Opening file...")
        next(reader)  # Skip the header row if there is one
        for row in tqdm(reader, desc="Loading Synonyms"):
            if len(row) >= 4:
                _term = row[1].strip().lower()
                chv_terms = row[2].strip().lower()
                standard_term = row[3].strip().lower()
                if standard_term in unique_concepts_set:
                    synonyms_list.append((standard_term, chv_terms))
                    synonyms_list.append((standard_term, _term))

    # Convert the synonyms list into a DataFrame
    synonyms_df = pd.DataFrame(synonyms_list, columns=['standard_term', 'synonym'])
    synonyms_df = synonyms_df.groupby('standard_term')['synonym'].apply(lambda x: ';;'.join(set(x))).reset_index()

    # Convert concept_df to lowercase and merge with the synonyms dataframe
    concept_df['concept_name'] = concept_df['concept_name'].str.lower()
    concept_df = concept_df.merge(synonyms_df, how='left', left_on='concept_name', right_on='standard_term')

    # Fill NaN in the synonym column and combine it with any existing synonyms
    concept_df['synonym'] = concept_df['synonym'].fillna('')
    if 'concept_synonym_name' in concept_df.columns:
        concept_df['concept_synonym_name'] = concept_df.apply(
            lambda row: row['concept_synonym_name'].strip(';;') + ';;' + row['synonym'].strip(';;')
            if row['concept_synonym_name'] and row['synonym'] 
            else (row['concept_synonym_name'].strip(';;') if row['concept_synonym_name'] else row['synonym'].strip(';;')), 
            axis=1
        )
    else:
        concept_df['concept_synonym_name'] = concept_df['synonym']
    
    # Clean up the concept_synonym_name column by removing leading/trailing delimiters
    concept_df['concept_synonym_name'] = concept_df['concept_synonym_name'].str.strip(';;')

    # Drop the temporary synonym and standard_term columns used for merging
    concept_df = concept_df.drop(columns=['synonym', 'standard_term'])

    # Write the result to a CSV file
    output_file = os.path.join(output_dir, 'output/concepts_syn.csv')
    concept_df.to_csv(output_file, index=False)
    print(f"Synonyms appended and saved to {output_file}")

    return concept_df

def combine_synonyms(row):
    existing = row.get('concept_synonym_name_existing', '')
    new = row.get('concept_synonym_name_new', '')
    
    # Handle NaN values
    if pd.notna(existing) and pd.notna(new):
        # Combine existing and new synonyms, ensuring no duplicates
        combined = set(existing.strip(';;').split(';;')) | set(new.strip(';;').split(';;'))
        return ';;'.join(sorted(combined))
    elif pd.notna(existing):
        return existing.strip(';;')
    elif pd.notna(new):
        return new.strip(';;')
    else:
        return ''

def standardize_concept_id(concept_df, synonyms_df):
    concept_df['concept_id'] = concept_df['concept_id'].astype(str).str.strip()
    synonyms_df['concept_id'] = synonyms_df['concept_id'].astype(str).str.strip()
    return concept_df, synonyms_df

def fetch_synonyms(file, concept_df, output_dir=DATA_DIR):
    # Read the synonyms file
    synonyms = pd.read_csv(file, sep='\t', low_memory=False, dtype=str)
    
    print("Initial synonyms columns:", synonyms.columns)
    print(f"Total synonyms loaded: {len(synonyms)}")

    # Standardize concept_id formats
    synonyms = synonyms.rename(columns=lambda x: x.strip())
    concept_df, synonyms = standardize_concept_id(concept_df, synonyms)
    
    # Use parallel processing to filter English synonyms
    num_cores = mp.cpu_count()
    chunks = np.array_split(synonyms, num_cores)
    
    print(f"Processing synonyms using {num_cores} cores...")
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Concatenate the results
    synonyms = pd.concat(results, ignore_index=True)
    print("Processed synonyms count:", synonyms.shape)
    
    # Keep only English synonyms
    english_synonyms = synonyms[synonyms['is_english']]
    print("English synonyms count:", english_synonyms.shape)
    
    # Group by concept_id
    grouped_synonyms = english_synonyms.groupby('concept_id')['concept_synonym_name'].apply(lambda x: ';;'.join(x)).reset_index()
    print("Grouped synonyms count:", grouped_synonyms.shape)
    
    # Check for duplicates in grouped_synonyms
    if grouped_synonyms['concept_id'].duplicated().any():
        print("Error: Duplicate concept_id entries found in grouped_synonyms.")
    
    # Merge with concept_df and retain all concepts with explicit suffixes
    merged_count_before = len(concept_df)
    concept_df = pd.merge(concept_df, grouped_synonyms, on='concept_id', how='left', suffixes=('_existing', '_new'))
    print(f"Merged concepts with synonyms count: {concept_df.shape}")
    
    # Verify that no rows were lost
    merged_count_after = len(concept_df)
    if merged_count_before != merged_count_after:
        print(f"Warning: Row count changed after merge. Before: {merged_count_before}, After: {merged_count_after}")
    else:
        print("Row count unchanged after merge.")
    
    # Check for duplicate concept_ids post-merge
    if concept_df['concept_id'].duplicated().any():
        print("Error: Duplicate concept_id entries found after merging synonyms.")
    
    # Update the concept_synonym_name column by appending new synonyms to existing ones
    concept_df['concept_synonym_name'] = concept_df.apply(combine_synonyms, axis=1)
    concept_df['concept_synonym_name'] = concept_df['concept_synonym_name'].str.strip(';;')
    print("Final concepts with synonyms count:", concept_df.shape)
    
    # Drop the temporary synonym columns used for merging
    concept_df = concept_df.drop(columns=['concept_synonym_name_existing', 'concept_synonym_name_new'], errors='ignore')
    print("Dropped temporary synonym columns.")
    
    # Calculate synonym count percentage per vocabulary
    total_concepts = len(concept_df)
    concepts_with_synonyms = concept_df['concept_synonym_name'].apply(lambda x: bool(x)).sum()
    print(f"\nTotal concepts with English synonyms: {concepts_with_synonyms} out of {total_concepts} concepts")
    overall_percentage = (concepts_with_synonyms / total_concepts) * 100
    print(f"Overall percentage of concepts with synonyms: {overall_percentage:.2f}%")
    
    # Calculate percentage for each vocabulary
    vocab_stats = concept_df.groupby('vocabulary_id').agg({
        'concept_id': 'count',
        'concept_synonym_name': lambda x: x.apply(lambda y: bool(y)).sum()
    }).reset_index()
    
    vocab_stats['percentage'] = (vocab_stats['concept_synonym_name'] / vocab_stats['concept_id']) * 100
    vocab_stats = vocab_stats.sort_values('percentage', ascending=False)
    
    print("\nPercentage of concepts with synonyms per vocabulary:")
    for _, row in vocab_stats.iterrows():
        print(f"{row['vocabulary_id']}: {row['percentage']:.2f}% ({row['concept_synonym_name']} out of {row['concept_id']} concepts)")
    
    # Save the final DataFrame
    output_csv_path = os.path.join(output_dir, 'output/concepts.csv')
    concept_df.to_csv(output_csv_path, index=False)
    print(f"Saving concepts with synonyms to {output_csv_path}")
    
    return concept_df



def update_non_standard_mapping(relationship_file,concept_df_file):
    
    concept_df = pd.read_csv(concept_df_file, sep='\t', low_memory=False, dtype=str)
    print(concept_df.columns)
    relationship_df = pd.read_csv(relationship_file, sep='\t', low_memory=False, dtype=str)
    concept_df['concept_synonym_name'] = np.nan
    # Filter for "Maps to" relationships
    maps_to_relationships = relationship_df[relationship_df['relationship_id'] == 'Maps to']
    
    # Filter non-standard ('C') and standard ('S') concepts
    non_standard_concepts = concept_df[~concept_df['standard_concept'].isin(['S','C'])]
    standard_concepts = concept_df[concept_df['standard_concept'].isin(['S','C'])]
    
    # Filter 'Maps To' relationships where non-standard maps to standard
    maps_to_filtered = maps_to_relationships[
        (maps_to_relationships['concept_id_1'].isin(non_standard_concepts['concept_id'])) &
        (maps_to_relationships['concept_id_2'].isin(standard_concepts['concept_id']))
    ]
    
    # Create a map of standard concept names to their non-standard synonyms
    id_to_name_map = pd.Series(concept_df['concept_name'].values, index=concept_df['concept_id']).to_dict()
    
    # Prepare list to hold new synonyms
    new_synonyms = []
    
    # Loop through filtered 'Maps To' relationships
    for _, row in maps_to_filtered.iterrows():
        non_standard_name = id_to_name_map.get(row['concept_id_1'])
        standard_name = id_to_name_map.get(row['concept_id_2'])
        
        # Add non-standard name as a synonym of the standard concept
        new_synonyms.append((standard_name, non_standard_name))
    
    # Convert new_synonyms list to a DataFrame
    synonyms_df = pd.DataFrame(new_synonyms, columns=['standard_term', 'synonym'])
    
    # Group synonyms by standard_term
    synonyms_df = synonyms_df.groupby('standard_term')['synonym'].apply(lambda x: ';;'.join(set(x))).reset_index()
    
    # Merge new synonyms into concept_df
    concept_df['concept_name'] = concept_df['concept_name'].str.lower()  # Ensure lowercase matching
    concept_df = concept_df.merge(synonyms_df, how='left', left_on='concept_name', right_on='standard_term')
    
    # Fill NaN in the synonym column and combine it with any existing synonyms
    concept_df['synonym'] = concept_df['synonym'].fillna('')
    if 'concept_synonym_name' in concept_df.columns:
        concept_df['concept_synonym_name'] = concept_df.apply(
            lambda row: str(row['concept_synonym_name']).strip(';;') + ';;' + str(row['synonym']).strip(';;')
            if row['concept_synonym_name'] and row['synonym'] 
            else (str(row['concept_synonym_name']).strip(';;') if row['concept_synonym_name'] else str(row['synonym']).strip(';;')), 
            axis=1
        )
    else:
        concept_df['concept_synonym_name'] = concept_df['synonym'].str.strip(';;')

# Remove any unwanted empty ';;' created by improper concatenation

    # Clean up the concept_synonym_name column by removing leading/trailing delimiters
    concept_df['concept_synonym_name'] = concept_df['concept_synonym_name'].replace(';;;;', ';;')
    # Drop temporary columns used for merging
    concept_df = concept_df.drop(columns=['synonym', 'standard_term'])
    
    # Save the updated concept dataframe
    # output_file = os.path.join(data_dir, 'output/concepts_syn.csv')
    # concept_df.to_csv(output_file, sep='\t', index=False)   
    # print(f"Synonyms appended and saved to {output_file}")
    
    return concept_df

def update_relationships(file,concept_df,relationship_names, data_dir=DATA_DIR):
    relationship_df = pd.read_csv(file,sep='\t',low_memory=False, dtype=str)
    print(f"Rel={relationship_df.shape}")
    concept_ids = concept_df['concept_id'].tolist()
    relationship_df = relationship_df[
    (relationship_df['concept_id_1'].isin(concept_ids)) & 
    (relationship_df['concept_id_2'].isin(concept_ids))
]
    print(f"Rel={relationship_df.shape}")
    print(f"unique relationship_id={relationship_df['relationship_id'].unique().tolist()}")
    relationship_counts = relationship_df.groupby('relationship_id').size().reset_index(name='counts')
# Log the output of the most common relationships
    print("Relationship counts:\n{}".format(relationship_counts.sort_values(by='counts', ascending=False).head()))

    id_to_name_map = pd.Series(concept_df['concept_name'].values, index=concept_df['concept_id']).to_dict()
    relationship_df['concept_name_1'] = relationship_df['concept_id_1'].map(id_to_name_map)
    relationship_df['concept_name_2'] = relationship_df['concept_id_2'].map(id_to_name_map)

    # Update the relationship mapping to use relationship IDs from a file
    relationship_name_map = pd.Series(relationship_names['relationship_name'].values, index=relationship_names['relationship_id']).to_dict()
    relationship_df['relationship_name'] = relationship_df['relationship_id'].map(relationship_name_map)
    relationship_df.to_csv(f"{os.path.join(data_dir, 'output/relationships.csv')}", index=False)
    
from collections import defaultdict
def preprocess_data():
    df = pd.read_csv('data/CONCEPT_ANCESTOR.csv', sep='\t')
    hierarchy = defaultdict(lambda: {'descendants': defaultdict(int), 'ancestors': defaultdict(int)})
    for ancestor, dependent, level in zip(df['ancestor_name'], df['descendant_name'], df['max_levels_of_separation']):
        hierarchy[ancestor]['descendants'][dependent] = level
        hierarchy[dependent]['ancestors'][ancestor] = level
    import pickle

    # Assuming 'hierarchy_mapping' is your dictionary from the preprocessing step
    file_name = 'data/hierarchy_mapping.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(hierarchy, file)
    print("Hierarchy mapping saved to", file_name)
    return hierarchy 
import os
import argparse
def main_process(data_directory='/workspace/mapping_tool/data/omop/ALL_VOCAB_29_07_2024', output_dir=DATA_DIR):
    print(f"Processing data in {data_directory}")
    concepts_df = fetch_concept(data_directory, sep='\t', output_dir = output_dir)
    concepts_df = fetch_synonyms(os.path.join(data_directory, 'CONCEPT_SYNONYM.csv'),concepts_df, output_dir=output_dir)
    relationship_names = pd.read_csv(os.path.join(data_directory, 'RELATIONSHIP.csv'), sep='\t', low_memory=False, dtype=str)
    print(f"unique relationships = {relationship_names['relationship_id'].unique().tolist()}")
    update_relationships(os.path.join(data_directory, 'CONCEPT_RELATIONSHIP.csv'),
                            concepts_df,relationship_names, data_dir=output_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process vocabulary data.")
    parser.add_argument("--data_dir", type=str, default="/workspace/mapping_tool/data/omo/ALL_VOCAB_29_07_2024",
                        help="Directory containing the vocabulary data files")
    parser.add_argument("--output_dir", type=str, default=DATA_DIR,
                        help="Directory containing the vocabulary data files")
    args = parser.parse_args()
    main_process(data_directory=args.data_dir, output_dir=args.output_dir)
