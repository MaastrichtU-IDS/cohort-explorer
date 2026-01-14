#!/usr/bin/env python3
"""Check for duplicate cohort entries in metadata file"""

import pandas as pd
import sys

try:
    filepath = "/data/cohorts_metadata/iCARE4CVD_Cohorts.xlsx"
    df = pd.read_excel(filepath, sheet_name="Descriptions")
    df.columns = df.columns.str.lower()
    
    print(f"Total rows in metadata file: {len(df)}")
    print(f"Unique cohorts: {df['study name'].nunique()}\n")
    
    # Find duplicates
    study_counts = df['study name'].value_counts()
    duplicates = study_counts[study_counts > 1]
    
    if len(duplicates) > 0:
        print("⚠️  COHORTS WITH MULTIPLE ROWS:")
        print("="*50)
        for study_name, count in duplicates.items():
            print(f"\n{study_name}: {count} rows")
            
            # Show what's different between rows
            study_rows = df[df['study name'] == study_name]
            print(f"  Row indices: {study_rows.index.tolist()}")
            
            # Check which columns differ
            differing_cols = []
            for col in study_rows.columns:
                if study_rows[col].nunique() > 1:
                    differing_cols.append(col)
            
            if differing_cols:
                print(f"  Columns with different values: {', '.join(differing_cols[:10])}")
            else:
                print("  All rows are identical (complete duplicates)")
    else:
        print("✅ No duplicate cohorts found - each study has exactly 1 row")
        
except FileNotFoundError:
    print(f"❌ File not found: {filepath}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
