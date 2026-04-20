import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .query_builder import SPARQLQueryBuilder
from .utils import execute_query
from .data_model import VariableProfileRow


class VariableProfile:
    @staticmethod
    def _fetch_chunk(chunk: list, study_name: str, graph_repo: str) -> list:
        results = []
        try:
            values_str = " ".join(f'"{v}"' for v in chunk)
            query = SPARQLQueryBuilder.build_statistic_query(study_name, values_str, graph_repo)
            response = execute_query(query)
            bindings = response.get("results", {}).get("bindings", [])
            for res in bindings:
                identifier = res['identifier']['value']
                def val(k): return res[k]['value'] if k in res and res[k]['value'].strip() else None
                row = VariableProfileRow(
                    identifier=identifier,
                    stat_label=val('stat_label'),
                    unit_label=val('unit_label'),
                    data_type=val('data_type_val'),
                    categories_labels=val('all_cat_labels'),
                    categories_omop_ids=val('cat_omop_ids'),
                    original_categories=val('all_original_cat_values'),
                    composite_code_labels=val('code_label'),
                    composite_code_values=val('code_value'),
                    composite_code_omop_ids=val('omop_ids'),
                    min_val=val('min_val'),
                    max_val=val('max_val'),
                )
                results.append(row.model_dump())
        except Exception as e:
            print(f"Error chunk fetch: {e}")
        return results

    # =================================================================
    # New: fetch-only — O(n) per study, called once upfront
    # =================================================================

    @classmethod
    def fetch_profiles(cls, var_names: list, study: str, graph_repo: str) -> pd.DataFrame:
        """Fetch variable profiles from the KG. Returns raw DataFrame keyed by 'identifier'.
        
        O(n) per study. Called once per study before matching.
        Derived variables are excluded (they set their own profile).
        """
        vars_ = [v for v in var_names if not v.startswith('derived_')]
        if not vars_:
            return pd.DataFrame()
        data = []
        chunks = [vars_[i:i + 50] for i in range(0, len(vars_), 50)]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(cls._fetch_chunk, c, study, graph_repo): c for c in chunks}
            for f in as_completed(futures):
                data.extend(f.result())
        return pd.DataFrame(data)

    # =================================================================
    # Legacy: combined fetch + merge (backward compatibility)
    # =================================================================

    @classmethod
    def _merge_side(cls, df: pd.DataFrame, profile_df: pd.DataFrame,
                    side: str, is_derived_mask: pd.Series) -> pd.DataFrame:
        if profile_df.empty:
            return df
        col_map = VariableProfileRow.column_map(side)
        profile_df = profile_df.rename(columns=col_map)
        new_cols = [c for c in profile_df.columns if c != side]
        df_normal = df[~is_derived_mask].drop(
            columns=[c for c in new_cols if c in df.columns], errors='ignore'
        )
        df_derived = df[is_derived_mask].copy()
        df_normal = df_normal.merge(profile_df, on=side, how="left")
        for col in new_cols:
            if col not in df_derived.columns:
                df_derived[col] = None
        return pd.concat([df_normal, df_derived], ignore_index=True)

    @classmethod
    def attach_attributes(cls, df: pd.DataFrame, src_study: str, tgt_study: str, graph_repo: str) -> pd.DataFrame:
        """Legacy: fetch + merge in one call."""
        src_is_derived = df["source"].str.contains("derived", case=False, na=False)
        tgt_is_derived = df["target"].str.contains("derived", case=False, na=False)
        src_vars = df.loc[~src_is_derived, "source"].dropna().unique().tolist()
        tgt_vars = df.loc[~tgt_is_derived, "target"].dropna().unique().tolist()
        src_df = cls.fetch_profiles(src_vars, src_study, graph_repo)
        tgt_df = cls.fetch_profiles(tgt_vars, tgt_study, graph_repo)
        df = cls._merge_side(df, src_df, "source", src_is_derived)
        tgt_is_derived = df["target"].str.contains("derived", case=False, na=False)
        df = cls._merge_side(df, tgt_df, "target", tgt_is_derived)
        return df
