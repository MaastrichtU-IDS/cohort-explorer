

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, zscore
import warnings
import decentriq_util
import re
from datetime import datetime
import json
warnings.filterwarnings('ignore')



# Load the dataset with corrected missing values replaced with NA from previous step
# data_correct_missing = pd.read_csv("/input/C3_map_missing_do_not_run/data_correct.csv", low_memory=False)

#Load the JSON files from C2
categorical_vars = pd.read_json("/input/c2_save_to_json/categorical_variables.json")
print("the categorical data: ", categorical_vars.keys())
numerical_vars = pd.read_json("/input/c2_save_to_json/numerical_variables.json")
#print("the numerical data: ", numerical_vars)
data = decentriq_util.read_tabular_data("/input/GISSI-HF")
#print(data)

#variables that should be graphed
#vars_to_graph = ['age', 'weight', 'cough1', 'angina1', 'hscrp_v6']
vars_to_graph = ['age', 'ALCOOL', 'ALLOPURI', 'ALT', 'ALTACE', 'ALTANO', 'COLETOT', 'CREATIN', 'DALTACE', 'DATAECG', 'DATALAB']
vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB']
vars_to_graph = [x.lower() for x in vars_to_graph]
#vars_to_graph = list(categorical_vars.columns) + list(numerical_vars.columns)
#vars_to_graph = [x.strip().lower() for x in vars_to_graph]


def variable_eda(df, categorical_vars, numerical_vars):
    vars_stats = {}
    df.columns = df.columns.str.lower().str.strip()
    for column in df.columns.tolist():
        # Continuous variables
        if column in list(numerical_vars.keys()):

            #if not pd.api.types.is_numeric_dtype(df[column]):
                # Skip if the column is not numeric
                # print("Column ", column, " skipped because non-numeric")
                #continue

            # Descriptive Stats
            stats = df[column].describe()
            mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            total_missing = df[column].isnull().sum()
            missing_percent = total_missing / len(df) * 100

            # Check for numeric values before computing skewness and kurtosis
            if len(df[column].dropna()) > 0:
                #print("COLUMN: ", column, "type: ", df[column].dtype, "values: ", df[column])
                #df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column] = df[column].astype(float)
                skewness = skew(df[column].dropna(), bias=False)
                #skewness = df[column].skew()
                kurt = kurtosis(df[column].dropna(), bias=False)
            else:
                skewness = np.nan
                kurt = np.nan

            # Normality Test
            if len(df[column].dropna()) > 3:  # Shapiro requires at least 3 values
                w_test_stat, p = shapiro(df[column].dropna())
                normality = "Normal" if p > 0.05 else "Non-Normal"
                p_value_str = f"{p:.4f}"
            else:
                normality = "Insufficient Data"
                p_value_str = "N/A"

            # Outlier Detection (IQR Method)
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].count()

            # Z-Scores for Outliers
            z_scores = zscore(df[column].dropna()) if len(df[column].dropna()) > 0 else np.array([])
            z_outliers = (np.abs(z_scores) > 3).sum() if z_scores.size > 0 else 0

            # Range Calculation
            range_value = stats['max'] - stats['min']

            # Stats Text
            stats_text = (
                f"Column: {column}",
                f"Label: {numerical_vars[column]['var_label']}",
                f"Type: Numeric (encoded as {df[column].dtype})",
                f"Number of non-null observations: {int(stats['count'])}",
                f"Number of Unique Values/Categories: {df[column].nunique()}",
                f"Mean:                {stats['mean']:.2f}",
                f"Median:              {stats['50%']:.2f}",
                f"Mode:                {mode_value:.2f}",
                f"Std Dev:             {stats['std']:.2f}",
                f"Variance:            {stats['std']**2:.2f}",
                f"Range:               {range_value:.2f}", 
                f"Q1:                  {Q1:.2f}",
                f"Q3:                  {Q3:.2f}",
                f"IQR:                 {IQR:.2f}",
                f"Missing:             {total_missing} ({missing_percent:.2f}%)",
                f"Outliers (IQR):      {outliers} ({(outliers / len(df) * 100):.2f}%)",
                f"Outliers (Z):        {z_outliers}",
                f"Skewness:            {skewness:.2f}",
                f"Kurtosis:            {kurt:.2f}",
                f"W_Test:              {w_test_stat:.2f}",
                f"Normality Test: p-value={p_value_str} => {normality}"
            )

            if column in vars_to_graph:
                create_save_graph(df, column, stats_text, 'numerical')


        # Categorical variables
        elif column in categorical_vars.keys() and categorical_vars[column]['var_type'] != 'datetime':
            stats = df[column].describe()
            value_counts = df[column].value_counts(dropna=False)
            total = len(df)

            # Get the categories mapping and normalize keys
            categories_mapping = categorical_vars[column].get("categories", [])
            categories_mapping = {str(k): v for (k, v) in categories_mapping.items()}

            if value_counts.empty:
                stats_text = (
                    f"Column: {column}",
                    f"Label: {categorical_vars[column]['var_label']}",
                    f"Type: Categorical (encoded as {df[column].dtype})",
                    f"Number of Unique Categories: 0",
                    f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)"
                )
            else:
                # Chi-square test
                expected = total / len(value_counts)
                chi_square_stat = ((value_counts - expected) ** 2 / expected).sum()

                # Class balance with corrected mapping
                class_balance_text = "\n" + "\n     ".join([
                    f"{(key, categories_mapping.get(str(key), 'Unknown'))} -> {round(count / total * 100)}%"
                    for key, count in value_counts.items()
                ])

                stats_text = (
                    f"Column: {column}",
                    f"Label: {categorical_vars[column]['var_label']}",
                    f"Type: Categorical (encoded as {df[column].dtype})",
                    f"Number of unique values/categories: {len(value_counts)}",
                    f"Most frequent category: {categories_mapping.get(str(value_counts.idxmax()).split('.')[0], 'Unknown')} ",
                    f"Number of non-null observations: {df[column].count()}",
                    f"Missing: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)",
                    f"Class balance: {class_balance_text}",
                    f"Chi-Square Test Statistic: {chi_square_stat:.2f}"
                )

            if column in vars_to_graph:
                create_save_graph(df, column, stats_text, 'categorical', category_mapping = categories_mapping)
        elif column in categorical_vars.keys() and categorical_vars[column]['var_type'] == 'datetime':
            stats = pd.to_datetime(df[column], format='%Y-%m-%d').describe()
            value_counts = df[column].value_counts(dropna=False)
            total = len(df)
            stats_text = [
                    f"Column: {column}",
                    f"Label: {categorical_vars[column]['var_label']}",
                    f"Type: Date (encoded as {df[column].dtype})",
                    f"Number of unique values: {len(value_counts)}",
                    f"Most frequent value: {categories_mapping.get(str(value_counts.idxmax()).split('.')[0], 'Unknown')}",
                    f"Number of non-null observations: {df[column].count()}",
                    f"Missing: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)"
            ]
            stats_text.extend([f"{k.capitalize()}: {v}" for k,v in stats.items()])
            if column in vars_to_graph:
                create_save_graph(df, column, stats_text, 'datetime')

        stats_text_dict = {i.split(":")[0].strip():i.split(":")[1].strip() for i in stats_text}
        if 'Class balance' in stats_text_dict:
            stats_text_dict['Class balance'].replace(" ->", ":")
        vars_stats[column] = stats_text_dict
    return vars_stats



def create_save_graph(df, varname, stats_text, vartype, category_mapping=None):
    if vartype == 'numerical':
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Display Summary Stats

        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor="lightgray")
        text_obj = axes[0].text(0.05, 0.95, '\n'.join(stats_text), transform=axes[0].transAxes, fontsize=11, va='top', ha='left', 
        family='monospace',  bbox=props, wrap=True, linespacing=1.5)
        if hasattr(text_obj, "_get_wrap_line_width"):
            text_obj._get_wrap_line_width = lambda: 400
        #axes[0].text(0.05, 0.9, , fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
        axes[0].axis("off")

        # Right: Plot histogram
        sns.histplot(df[varname].dropna(), kde=True, ax=axes[1])
        axes[0].set_title(f"Statistics Summary for {varname}.upper()", fontsize=12)
        axes[1].tick_params(axis='x')

        # Save the figure for the current feature
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")
        print(f"figure for {varname} saved!! ")
        plt.close()
    elif vartype == 'datetime':
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor="lightgray")
        text_obj = axes[0].text(0.05, 0.95, '\n'.join(stats_text), transform=axes[0].transAxes, fontsize=11, va='top', ha='left', 
        family='monospace',  bbox=props, wrap=True, linespacing=1.5)
        if hasattr(text_obj, "_get_wrap_line_width"):
            text_obj._get_wrap_line_width = lambda: 400
        #axes[0].text(0.05, 0.9, , fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
        axes[0].axis("off")
        axes[0].set_title(f"Statistics Summary for {varname}.upper()", fontsize=12)
        date_vals =  pd.to_datetime(df[varname].dropna(), format='%Y-%m-%d')
    
        date_nums = mdates.date2num(date_vals)
    
        min_date = date_vals.min()
        max_date = date_vals.max()
        date_range = max_date - min_date
    
        # bin frequency based on date range
        if date_range.days > 365:
            bin_freq = 'M'  # Monthly bins for ranges > 1 year
        elif date_range.days > 90:
            bin_freq = 'W'  # Weekly bins for ranges > 3 months
        else:
            bin_freq = 'D'  # Daily bins for shorter ranges
        
        bins = mdates.date2num(pd.date_range(min_date, max_date, freq=bin_freq))
        
        axes[1].hist(date_nums, bins=bins, alpha=0.7)
        axes[1].set_title(f"Distribution of {varname}.upper()", fontsize=12)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
        if date_range.days > 365*2:
            axes[1].xaxis.set_major_locator(mdates.YearLocator())
        elif date_range.days > 180:
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Quarterly
        else:
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())  # Monthly
            
        axes[1].tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")
        print(f"figure for {varname} saved!! ")


    elif vartype == 'categorical':

        value_counts = df[varname].value_counts(dropna=False)
        total = len(df)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Summary stats text
        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor="lightgray")
        text_obj = axes[0].text(0.05, 0.95, '\n'.join(stats_text), transform=axes[0].transAxes, fontsize=11, va='top', ha='left', 
        family='monospace', bbox=props, wrap=True, linespacing=1.5)
        if hasattr(text_obj, "_get_wrap_line_width"):
            text_obj._get_wrap_line_width = lambda: 400
        #axes[0].text(0.1, 0.5, stats_text, fontsize=10, va='center', ha='left', family='monospace', wrap=True)
        
        axes[0].axis("off")

        # Bar chart
        if not value_counts.empty:
            colors = sns.color_palette("husl", len(value_counts))
            ax = value_counts.plot(kind='bar', color=colors, edgecolor='black', ax=axes[1])
            axes[0].set_title(f"Statistics Summary for {varname}.upper()", fontsize=12)
            ax.set_xlabel("Categories")
            ax.set_ylabel("Count")

            # Add labels to the bars
            for idx, value in enumerate(value_counts):
                percentage = (value / total) * 100
                ax.text(idx, value + total * 0.02, f"{value}({percentage:.1f}%)",
                        ha='center', fontsize=10)

            # Adjust x-axis labels to be horizontal
            xticks = []
            for v in value_counts.index.astype(str):
                if v in category_mapping:
                    xticks.append(category_mapping[v])
                else:
                    xticks.append(v)
            ax.set_xticklabels(xticks, rotation=90, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")
        print(f"figure for {varname} saved!! ")
        plt.close()


def integrate_eda_with_metadata(vars_stats):
    meta_data = decentriq_util.read_tabular_data("/input/GISSI-HF-metadata")
    metadata_vars = [x.lower().strip() for x in meta_data['VARIABLE NAME'].values]
    #print("vars from var_stats:", vars_stats.keys())
    #print("vars in metadata: ", metadata_vars)
    #print(" vars in common: ", [x for x in metadata_vars if x in vars_stats.keys()])
    #print(" vars no stats: ", [x for x in metadata_vars if x not in vars_stats.keys()])
    additional_cols = set()
    for s in vars_stats.values():
        additional_cols.update(s.keys())
    for c in additional_cols:
        cvals = []
        for vname in metadata_vars:
            if not vname in vars_stats or not c in vars_stats[vname]:
                cvals.append(None)
            else:
                cvals.append(vars_stats[vname][c])
        meta_data[c] =cvals
    meta_data.to_csv("/output/meta_data_enriched.csv")
    return meta_data


def dataframe_to_json_dicts(df):
    json_dicts = {}
    for _, row in df.iterrows():
        variable_name = row['VARIABLE NAME']
        var_dict = {}
        for col in df.columns:
            if col not in ['VARIABLE NAME', 'Column'] and pd.notna(row[col]) and row[col] != "":
                var_dict[col] = row[col]
        json_dicts[variable_name] = var_dict
    with open("/output/eda_output_GISSI-HF.json", 'w', encoding='utf-8') as f:
        json.dump(json_dicts, f, indent=4)


vars_to_stats = variable_eda(data, categorical_vars, numerical_vars)
meta_data_enriched = integrate_eda_with_metadata(vars_to_stats)
json_dicts = dataframe_to_json_dicts(meta_data_enriched)
