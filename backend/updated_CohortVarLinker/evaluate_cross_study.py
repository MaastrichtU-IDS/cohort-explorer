
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score
)
import os
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler('eval.log'))
logger.info("Starting evaluation...")

# Set style for all plots
PALETTE = {
    # Primary comparison pair (ground truth vs predicted)
    'primary':   '#2066A8',   # dark blue   (GeoDataViz sequential)
    'secondary': '#D4764E',   # muted terracotta / orange
    # Metric triad (accuracy / F1-weighted / F1-macro)
    'metric_1':  '#298C8C',   # teal        (image 2, colorblind-safe)
    'metric_2':  '#3594CC',   # medium blue (image 2 sequential)
    'metric_3':  '#EA801C',   # amber       (image 2 divergent pair)
    # Per-class metrics (precision / recall / F1)
    'precision': '#EA801C',   # amber
    'recall':    '#2066A8',   # dark blue
    'f1':        '#AF58BA',   # muted purple (GeoDataViz qualitative, CB-safe)
    # Supplementary / accent
    'accent':    '#AF58BA',   # muted purple
    'neutral':   '#888888',   # mid-gray
    'alert':     '#E8601C',   # deep orange  (avoids pure red)
    'good':      '#298C8C',   # teal         (avoids pure green)
    # Heatmap colormaps — colorblind-safe (Crameri 2024)
    'cmap_seq':  'YlGnBu',   # sequential   (blue-dominant, CB-safe)
    'cmap_div':  'PuOr',     # diverging    (purple-orange, CB-safe; avoids RdYlGn)
}

# Mode palette for consistent color coding across plots
MODE_COLORS = {
    'OO':  '#298C8C',   # teal
    'NE':  '#3594CC',   # blue
    'OEH': '#EA801C',   # amber
    'OEC': '#AF58BA',   # purple
    'OED': '#D4764E',   # terracotta
}

def _mode_color(mode: str) -> str:
    """Return a consistent color for a given mode, with fallback."""
    return MODE_COLORS.get(mode, PALETTE['neutral'])

# Set style for all plots
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.linewidth':    0.8,
    'axes.edgecolor':    '#333333',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})
sns.set_palette([PALETTE['primary'], PALETTE['secondary'], PALETTE['metric_1'],
                 PALETTE['accent'], PALETTE['metric_3']])



# def load_ground_truth(file_path: str, source_study: str, target_study: str) -> pd.DataFrame:
#     """Reads the ground truth CSV file and returns a DataFrame."""
#     t_df = pd.read_csv(file_path)
#     ground_truth = []
#     for id, row in t_df.iterrows():
#         if row['source_study'].strip().lower() == source_study and row['target_study'].strip().lower() == target_study:
#             source_var = row['source_var_name'].strip().lower()
#             target_var = row['target_var_name'].strip().lower()
#             level = row['harmonization level'].strip().lower()
#             ground_truth.append({
#                 'src_var': source_var,
#                 'tgt_var': target_var,
#                 'correct class': level
#             })
#     return pd.DataFrame(ground_truth)
def load_ground_truth(file_path: str, source_study: str, target_study: str) -> pd.DataFrame:
    """Reads the ground truth CSV file and returns a DataFrame."""
    for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
        try:
            t_df = pd.read_csv(file_path, encoding=enc)
            logger.info(t_df['harmonization level'].value_counts())
            logger.info(f"\nTotal pairs: {len(t_df)}")
            logger.info(f"\nStudy pairs:")
            logger.info(t_df.groupby(['source_study','target_study'])['harmonization level'].value_counts())

            break
        except UnicodeDecodeError:
            continue
    
    ground_truth = []
    for id, row in t_df.iterrows():
        if row['source_study'].strip().lower() == source_study and row['target_study'].strip().lower() == target_study:
            source_var = row['source_var_name'].strip().lower()
            target_var = row['target_var_name'].strip().lower()
            level = row['harmonization level'].strip().lower()
            ground_truth.append({
                'src_var': source_var,
                'tgt_var': target_var,
                'correct class': level
            })
    return pd.DataFrame(ground_truth)

def load_predictions(file_path: str) -> pd.DataFrame:
    """Reads the predictions CSV file and returns a DataFrame."""
    p_df = pd.read_csv(file_path)
    predictions = []
    for id, row in p_df.iterrows():
        source_var = row['source'].strip().lower()
        target_var = row['target'].strip().lower()
        score = row['harmonization_status'].strip().lower()
        predictions.append({
            'src_var': source_var,
            'tgt_var': target_var,
            'predicted class': score
        })
    return pd.DataFrame(predictions)


def analyze_class_distribution(ground_truth: pd.DataFrame, predictions: pd.DataFrame, 
                                study_pair: str = "") -> dict:
    """
    Analyze and compare class distributions between ground truth and predictions.
    Returns distribution stats and flags imbalance issues.
    """
    merged_df = pd.merge(ground_truth, predictions, on=['src_var', 'tgt_var'], 
                         how='left', suffixes=('_gt', '_pred'))
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')
    
    gt_dist = ground_truth['correct class'].value_counts()
    pred_dist = merged_df['predicted class'].value_counts()
    all_classes = sorted(set(gt_dist.index) | set(pred_dist.index))
    
    comparison = pd.DataFrame({
        'class': all_classes,
        'gt_count': [gt_dist.get(c, 0) for c in all_classes],
        'pred_count': [pred_dist.get(c, 0) for c in all_classes],
    })
    comparison['gt_pct'] = (comparison['gt_count'] / comparison['gt_count'].sum() * 100).round(1)
    comparison['pred_pct'] = (comparison['pred_count'] / comparison['pred_count'].sum() * 100).round(1)
    comparison['diff_pct'] = (comparison['pred_pct'] - comparison['gt_pct']).round(1)
    
    total_samples = len(ground_truth)
    min_class_count = gt_dist.min() if len(gt_dist) > 0 else 0
    max_class_count = gt_dist.max() if len(gt_dist) > 0 else 0
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
    
    warnings_list = []
    for cls in all_classes:
        count = gt_dist.get(cls, 0)
        if count < 5:
            warnings_list.append(f"⚠️  Class '{cls}' has only {count} samples - metrics unreliable")
    if imbalance_ratio > 5:
        warnings_list.append(f"⚠️  High class imbalance (ratio {imbalance_ratio:.1f}:1) - consider macro-averaged metrics")
    
    return {
        'comparison': comparison,
        'gt_distribution': gt_dist,
        'pred_distribution': pred_dist,
        'imbalance_ratio': imbalance_ratio,
        'total_samples': total_samples,
        'warnings': warnings_list,
        'study_pair': study_pair
    }


def compute_comprehensive_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Compute comprehensive metrics including per-class and aggregate measures."""
    merged_df = pd.merge(ground_truth, predictions, on=['src_var', 'tgt_var'], 
                         how='left', suffixes=('_gt', '_pred'))
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')
    
    y_true = merged_df['correct class']
    y_pred = merged_df['predicted class']
    labels = sorted(set(y_true) | set(y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class_metrics = pd.DataFrame({
        'class': labels,
        'precision': precision.round(3),
        'recall': recall.round(3),
        'f1_score': f1.round(3),
        'support': support
    })
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'labels': labels,
        'y_true': y_true,
        'y_pred': y_pred,
        'classification_report': report
    }


def evaluate_predictions(ground_truth: pd.DataFrame, predictions: pd.DataFrame, 
                         data_dir: str, predict_studies_names: str) -> dict:
    """Evaluates the predictions against the ground truth and returns metrics."""
    merged_df = pd.merge(ground_truth, predictions, on=['src_var', 'tgt_var'], 
                         how='left', suffixes=('_gt', '_pred'))
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')
    
    total = len(merged_df)
    correct = (merged_df['correct class'] == merged_df['predicted class']).sum()
    accuracy = correct / total if total > 0 else 0.0
    
    all_incorrect = merged_df[merged_df['correct class'] != merged_df['predicted class']]
    logger.info("Incorrect Predictions:")
    logger.info(all_incorrect)
    all_incorrect.to_csv(f"{data_dir}/incorrect_predictions_{predict_studies_names}.csv", index=False)
    
    gt_pairs = set(zip(ground_truth['src_var'], ground_truth['tgt_var']))
    pred_pairs = set(zip(predictions['src_var'], predictions['tgt_var']))
    
    not_in_gt = pred_pairs - gt_pairs
    not_in_pred = gt_pairs - pred_pairs
    
    not_in_gt_df = predictions[
        predictions.apply(lambda r: (r['src_var'], r['tgt_var']) in not_in_gt, axis=1)
    ].copy()
    
    not_in_pred_df = ground_truth[
        ground_truth.apply(lambda r: (r['src_var'], r['tgt_var']) in not_in_pred, axis=1)
    ].copy()
    
    not_in_gt_df.to_csv(f"{data_dir}/predicted_not_in_ground_truth_{predict_studies_names}.csv", index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Predicted pairs NOT in ground truth: {len(not_in_gt_df)}")
    logger.info(f"Saved to: {data_dir}/predicted_not_in_ground_truth_{predict_studies_names}.csv")
    logger.info(f"{'='*60}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Ground truth pairs NOT in predictions: {len(not_in_pred_df)}")
    logger.info(f"{'='*60}")
    logger.info(not_in_pred_df.to_string(index=False))
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'incorrect_predictions': all_incorrect,
        'not_in_ground_truth': not_in_gt_df,
        'not_in_predictions': not_in_pred_df,
    }


def plot_class_distribution_comparison(dist_results: dict, output_path: str = None):
    """Create side-by-side bar chart comparing GT vs Predicted class distributions."""
    comparison = dist_results['comparison']
    study_pair = dist_results.get('study_pair', '')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(comparison))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, comparison['gt_count'], width, label='Ground Truth', color=PALETTE['primary'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, comparison['pred_count'], width, label='Predicted', color=PALETTE['secondary'], alpha=0.8)
    ax1.set_xlabel('Harmonization Level', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Class Distribution: Count\n{study_pair}', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison['class'], rotation=45, ha='right')
    ax1.legend()
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, comparison['gt_pct'], width, label='Ground Truth %', color=PALETTE['primary'], alpha=0.8)
    bars4 = ax2.bar(x + width/2, comparison['pred_pct'], width, label='Predicted %', color=PALETTE['secondary'], alpha=0.8)
    ax2.set_xlabel('Harmonization Level', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title(f'Class Distribution: Percentage\n{study_pair}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison['class'], rotation=45, ha='right')
    ax2.legend()
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to: {output_path}")
    plt.close()
    return fig


def plot_confusion_matrix(metrics: dict, study_pair: str = "", output_path: str = None):
    """Plot confusion matrix heatmap."""
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap=PALETTE['cmap_seq'], xticklabels=labels, 
                yticklabels=labels, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Ground Truth', fontsize=11)
    ax1.set_title(f'Confusion Matrix (Counts)\n{study_pair}', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=PALETTE['cmap_div'], xticklabels=labels, 
                yticklabels=labels, ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Ground Truth', fontsize=11)
    ax2.set_title(f'Confusion Matrix (Normalized = Recall)\n{study_pair}', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to: {output_path}")
    plt.close()
    return fig


def plot_per_class_metrics(metrics: dict, study_pair: str = "", output_path: str = None):
    """Plot per-class precision, recall, F1 as grouped bar chart."""
    pcm = metrics['per_class_metrics']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pcm))
    width = 0.25
    
    ax.bar(x - width, pcm['precision'], width, label='Precision', color=PALETTE['precision'], alpha=0.8)
    ax.bar(x, pcm['recall'], width, label='Recall', color=PALETTE['recall'], alpha=0.8)
    ax.bar(x + width, pcm['f1_score'], width, label='F1 Score', color=PALETTE['f1'], alpha=0.8)
    
    ax.set_xlabel('Harmonization Level', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Per-Class Metrics\n{study_pair}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pcm['class'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    for i, (cls, support) in enumerate(zip(pcm['class'], pcm['support'])):
        ax.annotate(f'n={support}', xy=(i, -0.08), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=9, color=PALETTE['neutral'])
    
    ax.axhline(y=0.5, color=PALETTE['neutral'], linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved per-class metrics to: {output_path}")
    plt.close()
    return fig


def plot_multi_study_comparison(all_study_metrics: list, output_path: str = None):
    """Create comparison visualization across multiple study pairs."""
    study_names = [m['study_pair'] for m in all_study_metrics]
    f1_macro = [m['metrics']['f1_macro'] for m in all_study_metrics]
    imbalance_ratios = [m['distribution']['imbalance_ratio'] for m in all_study_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    x = np.arange(len(study_names))
    bars = ax1.bar(x, f1_macro, color=PALETTE['metric_2'], alpha=0.85)
    ax1.set_ylabel('F1 (Macro)', fontsize=11)
    ax1.set_title('F1 (Macro) by Study Pair', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(study_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.5, linewidth=0.8)
    for bar, val in zip(bars, f1_macro):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    colors = [PALETTE['alert'] if r > 5 else PALETTE['good'] for r in imbalance_ratios]
    bars = ax2.bar(study_names, imbalance_ratios, color=colors, alpha=0.8)
    ax2.set_ylabel('Imbalance Ratio', fontsize=11)
    ax2.set_title('Class Imbalance Ratio by Study Pair\n(Red = High Imbalance > 5:1)', fontsize=12, fontweight='bold')
    ax2.axhline(y=5, color=PALETTE['alert'], linestyle='--', alpha=0.7, linewidth=1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    for bar, ratio in zip(bars, imbalance_ratios):
        ax2.annotate(f'{ratio:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    ax3 = axes[2]
    ax3.axis('off')
    table_data = []
    for m in all_study_metrics:
        table_data.append([
            m['study_pair'],
            f"{m['metrics']['f1_macro']:.3f}",
            f"{m['distribution']['imbalance_ratio']:.1f}",
            f"{m['distribution']['total_samples']}"
        ])
    table = ax3.table(
        cellText=table_data,
        colLabels=['Study Pair', 'F1(M)', 'Imbalance', 'N'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax3.set_title('Summary Table', fontsize=12, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-study comparison to: {output_path}")
    plt.close()
    return fig


# ============================================================================
# CROSS-MODEL / CROSS-MODE COMPARISON FUNCTIONS
# ============================================================================

def plot_metric_heatmap(master_df: pd.DataFrame, metric: str, output_path: str = None,
                        title: str = None):
    """
    Plot a heatmap of a given metric with rows = (model, mode) and columns = study pair.
    OO rows are labelled as 'OO (baseline)' instead of showing model 'N/A'.
    """
    df = master_df.copy()
    # Relabel OO so it reads as a baseline, not as a model
    df.loc[df['mode'] == 'OO', 'model'] = 'OO (baseline)'
    df.loc[df['mode'] == 'OO', 'mode'] = '—'

    pivot = df.pivot_table(
        index=['model', 'mode'], columns='study_pair', values=metric, aggfunc='first'
    )
    # Sort rows by mean metric value (best at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2.5), max(6, len(pivot) * 0.7)))
    
    sns.heatmap(pivot.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    ax.set_title(title or f'{metric.replace("_", " ").title()} by Model × Mode × Study Pair',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Model / Mode', fontsize=11)
    ax.set_xlabel('Study Pair', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved heatmap to: {output_path}")
    plt.close()
    return fig


def plot_aggregate_comparison(master_df: pd.DataFrame, output_path: str = None):
    """
    Single horizontal grouped-bar chart comparing all models across all modes
    by F1 (Weighted), averaged across study pairs.
    Rows = models (sorted best→worst), bars grouped/coloured by mode.
    OO is model-independent: shown as a separate hatched bar at the bottom.
    """
    # Separate OO baseline from model-specific results
    oo_df = master_df[master_df['mode'] == 'OO']
    model_df = master_df[(master_df['mode'] != 'OO') & (master_df['model'] != 'N/A')]
    has_oo = not oo_df.empty
    logger.info(f"  plot_aggregate_comparison: OO rows in master_df = {len(oo_df)} "
                f"({'will show baseline' if has_oo else 'OO NOT FOUND — check file discovery'})")

    agg = model_df.groupby(['model', 'mode']).agg(
        f1_macro=('f1_macro', 'mean'),
    ).reset_index()

    if agg.empty:
        logger.info("  No model-specific results to plot in aggregate comparison.")
        return None

    models = (agg.groupby('model')['f1_macro'].mean()
                  .sort_values(ascending=True).index.tolist())
    modes  = sorted(agg['mode'].unique())
    n_modes = len(modes)

    has_oo = not oo_df.empty
    gap = 0.15  # visual gap between OO and model rows

    bar_h   = 0.8 / max(n_modes, 1)
    # Model rows start after the OO row + gap
    if has_oo:
        y_base = np.arange(len(models)) + 1 + gap
    else:
        y_base = np.arange(len(models))

    fig, ax = plt.subplots(
        figsize=(10, max(4, (len(models) + has_oo) * 0.65 + 1.5))
    )

    # --- OO bar (single, hatched) ---
    if has_oo:
        oo_mean = oo_df['f1_macro'].mean()
        ax.barh(0, oo_mean, height=0.5,
                color=_mode_color('OO'), alpha=0.85,
                edgecolor='white', linewidth=0.3,
                hatch='///', label='OO (no model)')
        ax.text(oo_mean + 0.006, 0, f'{oo_mean:.2f}',
                va='center', ha='left', fontsize=8,
                fontweight='bold', color=_mode_color('OO'))

    # --- Model-specific bars ---
    for i, mode in enumerate(modes):
        mode_data = agg[agg['mode'] == mode].set_index('model')
        vals = [mode_data.loc[m, 'f1_macro'] if m in mode_data.index else 0
                for m in models]
        offset = (i - (n_modes - 1) / 2) * bar_h
        bars = ax.barh(y_base + offset, vals, height=bar_h,
                       label=mode, color=_mode_color(mode),
                       alpha=0.85, edgecolor='white', linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(v + 0.006, bar.get_y() + bar.get_height() / 2,
                        f'{v:.2f}', va='center', ha='left', fontsize=7.5,
                        fontweight='bold', color=_mode_color(mode))

    # --- Y-axis ---
    all_ticks = ([0] if has_oo else []) + y_base.tolist()
    all_labels = (['OO (baseline)'] if has_oo else []) + models
    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.set_xlabel('F1 (Macro)', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.axvline(x=0.8, color=PALETTE['good'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.legend(title='Mode', fontsize=9, title_fontsize=10,
              loc='lower right', framealpha=0.9)
    ax.set_title('F1 (Macro) by Model × Mode  (avg. across study pairs)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved aggregate comparison to: {output_path}")
    plt.close()
    return fig


def plot_mode_comparison_per_model(master_df: pd.DataFrame, output_path: str = None):
    """
    Single vertical grouped-bar chart comparing all modes across all models
    by F1 (Macro), averaged across study pairs.
    X-axis = modes, bars grouped/coloured by model.
    OO is model-independent: shown as a single hatched bar at its own x-position.
    """
    # Separate OO baseline from model-specific results
    oo_df = master_df[master_df['mode'] == 'OO']
    model_df = master_df[(master_df['mode'] != 'OO') & (master_df['model'] != 'N/A')]
    logger.info(f"  plot_mode_comparison_per_model: {len(oo_df)} OO rows, {len(model_df)} model rows"
                f"  (modes in master_df: {sorted(master_df['mode'].unique())})")
    logger.info(f"  models in master_df: {sorted(master_df['model'].unique())}")

    agg = model_df.groupby(['model', 'mode']).agg(
        f1_macro=('f1_macro', 'mean'),
    ).reset_index()

    if agg.empty:
        logger.info("  No model-specific results to plot in mode comparison.")
        return None

    model_modes = sorted(agg['mode'].unique())
    models = sorted(agg['model'].unique())
    n_models = len(models)

    _model_cmap = plt.cm.get_cmap('tab10', max(n_models, 3))
    model_colors = {m: _model_cmap(i) for i, m in enumerate(models)}

    # Build x-positions: OO first (position 0), then a gap, then model-specific modes
    has_oo = not oo_df.empty
    gap = 0.15  # visual gap between OO and model-specific modes
    if has_oo:
        x_model_modes = np.arange(len(model_modes)) + 1 + gap
    else:
        x_model_modes = np.arange(len(model_modes)) + 0.25

    bar_w = 0.85 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, (len(model_modes) + has_oo) * 2.5), 6))

    # --- OO bar (single, hatched, centered) ---
    if has_oo:
        oo_mean = oo_df['f1_macro'].mean()
        ax.bar(0, oo_mean, width=0.5,
               color=_mode_color('OO'), alpha=0.85,
               edgecolor='white', linewidth=0.3,
               hatch='///', label='OO (no model)')
        ax.annotate(f'{oo_mean:.2f}',
                    xy=(0, oo_mean), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=_mode_color('OO'))

    # --- Model-specific bars ---
    for i, model in enumerate(models):
        model_data = agg[agg['model'] == model].set_index('mode')
        vals = [model_data.loc[m, 'f1_macro'] if m in model_data.index else 0
                for m in model_modes]
        offset = (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x_model_modes + offset, vals, width=bar_w,
                      label=model, color=model_colors[model],
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6.5,
                            fontweight='bold', color=model_colors[model])

    # --- X-axis ---
    all_ticks = ([0] if has_oo else []) + x_model_modes.tolist()
    all_labels = (['OO'] if has_oo else []) + model_modes
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=11)
    ax.set_xlabel('Mode', fontsize=11)
    ax.set_ylabel('F1 (Macro)', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.legend(title='Model', fontsize=8, title_fontsize=9,
    loc='upper center', bbox_to_anchor=(0.5, -0.1),
    framealpha=0.9, ncol=5)
    ax.set_title('F1 (Macro) by Mode × Model  (avg. across study pairs)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved mode comparison to: {output_path}")
    plt.close()
    return fig


def plot_per_class_heatmap(all_per_class: pd.DataFrame, metric: str = 'f1_score',
                           output_path: str = None):
    """
    Heatmap of per-class F1 (or precision/recall) across model×mode combinations.
    Rows = harmonization class, Columns = model+mode.
    """
    all_per_class = all_per_class.copy()
    # Label OO as baseline instead of showing N/A as a model
    oo_mask = all_per_class['mode'] == 'OO'
    all_per_class.loc[oo_mask, 'config'] = 'OO (baseline)'
    all_per_class.loc[~oo_mask, 'config'] = (
        all_per_class.loc[~oo_mask, 'model'] + ' / ' + all_per_class.loc[~oo_mask, 'mode']
    )

    pivot = all_per_class.pivot_table(
        index='class', columns='config', values=metric, aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5),
                                     max(5, len(pivot) * 0.8)))
    
    sns.heatmap(pivot.astype(float), annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    ax.set_title(f'Per-Class {metric.replace("_", " ").title()} (avg across study pairs)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Mapping Class', fontsize=11)
    ax.set_xlabel('Model / Mode', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved per-class heatmap to: {output_path}")
    plt.close()
    return fig


def plot_best_configs_radar(master_df: pd.DataFrame, output_path: str = None):
    """
    Bar chart showing the best model+mode config per study pair (by F1 macro).
    """
    best = master_df.loc[master_df.groupby('study_pair')['f1_macro'].idxmax()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(best))
    bars = ax.bar(x, best['f1_macro'], color=PALETTE['f1'], alpha=0.85, edgecolor=PALETTE['neutral'], linewidth=0.5)
    
    for bar, (_, row) in zip(bars, best.iterrows()):
        h = bar.get_height()
        label = 'OO (baseline)' if row['mode'] == 'OO' else f"{row['model']}/{row['mode']}"
        ax.annotate(f"{label}\n{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(best['study_pair'], rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('F1 (Macro)', fontsize=11)
    ax.set_title('Best Model+Mode Configuration per Study Pair (by F1 Macro)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.4, linewidth=0.8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved best configs chart to: {output_path}")
    plt.close()
    return fig


def run_full_evaluation(base_dir: str, ground_truth_file: str, model_names: list,
                        modes: list, source_study: str, target_studies: list,
                        output_dir: str):
    """
    Run evaluation across ALL models × modes × study pairs.
    Returns a master DataFrame and per-class DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    master_rows = []
    per_class_rows = []
    all_incorrect = []
    all_not_in_gt = []
    all_not_in_pred = []
    skipped = []
    
    model_specific_modes = [m for m in modes if m != 'OO']
    has_oo = 'OO' in modes
    
    total_combos = (len(model_names) * len(model_specific_modes) * len(target_studies)
                    + (len(target_studies) if has_oo else 0))
    combo_idx = 0
    
    logger.info("=" * 90)
    logger.info("FULL CROSS-MODEL × CROSS-MODE EVALUATION")
    logger.info(f"Models: {model_names}")
    logger.info(f"Model-specific modes: {model_specific_modes}")
    if has_oo:
        logger.info(f"Model-independent modes: OO (ontology only — evaluated once)")
    logger.info(f"Source: {source_study}  →  Targets: {target_studies}")
    logger.info(f"Total configurations: {total_combos}")
    logger.info("=" * 90)
    
    def _evaluate_combo(model_label, mode, data_dir, filename_model):
        nonlocal combo_idx
        results = []
        for target_study in target_studies:
            combo_idx += 1
            study_pair = f"{source_study} → {target_study}"
            config_label = f"{model_label}/{mode}" if model_label != "N/A" else mode
            predict_studies_names = f"{source_study}_{target_study}"
            pred_file = os.path.join(
                data_dir,
                f"time-chf_{target_study}_{filename_model}_{mode}_full.csv"
            )
            
            logger.info(f"\n[{combo_idx}/{total_combos}] {config_label} | {study_pair}")
            logger.info("-" * 60)
            
            if not os.path.exists(pred_file):
                logger.info(f"  SKIPPED: Prediction file not found: {pred_file}")
                skipped.append({
                    'model': model_label, 'mode': mode,
                    'study_pair': study_pair, 'reason': 'file_not_found'
                })
                continue
            
            gt = load_ground_truth(ground_truth_file, source_study, target_study)
            if len(gt) == 0:
                logger.info(f"  SKIPPED: No ground truth for {study_pair}")
                skipped.append({
                    'model': model_label, 'mode': mode,
                    'study_pair': study_pair, 'reason': 'no_ground_truth'
                })
                continue
            
            pred = load_predictions(pred_file)
            
            metrics = compute_comprehensive_metrics(gt, pred)
            dist = analyze_class_distribution(gt, pred, study_pair)
            
            combo_output_dir = os.path.join(output_dir, mode) if model_label == "N/A" else os.path.join(output_dir, model_label, mode)
            os.makedirs(combo_output_dir, exist_ok=True)
            basic = evaluate_predictions(gt, pred, combo_output_dir, predict_studies_names)
            
            master_rows.append({
                'model': model_label,
                'mode': mode,
                'source_study': source_study,
                'target_study': target_study,
                'study_pair': study_pair,
                'total_gt_pairs': basic['total'],
                'correct': basic['correct'],
                'accuracy': round(metrics['accuracy'], 4),
                'f1_weighted': round(metrics['f1_weighted'], 4),
                'f1_macro': round(metrics['f1_macro'], 4),
                'f1_micro': round(metrics['f1_micro'], 4),
                'imbalance_ratio': round(dist['imbalance_ratio'], 2),
                'predicted_not_in_gt': len(basic['not_in_ground_truth']),
                'gt_not_in_pred': len(basic['not_in_predictions']),
                'warnings': '; '.join(dist['warnings']) if dist['warnings'] else '',
            })
            
            pcm = metrics['per_class_metrics']
            for _, row in pcm.iterrows():
                per_class_rows.append({
                    'model': model_label,
                    'mode': mode,
                    'study_pair': study_pair,
                    'class': row['class'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1_score': row['f1_score'],
                    'support': row['support'],
                })
            
            if len(basic['incorrect_predictions']) > 0:
                inc = basic['incorrect_predictions'].copy()
                inc['model'] = model_label
                inc['mode'] = mode
                all_incorrect.append(inc)
            
            if len(basic['not_in_ground_truth']) > 0:
                nig = basic['not_in_ground_truth'].copy()
                nig['model'] = model_label
                nig['mode'] = mode
                all_not_in_gt.append(nig)
            
            if len(basic['not_in_predictions']) > 0:
                nip = basic['not_in_predictions'].copy()
                nip['model'] = model_label
                nip['mode'] = mode
                all_not_in_pred.append(nip)
            
            logger.info(f"  Accuracy: {metrics['accuracy']:.3f}  |  "
                  f"F1(W): {metrics['f1_weighted']:.3f}  |  "
                  f"F1(M): {metrics['f1_macro']:.3f}  |  "
                  f"Correct: {basic['correct']}/{basic['total']}")
            
            plot_confusion_matrix(
                metrics, f"{config_label} | {study_pair}",
                os.path.join(combo_output_dir, f"{predict_studies_names}_confusion_matrix.png")
            )
            plot_per_class_metrics(
                metrics, f"{config_label} | {study_pair}",
                os.path.join(combo_output_dir, f"{predict_studies_names}_per_class_metrics.png")
            )
    
    # -----------------------------------------------------------------
    # 1. Evaluate OO once (model-independent)
    # -----------------------------------------------------------------
    if has_oo:
        import glob
        
        oo_data_dir = None
        oo_filename_model = None
        first_target = target_studies[0]
        
        # Candidate directories to search for OO files
        candidate_dirs = [
            os.path.join(base_dir, "OO"),                          # base_dir/OO/
        ]
        for m in model_names:
            candidate_dirs.append(os.path.join(base_dir, m, "OO")) # base_dir/{model}/OO/
        
        logger.info(f"\n{'='*60}")
        logger.info("OO FILE DISCOVERY — searching for OO prediction files")
        logger.info(f"{'='*60}")
        logger.info(f"  base_dir = {base_dir}")
        logger.info(f"  first_target = {first_target}")
        logger.info(f"  candidate_dirs = {candidate_dirs}")
        
        for cdir in candidate_dirs:
            if not os.path.isdir(cdir):
                logger.info(f"  ✗ {cdir}  — directory does not exist")
                continue
            
            # List ALL csv files in this directory for diagnostics
            all_csvs = glob.glob(os.path.join(cdir, "*.csv"))
            logger.info(f"  ✓ {cdir}  — exists, contains {len(all_csvs)} CSV file(s)")
            if all_csvs:
                for f in sorted(all_csvs)[:10]:  # show up to 10
                    logger.info(f"      {os.path.basename(f)}")
                if len(all_csvs) > 10:
                    logger.info(f"      ... and {len(all_csvs) - 10} more")
            
            # Strategy 1: time-chf_{target}_{model}_OO_full.csv  (model token present)
            pattern1 = os.path.join(cdir, f"time-chf_{first_target}_*_OO_full.csv")
            matches1 = glob.glob(pattern1)
            
            # Strategy 2: time-chf_{target}_OO_full.csv  (no model token)
            pattern2 = os.path.join(cdir, f"time-chf_{first_target}_OO_full.csv")
            match2_exists = os.path.exists(pattern2)
            
            # Strategy 3: broader — any file containing OO and full
            pattern3 = os.path.join(cdir, f"*{first_target}*OO*full*.csv")
            matches3 = glob.glob(pattern3)
            
            # Strategy 4: any *_OO_full.csv at all
            pattern4 = os.path.join(cdir, "*_OO_full.csv")
            matches4 = glob.glob(pattern4)
            
            logger.info(f"    Pattern '{os.path.basename(pattern1)}': {len(matches1)} match(es)")
            logger.info(f"    Pattern '{os.path.basename(pattern2)}': {'FOUND' if match2_exists else 'not found'}")
            logger.info(f"    Pattern '*{first_target}*OO*full*.csv': {len(matches3)} match(es)")
            logger.info(f"    Pattern '*_OO_full.csv': {len(matches4)} match(es)")
            
            if matches1:
                # Extract model token from filename
                basename = os.path.basename(matches1[0])
                prefix = f"time-chf_{first_target}_"
                suffix = "_OO_full.csv"
                if basename.startswith(prefix) and basename.endswith(suffix):
                    oo_filename_model = basename[len(prefix):-len(suffix)]
                    oo_data_dir = cdir
                    logger.info(f"  → MATCH (strategy 1): model token = '{oo_filename_model}'")
                    break
            
            if match2_exists:
                # No model token — need special handling in _evaluate_combo
                # We'll pass empty string as model token
                oo_filename_model = ""
                oo_data_dir = cdir
                logger.info(f"  → MATCH (strategy 2): no model token in filename")
                break
            
            if matches3:
                # Try to parse the first match
                basename = os.path.basename(matches3[0])
                logger.info(f"  → MATCH (strategy 3): {basename}")
                # Try to extract model token
                prefix = f"time-chf_{first_target}_"
                suffix = "_OO_full.csv"
                if basename.startswith(prefix) and basename.endswith(suffix):
                    oo_filename_model = basename[len(prefix):-len(suffix)]
                elif basename == f"time-chf_{first_target}_OO_full.csv":
                    oo_filename_model = ""
                else:
                    # Can't parse, use glob-based approach for _evaluate_combo
                    oo_filename_model = "__GLOB__"
                oo_data_dir = cdir
                break
            
            if matches4:
                basename = os.path.basename(matches4[0])
                logger.info(f"  → MATCH (strategy 4): {basename}")
                oo_filename_model = "__GLOB__"
                oo_data_dir = cdir
                break
        
        if oo_data_dir and oo_filename_model is not None:
            if oo_filename_model == "__GLOB__":
                # Can't use _evaluate_combo's filename template; evaluate directly with glob
                logger.info("  Using glob-based OO evaluation (non-standard filenames)")
                for target_study in target_studies:
                    combo_idx += 1
                    study_pair = f"{source_study} → {target_study}"
                    predict_studies_names = f"{source_study}_{target_study}"
                    
                    # Find the OO file for this target
                    oo_glob = glob.glob(os.path.join(oo_data_dir, f"*{target_study}*OO*full*.csv"))
                    if not oo_glob:
                        logger.info(f"  SKIPPED OO for {target_study}: no matching file in {oo_data_dir}")
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'file_not_found'})
                        continue
                    
                    pred_file = oo_glob[0]
                    logger.info(f"\n[{combo_idx}/{total_combos}] OO | {study_pair}")
                    logger.info(f"  Using: {pred_file}")
                    
                    gt = load_ground_truth(ground_truth_file, source_study, target_study)
                    if len(gt) == 0:
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'no_ground_truth'})
                        continue
                    
                    pred = load_predictions(pred_file)
                    metrics = compute_comprehensive_metrics(gt, pred)
                    dist = analyze_class_distribution(gt, pred, study_pair)
                    combo_output_dir = os.path.join(output_dir, "OO")
                    os.makedirs(combo_output_dir, exist_ok=True)
                    basic = evaluate_predictions(gt, pred, combo_output_dir, predict_studies_names)
                    
                    master_rows.append({
                        'model': 'N/A', 'mode': 'OO',
                        'source_study': source_study, 'target_study': target_study,
                        'study_pair': study_pair,
                        'total_gt_pairs': basic['total'], 'correct': basic['correct'],
                        'accuracy': round(metrics['accuracy'], 4),
                        'f1_weighted': round(metrics['f1_weighted'], 4),
                        'f1_macro': round(metrics['f1_macro'], 4),
                        'f1_micro': round(metrics['f1_micro'], 4),
                        'imbalance_ratio': round(dist['imbalance_ratio'], 2),
                        'predicted_not_in_gt': len(basic['not_in_ground_truth']),
                        'gt_not_in_pred': len(basic['not_in_predictions']),
                        'warnings': '; '.join(dist['warnings']) if dist['warnings'] else '',
                    })
                    for _, row in metrics['per_class_metrics'].iterrows():
                        per_class_rows.append({
                            'model': 'N/A', 'mode': 'OO', 'study_pair': study_pair,
                            'class': row['class'], 'precision': row['precision'],
                            'recall': row['recall'], 'f1_score': row['f1_score'],
                            'support': row['support'],
                        })
                    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}  |  "
                                f"F1(W): {metrics['f1_weighted']:.3f}  |  "
                                f"Correct: {basic['correct']}/{basic['total']}")
            elif oo_filename_model == "":
                # Filename has no model token: time-chf_{target}_OO_full.csv
                # We need a custom call since _evaluate_combo adds _{model}_ in the middle
                logger.info("  Using no-model-token OO evaluation")
                for target_study in target_studies:
                    combo_idx += 1
                    study_pair = f"{source_study} → {target_study}"
                    predict_studies_names = f"{source_study}_{target_study}"
                    pred_file = os.path.join(oo_data_dir, f"time-chf_{target_study}_OO_full.csv")
                    
                    logger.info(f"\n[{combo_idx}/{total_combos}] OO | {study_pair}")
                    
                    if not os.path.exists(pred_file):
                        logger.info(f"  SKIPPED: {pred_file} not found")
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'file_not_found'})
                        continue
                    
                    gt = load_ground_truth(ground_truth_file, source_study, target_study)
                    if len(gt) == 0:
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'no_ground_truth'})
                        continue
                    
                    pred = load_predictions(pred_file)
                    metrics = compute_comprehensive_metrics(gt, pred)
                    dist = analyze_class_distribution(gt, pred, study_pair)
                    combo_output_dir = os.path.join(output_dir, "OO")
                    os.makedirs(combo_output_dir, exist_ok=True)
                    basic = evaluate_predictions(gt, pred, combo_output_dir, predict_studies_names)
                    
                    master_rows.append({
                        'model': 'N/A', 'mode': 'OO',
                        'source_study': source_study, 'target_study': target_study,
                        'study_pair': study_pair,
                        'total_gt_pairs': basic['total'], 'correct': basic['correct'],
                        'accuracy': round(metrics['accuracy'], 4),
                        'f1_weighted': round(metrics['f1_weighted'], 4),
                        'f1_macro': round(metrics['f1_macro'], 4),
                        'f1_micro': round(metrics['f1_micro'], 4),
                        'imbalance_ratio': round(dist['imbalance_ratio'], 2),
                        'predicted_not_in_gt': len(basic['not_in_ground_truth']),
                        'gt_not_in_pred': len(basic['not_in_predictions']),
                        'warnings': '; '.join(dist['warnings']) if dist['warnings'] else '',
                    })
                    for _, row in metrics['per_class_metrics'].iterrows():
                        per_class_rows.append({
                            'model': 'N/A', 'mode': 'OO', 'study_pair': study_pair,
                            'class': row['class'], 'precision': row['precision'],
                            'recall': row['recall'], 'f1_score': row['f1_score'],
                            'support': row['support'],
                        })
                    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}  |  "
                                f"F1(W): {metrics['f1_weighted']:.3f}  |  "
                                f"Correct: {basic['correct']}/{basic['total']}")
            else:
                # Standard case: model token found, use _evaluate_combo
                _evaluate_combo("N/A", "OO", oo_data_dir, oo_filename_model)
        else:
            logger.info("\n  ⚠️  SKIPPED OO: Could not find any OO prediction files.")
            logger.info(f"     Searched directories: {candidate_dirs}")
            logger.info(f"     TIP: Check your OO output directory and filenames.")
            logger.info(f"     Expected patterns like: time-chf_{{target}}_*_OO_full.csv")
            logger.info(f"     Or: time-chf_{{target}}_OO_full.csv")
            for target_study in target_studies:
                combo_idx += 1
                skipped.append({
                    'model': 'N/A', 'mode': 'OO',
                    'study_pair': f"{source_study} → {target_study}",
                    'reason': 'file_not_found'
                })
    
    # -----------------------------------------------------------------
    # 2. Evaluate model-specific modes (NE, OEH, OEC, OED, ...)
    # -----------------------------------------------------------------
    for model_name in model_names:
        for mode in model_specific_modes:
            data_dir = os.path.join(base_dir, model_name, mode)
            _evaluate_combo(model_name, mode, data_dir, model_name)
    
    # =========================================================================
    # BUILD MASTER DATAFRAMES
    # =========================================================================
    master_df = pd.DataFrame(master_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    skipped_df = pd.DataFrame(skipped) if skipped else pd.DataFrame()
    
    if master_df.empty:
        logger.info("\nNo valid results found. Check that prediction files exist.")
        return master_df, per_class_df
    
    # =========================================================================
    # SAVE MASTER CSVs
    # =========================================================================
    master_path = os.path.join(output_dir, "master_evaluation_summary.csv")
    master_df.to_csv(master_path, index=False)
    logger.info(f"\n  Saved master summary ({len(master_df)} rows) to: {master_path}")
    
    per_class_path = os.path.join(output_dir, "master_per_class_metrics.csv")
    per_class_df.to_csv(per_class_path, index=False)
    logger.info(f"  Saved per-class metrics ({len(per_class_df)} rows) to: {per_class_path}")
    
    if all_incorrect:
        inc_df = pd.concat(all_incorrect, ignore_index=True)
        inc_path = os.path.join(output_dir, "all_incorrect_predictions.csv")
        inc_df.to_csv(inc_path, index=False)
        logger.info(f"  Saved all incorrect predictions ({len(inc_df)} rows) to: {inc_path}")
    
    if all_not_in_gt:
        nig_df = pd.concat(all_not_in_gt, ignore_index=True)
        nig_path = os.path.join(output_dir, "all_predicted_not_in_ground_truth.csv")
        nig_df.to_csv(nig_path, index=False)
        logger.info(f"  Saved all predicted-not-in-GT ({len(nig_df)} rows) to: {nig_path}")
    
    if all_not_in_pred:
        nip_df = pd.concat(all_not_in_pred, ignore_index=True)
        nip_path = os.path.join(output_dir, "all_gt_not_in_predictions.csv")
        nip_df.to_csv(nip_path, index=False)
        logger.info(f"  Saved all GT-not-in-predictions ({len(nip_df)} rows) to: {nip_path}")
    
    if not skipped_df.empty:
        skip_path = os.path.join(output_dir, "skipped_configurations.csv")
        skipped_df.to_csv(skip_path, index=False)
        logger.info(f"  Saved skipped configurations ({len(skipped_df)} rows) to: {skip_path}")
    
    # =========================================================================
    # GENERATE CROSS-MODEL / CROSS-MODE PLOTS
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("GENERATING CROSS-MODEL × CROSS-MODE COMPARISON PLOTS")
    logger.info("=" * 90)
    
    for metric in ['f1_macro']:
        plot_metric_heatmap(
            master_df, metric,
            os.path.join(output_dir, f"heatmap_{metric}.png"),
            title=f'{metric.replace("_", " ").title()} — All Models × Modes × Study Pairs'
        )
    
    plot_aggregate_comparison(
        master_df,
        os.path.join(output_dir, "comparison_models_by_mode.png")
    )
    
    plot_mode_comparison_per_model(
        master_df,
        os.path.join(output_dir, "comparison_modes_by_model.png")
    )
    
    if not per_class_df.empty:
        for metric in ['f1_score', 'precision', 'recall']:
            plot_per_class_heatmap(
                per_class_df.copy(), metric,
                os.path.join(output_dir, f"per_class_heatmap_{metric}.png")
            )
    
    plot_best_configs_radar(
        master_df,
        os.path.join(output_dir, "best_config_per_study_pair.png")
    )
    
    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("FINAL MASTER SUMMARY")
    logger.info("=" * 90)
    logger.info(master_df.to_string(index=False))
    
    # Ranking by F1 Macro (averaged across study pairs)
    ranking = master_df.groupby(['model', 'mode']).agg({
        'f1_macro': 'mean',
    }).reset_index().sort_values('f1_macro', ascending=False)
    ranking.columns = ['Model', 'Mode', 'Avg F1(M)']
    
    logger.info("\n" + "=" * 90)
    logger.info("RANKING BY AVG F1 MACRO (across all study pairs)")
    logger.info("=" * 90)
    for rank, (_, row) in enumerate(ranking.iterrows(), 1):
        avg_f1_macro = round(row['Avg F1(M)'], 2)
        model_label = 'OO (baseline)' if row['Mode'] == 'OO' else f"{row['Model']:>10s} / {row['Mode']:<5s}"
        logger.info(f"  #{rank}  {model_label}  |  "
              f"F1(M)={avg_f1_macro:>5.2f}")
    
    ranking_path = os.path.join(output_dir, "ranking_by_f1_macro.csv")
    ranking.to_csv(ranking_path, index=False)
    logger.info(f"\n  Saved ranking to: {ranking_path}")
    
    return master_df, per_class_df


def run_comprehensive_evaluation(ground_truth_file: str, predictions_files: list, 
                                  source_study: str, target_studies: list,
                                  output_dir: str):
    """Run comprehensive evaluation across all study pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_study_metrics = []
    all_results_summary = []
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION WITH CLASS BALANCE ANALYSIS")
    logger.info("=" * 80)
    
    for target_study, pred_file in zip(target_studies, predictions_files):
        study_pair = f"{source_study} → {target_study}"
        predict_studies_names = f"{source_study}_{target_study}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {study_pair}")
        logger.info(f"{'='*60}")
        
        gt = load_ground_truth(ground_truth_file, source_study, target_study)
        pred = load_predictions(pred_file)
        
        if len(gt) == 0:
            logger.info(f"⚠️  No ground truth found for {study_pair}")
            continue
        
        basic_results = evaluate_predictions(gt, pred, output_dir, predict_studies_names)
        dist_results = analyze_class_distribution(gt, pred, study_pair)
        metrics = compute_comprehensive_metrics(gt, pred)
        
        logger.info(f"\n📊 CLASS DISTRIBUTION COMPARISON:")
        logger.info(dist_results['comparison'].to_string(index=False))
        
        if dist_results['warnings']:
            logger.info(f"\n⚠️  WARNINGS:")
            for w in dist_results['warnings']:
                logger.info(f"   {w}")
        
        logger.info(f"\n📈 AGGREGATE METRICS:")
        logger.info(f"   Accuracy:      {metrics['accuracy']:.3f}")
        logger.info(f"   F1 (Weighted): {metrics['f1_weighted']:.3f}")
        logger.info(f"   F1 (Macro):    {metrics['f1_macro']:.3f}")
        logger.info(f"   F1 (Micro):    {metrics['f1_micro']:.3f}")
        
        logger.info(f"\n📋 PER-CLASS METRICS:")
        logger.info(metrics['per_class_metrics'].to_string(index=False))
        
        logger.info(f"\n📝 CLASSIFICATION REPORT:")
        logger.info(metrics['classification_report'])
        
        safe_name = predict_studies_names
        
        plot_class_distribution_comparison(
            dist_results, 
            os.path.join(output_dir, f"{safe_name}_distribution.png")
        )
        plot_confusion_matrix(
            metrics, study_pair,
            os.path.join(output_dir, f"{safe_name}_confusion_matrix.png")
        )
        plot_per_class_metrics(
            metrics, study_pair,
            os.path.join(output_dir, f"{safe_name}_per_class_metrics.png")
        )
        
        all_study_metrics.append({
            'study_pair': study_pair,
            'metrics': metrics,
            'distribution': dist_results,
            'basic_results': basic_results
        })
        
        all_results_summary.append({
            'source_study': source_study,
            'target_study': target_study,
            'total_pairs': basic_results['total'],
            'correct': basic_results['correct'],
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro'],
            'imbalance_ratio': dist_results['imbalance_ratio'],
            'warnings': '; '.join(dist_results['warnings']) if dist_results['warnings'] else ''
        })
    
    if len(all_study_metrics) > 1:
        plot_multi_study_comparison(
            all_study_metrics,
            os.path.join(output_dir, "multi_study_comparison.png")
        )
    
    summary_df = pd.DataFrame(all_results_summary)
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n💾 Saved evaluation summary to: {summary_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(summary_df.to_string(index=False))
    
    return all_study_metrics, summary_df


def evaluate_f1_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Evaluates the predictions against the ground truth and returns F1 score."""
    merged_df = pd.merge(ground_truth, predictions, on=['src_var', 'tgt_var'], 
                         how='left', suffixes=('_gt', '_pred'))
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')
    
    y_true = merged_df['correct class']
    y_pred = merged_df['predicted class']
    
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    logger.info("Classification Report:")
    logger.info(report)
    
    return {
        'f1_score': f1_weighted,
        'f1_macro': f1_macro
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive cross-model × cross-mode evaluation")
    parser.add_argument('--models', nargs='+',
                        default=["biolord+no_llm","sapbert+no_llm", "openai+no_llm", "qwen3-0.6b+no_llm",
                        "biolord+gemini-2.5-flash-lite", "sapbert+gemini-2.5-flash-lite", "openai+gemini-2.5-flash-lite", "qwen3-0.6b+gemini-2.5-flash-lite",
                        "biolord+gpt-oss-120b", "sapbert+gpt-oss-120b", "openai+gpt-oss-120b", "qwen3-0.6b+gpt-oss-120b",
                        "sapbert+llama-4-maverick"
                        ],
                        help='Embedding model names to evaluate')
    parser.add_argument('--modes', nargs='+', default=["OO","NE", "OEH"],
                        help='Mapping modes to evaluate')
    parser.add_argument('--source', default="time-chf", help='Source study name')
    parser.add_argument('--targets', nargs='+', default=["aric","aachen-hf", "gissi-hf", "viennahf-register"],
                        help='Target study names')
    parser.add_argument('--single-model', type=str, default=None,
                        help='Run evaluation for a single model only (e.g. sapbert)')
    parser.add_argument('--single-mode', type=str, default=None,
                        help='Run evaluation for a single mode only (e.g. NE)')
    
    args = parser.parse_args()
    
    base_dir = "/Users/komalgilani/phd_projects/CohortVarLinker/data/output/cross_mapping"
    ground_truth_file = "/Users/komalgilani/phd_projects/CohortVarLinker/data/ground_truth_pairs.csv"
    output_dir = os.path.join(base_dir, "evaluation_results")
    
    model_names = [args.single_model] if args.single_model else args.models
    modes = [args.single_mode] if args.single_mode else args.modes
    
    master_df, per_class_df = run_full_evaluation(
        base_dir=base_dir,
        ground_truth_file=ground_truth_file,
        model_names=model_names,
        modes=modes,
        source_study=args.source,
        target_studies=args.targets,
        output_dir=output_dir,
    )
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
