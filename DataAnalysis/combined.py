#!/usr/bin/env python3
"""
combined.py

Combines triage, edstays, and transfers CSVs to compute and visualize length-of-stay (LOS)
by patient acuity across hospital care unit groups.

Defaults assume your project root has:
  unused/triage.csv/triage.csv
  unused/edstays.csv/edstays.csv
  unused/transfers.csv/transfers.csv

Outputs to `DataAnalysis/graphss/`:
  - acuity_duration_stats.csv
  - kruskal_results.csv
  - acuity_los_barplot.png

Usage:
  Simply run `python combined.py` (no flags needed).

Statistics:
  - Kruskal-Wallis test: a nonparametric method to detect differences in medians across
    multiple independent groups (acuity levels) within each care unit.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal  # nonparametric test for difference in group distributions

# ----------------------- Configuration & Logging -----------------------
DEFAULT_ROOT = Path(__file__).parent.parent
DEFAULT_PATHS = {
    'triage':    DEFAULT_ROOT / 'unused' / 'triage.csv'    / 'triage.csv',
    'edstays':   DEFAULT_ROOT / 'unused' / 'edstays.csv'   / 'edstays.csv',
    'transfers': DEFAULT_ROOT / 'unused' / 'transfers.csv' / 'transfers.csv',
}
# Save into DataAnalysis/graphss as requested
OUTPUT_DIR = DEFAULT_ROOT / 'DataAnalysis' / 'graphss'

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------ Care Unit Helpers ---------------------------
def simplify(u: str) -> str:
    m = {
        'Emergency Department':'ED', 'Emergency Department Observation':'ED Obs',
        'Medical Intensive Care Unit (MICU)':'MICU', 'Surgical Intensive Care Unit (SICU)':'SICU',
        'Neuro Surgical Intensive Care Unit (Neuro SICU)':'Neuro ICU', 'Cardiac Vascular Intensive Care Unit (CVICU)':'Cardiac ICU',
        'Intensive Care Unit (ICU)':'ICU', 'Medical/Surgical Intensive Care Unit (MICU/SICU)':'ICU',
        'Med/Surg':'MedSurg', 'Med/Surg/Trauma':'MedSurg', 'Med/Surg/GYN':'MedSurg',
        'Medicine/Cardiology':'Cardiology', 'Coronary Care Unit (CCU)':'Cardiology',
        'Hematology/Oncology':'Oncology', 'Hematology/Oncology Intermediate':'Onc-Step',
        'Neuro Intermediate':'Neuro-Step', 'Medical/Surgical (Gynecology)':'GYN-Step',
        'Transplant':'Tx Ward', 'Surgery':'Surg Ward', 'Surgery/Trauma':'Surg Ward',
        'Surgery/Vascular/Intermediate':'Surg-Step', 'Thoracic Surgery':'Surg Ward',
        'Observation':'Obs Unit', 'Discharge Lounge':'Discharge',
        'UNKNOWN':'Discharge', 'Unknown':'Discharge',
        'Labor & Delivery':'L&D', 'Obstetrics Postpartum':'OB Post',
        'Obstetrics Antepartum':'OB Ante', 'Psychiatry':'Psych',
        'Nursery':'Nursery', 'Special Care Nursery (SCN)':'Nursery'
    }
    return m.get(u, u[:12]) if pd.notna(u) else 'UNKNOWN'

def group(u: str) -> str:
    if u in ['ED','ED Obs']:
        return 'Emergency'
    if u in ['MICU','SICU','ICU','Neuro ICU','Cardiac ICU']:
        return 'ICU'
    if 'Step' in u:
        return 'Step-Down'
    if u in ['MedSurg','Cardiology','Oncology','Tx Ward','OB Post','OB Ante']:
        return 'Medical Ward'
    if 'Surg' in u:
        return 'Surgical Ward'
    if u in ['L&D','OB Ante','OB Post']:
        return 'Obstetrics'
    if u in ['Nursery','SCN']:
        return 'Pediatrics'
    if u == 'Psych':
        return 'Psychiatry'
    if u == 'Discharge':
        return 'Discharge'
    if 'Obs' in u:
        return 'Observation'
    return 'Other'

# ---------------------------- Main Logic ------------------------------
def main():
    # Load
    tri = pd.read_csv(DEFAULT_PATHS['triage'])
    ed  = pd.read_csv(DEFAULT_PATHS['edstays'])
    tf  = pd.read_csv(DEFAULT_PATHS['transfers'], parse_dates=['intime','outtime'])
    logger.info('Files loaded: triage(%d), edstays(%d), transfers(%d)', len(tri), len(ed), len(tf))

    # Triage → hadm_id
    tri = tri.dropna(subset=['acuity','subject_id','stay_id'])
    tri['acuity'] = pd.to_numeric(tri['acuity'], errors='coerce')
    tri = (tri.merge(ed[['subject_id','stay_id','hadm_id']], on=['subject_id','stay_id'], how='left')
               .dropna(subset=['hadm_id','acuity']).drop_duplicates('hadm_id'))
    tri['hadm_id'] = tri['hadm_id'].astype(int)
    logger.info('Triage mapped: %d unique hadm_ids', len(tri))

    # Clean transfers
    tf = tf.dropna(subset=['intime','outtime','careunit','hadm_id']).copy()
    tf['duration_min'] = (tf['outtime'] - tf['intime']).dt.total_seconds() / 60
    before = len(tf)
    tf = tf[(tf['duration_min'] >= 1) & (tf['duration_min'] <= 30*24*60)]
    logger.info('Filtered out %d invalid durations', before - len(tf))

    tf['hadm_id'] = tf['hadm_id'].astype(int)
    tf['unit_simple'] = tf['careunit'].apply(simplify)
    tf['unit_group']  = tf['unit_simple'].apply(group)
    logger.info('Transfers cleaned: %d rows remain', len(tf))

    # IQR trimming
    cleaned = []
    for grp_name, sub in tf.groupby('unit_group'):
        q1,q3 = sub['duration_min'].quantile([0.25,0.75])
        iqr = q3 - q1
        kept = sub[sub['duration_min'].between(q1-1.5*iqr, q3+1.5*iqr)]
        if len(kept) < 0.5 * len(sub):
            lo, hi = sub['duration_min'].quantile([0.01,0.99])
            kept = sub[sub['duration_min'].between(lo, hi)]
        cleaned.append(kept)
    tf_clean = pd.concat(cleaned)
    logger.info('After IQR trimming: %d rows', len(tf_clean))

    # Merge acuity
    df = tf_clean.merge(tri[['hadm_id','acuity']], on='hadm_id', how='inner')
    logger.info('Merged records: %d rows with acuity', len(df))

    # Compute stats with debug logs
    stats = []
    logger.info('Computing mean and variance per (acuity, unit_group)...')
    for (acu, grp_name), sub in df.groupby(['acuity','unit_group']):
        n = len(sub)
        mean = sub['duration_min'].mean()
        var = sub['duration_min'].var(ddof=1)
        sd = np.sqrt(var)
        ci95 = 1.96 * sd / np.sqrt(n)
        lower_err = min(ci95, mean)
        logger.debug(f'Acuity={acu}, Group={grp_name}: n={n}, mean={mean:.2f}, var={var:.2f}')
        stats.append({
            'acuity': acu,
            'unit_group': grp_name,
            'n': n,
            'mean_min': mean,
            'var_min': var,
            'ci95_min': ci95,
            'lower_err': lower_err
        })
    stats = pd.DataFrame(stats)
    logger.info('Stats computed: %d groups', len(stats))

    # Kruskal-Wallis
    kw_results = []
    for grp_name, sub in df.groupby('unit_group'):
        groups = [g['duration_min'].values for _, g in sub.groupby('acuity')]
        h_stat, p_val = kruskal(*groups)
        kw_results.append({'unit_group': grp_name, 'H': h_stat, 'p': p_val})
    kw_df = pd.DataFrame(kw_results)
    logger.info('Kruskal-Wallis tests complete')

    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)
    stats.to_csv(OUTPUT_DIR / 'acuity_duration_stats.csv', index=False)
    kw_df.to_csv(OUTPUT_DIR / 'kruskal_results.csv', index=False)
    logger.info('Results written to %s', OUTPUT_DIR)

    # Plot
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=stats, x='unit_group', y='mean_min', hue='acuity', ci=None, palette='tab10'
    )
    for i, row in stats.iterrows():
        x = list(stats['unit_group'].unique()).index(row['unit_group'])
        ax.errorbar(
            x + (row['acuity'] - stats['acuity'].mean())*0.2,
            row['mean_min'],
            yerr=[[row['lower_err']], [row['ci95_min']]],
            fmt='none', c='k', capsize=3
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('Mean LOS (min) ±95% CI')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'acuity_los_barplot.png')
    plt.show()

if __name__ == '__main__':
    main()
