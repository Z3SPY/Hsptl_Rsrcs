#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
merge_patient_one_row.py

Reads 'transfers.csv', merges all admissions (hadm_ids) for each subject_id
into a single row. Aggregates:
  • all_hadm_ids: list of admissions
  • earliest_intime / latest_outtime: overall timeline
  • total_los_minutes: from earliest to latest
  • careunits_visited: all unique careunits across visits
  • severity_label: max severity across entire patient history

Finally, outputs:
  1) one_row_per_subject.csv
  2) a bar chart for severity distribution across patients
  3) a bar chart for how many admissions each patient had
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Paths
    TRANSFERS_CSV = "unused/transfers.csv/transfers.csv"
    OUTPUT_CSV = "one_row_per_subject.csv"

    # 2. Load 'transfers.csv'
    print("[INFO] Loading CSV...")
    df = pd.read_csv(TRANSFERS_CSV, parse_dates=['intime', 'outtime'])

    # 3. Optional cleaning
    df = df[df['careunit'].notnull()]  # remove rows without careunit
    df['careunit'] = df['careunit'].str.strip()  # remove leading/trailing spaces
    df = df.sort_values(by=['subject_id', 'intime'])

    # 4. Group by subject_id only
    grouped = df.groupby('subject_id')

    def get_patient_aggregate(subdf):
        """
        subdf: all rows for one subject_id across multiple admissions (hadm_ids).
        We'll:
         - gather unique hadm_ids
         - gather all careunits
         - earliest and latest times
         - total LOS in minutes
         - severity label from entire path
        """
        hadms = subdf['hadm_id'].dropna().unique().tolist()  # all admissions for this patient
        earliest_intime = subdf['intime'].min()
        latest_outtime = subdf['outtime'].max()

        total_los_minutes = None
        if pd.notnull(earliest_intime) and pd.notnull(latest_outtime):
            total_los_minutes = (latest_outtime - earliest_intime).total_seconds() / 60.0

        careunits_visited = subdf['careunit'].unique().tolist()

        # define a severity labeling function
        # e.g., severity 3 if 'ICU' or 'MICU', severity 2 if 'Med', 'Surg', 'Transplant', else 1
        severity_score = 1
        path_str = " ".join(careunits_visited).lower()
        if "icu" in path_str or "micu" in path_str:
            severity_score = 3
        elif ("med" in path_str or "surg" in path_str or 
              "transplant" in path_str or "trauma" in path_str):
            severity_score = 2

        return pd.Series({
            'all_hadm_ids': hadms,
            'earliest_intime': earliest_intime,
            'latest_outtime': latest_outtime,
            'total_los_minutes': total_los_minutes,
            'careunits_visited': careunits_visited,
            'severity_label': severity_score
        })

    # 5. Apply aggregator
    print("[INFO] Merging all admissions into one row per subject_id...")
    merged_df = grouped.apply(get_patient_aggregate).reset_index()

    # 6. Save CSV
    print("[INFO] Saving CSV => one_row_per_subject.csv")
    merged_df.to_csv(OUTPUT_CSV, index=False)

    # 7. Graph #1: Severity distribution
    print("[INFO] Plotting severity distribution bar chart...")
    severity_counts = merged_df['severity_label'].value_counts().sort_index()
    plt.figure()
    severity_counts.plot(kind='bar')
    plt.title("Patient-level Severity Distribution")
    plt.xlabel("Severity Label")
    plt.ylabel("Count of Patients")
    plt.savefig("patient_severity_distribution.png")
    plt.close()
    print("[INFO] Saved => patient_severity_distribution.png")

    # 8. Graph #2: Admission count distribution
    print("[INFO] Plotting how many admissions each patient had...")
    merged_df['hadm_count'] = merged_df['all_hadm_ids'].apply(lambda x: len(x))
    hadm_counts = merged_df['hadm_count'].value_counts().sort_index()
    plt.figure()
    hadm_counts.plot(kind='bar')
    plt.title("Number of Admissions per Patient")
    plt.xlabel("Admissions Count")
    plt.ylabel("Count of Patients")
    plt.savefig("patient_admission_count.png")
    plt.close()
    print("[INFO] Saved => patient_admission_count.png")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
