from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    spearmanr,
    ranksums,
    kruskal
)

import statsmodels.formula.api as smf


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_spike_sz_pipeline_clean():

    # ========================================================
    # RNG
    # ========================================================
    np.random.seed(1)

    # ========================================================
    # PATHS
    # ========================================================
    spike_summary_csv = Path("/Users/edwardyao/Documents/PURM/data/spike_counts.csv")
    report_csv = Path("/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv")

    fig1_out = Path("/Users/edwardyao/Documents/PURM/spike_project_output/Fig1.png")
    fig2_out = Path("/Users/edwardyao/Documents/PURM/spike_project_output/Fig2.png")
    figS1_out = Path("/Users/edwardyao/Documents/PURM/spike_project_output/FigS1.png")
    figS2_out = Path("/Users/edwardyao/Documents/PURM/spike_project_output/FigS2.png")
    figMain_out = Path("/Users/edwardyao/Documents/PURM/spike_project_output/FigModel.png")
    Table1Csv = Path("/Users/edwardyao/Documents/PURM/spike_project_output/Table1.csv")
    tableS1Csv = Path("/Users/edwardyao/Documents/PURM/spike_project_output/TableS1.csv")
    resultsHtml = Path("/Users/edwardyao/Documents/PURM/spike_project_output/results_summary.html")


    # ========================================================
    # PARAMETERS
    # ======================================================== 
    MAX_ROUTINE_HOURS = 4

    NESD_LABEL = "Non-Epileptic Seizure Disorder"

    bad_types = [
        "uncertain if epilepsy",
        "unknown or mrn not found",
        ""
    ]

    canonical3 = [
        "General",
        "Temporal",
        "Frontal"
    ]

    EPS_RATE = 30e-3

    count_col = "count_0_46"
    dur_col = "Duration_sec"

    allowable_visits = [
        "CONSULT VISIT",
        "ESTABLISHED PATIENT VISIT",
        "FOLLOW-UP PATIENT CLINIC",
        "NEW PATIENT CLINIC",
        "NEW PATIENT VISIT",
        "NPV MANAGEMENT DURING COVID-19",
        "NPV NEUROLOGY",
        "RETURN ANNUAL VISIT",
        "RETURN PATIENT EXTENDED",
        "RETURN PATIENT VISIT",
        "RPV MANAGEMENT DURING COVID-19",
        "TELEHEALTH VIDEO VISIT RETURN"
    ]

    # ========================================================
    # LOAD DATA
    # ========================================================
    spike_df = pd.read_csv(spike_summary_csv)

    require_cols(
        spike_df,
        [
            "Patient",
            "Session",
            count_col,
            dur_col
        ],
        "SpikeSummaryTable"
    )

    spike_df["SpikeRate_perHour"] = (
        spike_df[count_col]
        / spike_df[dur_col]
        * 3600
    )

    report_df = pd.read_csv(report_csv)

    require_cols(
        report_df,
        [
            "patient_id",
            "session_number",
            "acquired_on",
            "report_PATIENT_CLASS",
            "jay_in_or_out",
            "visit_type",
            "visit_dates_deid",
            "sz_freqs",
            "visit_hasSz",
            "epilepsy_type",
            "epilepsy_specific"
        ],
        "ReportTable"
    )

    # ========================================================
    # FILTER VISITS
    # ========================================================
    report_df = filter_visit_arrays_by_type(
        report_df,
        allowable_visits
    )

    (
        spike_df,
        report_df,
        n_patients_total
    ) = filter_outpatient_routine(
        spike_df,
        report_df,
        dur_col,
        MAX_ROUTINE_HOURS
    )

    # ========================================================
    # BUILD VISIT TABLE
    # ========================================================
    vuniq = build_visit_level_table(report_df)

    # ========================================================
    # BUILD PATIENT METRICS
    # ========================================================
    sz_metrics = build_patient_seizure_metrics(vuniq)

    # ========================================================
    # BUILD TYPING TABLE
    # ========================================================
    patient_typing = build_patient_typing(
        report_df,
        canonical3
    )

    # ========================================================
    # JOIN EVERYTHING
    # ========================================================
    merged = (
        spike_df.groupby("Patient")["SpikeRate_perHour"]
        .mean()
        .reset_index(name="MeanSpikeRate_perHour")
    )

    merged = merged.merge(
        sz_metrics,
        on="Patient",
        how="inner"
    )

    merged = merged.merge(
        patient_typing,
        on="Patient",
        how="left"
    )

    # ========================================================
    # FILTER EPILEPSY
    # ========================================================
    merged["EpilepsyType_norm"] = (
        merged["EpilepsyType"]
        .fillna("")
        .str.lower()
        .str.strip()
    )

    is_nesd = (
        merged["EpilepsyType_norm"]
        == NESD_LABEL.lower()
    )

    is_bad = merged["EpilepsyType_norm"].isin(bad_types)

    keep_mask = (
        ~is_nesd
        & ~is_bad
        & merged["MeanSzFreq"].notna()
    )

    cohort = merged[keep_mask].copy()

    print(
        f"[Cohort] {len(cohort)} epilepsy patients "
        f"with documented seizure frequency"
    )

    # ========================================================
    # SPEARMAN CORRELATION
    # ========================================================
    rho, p = spearmanr(
        cohort["MeanSpikeRate_perHour"],
        cohort["MeanSzFreq"],
        nan_policy="omit"
    )

    print("\n=== SPEARMAN CORRELATION ===")
    print(f"rho = {rho:.4f}")
    print(f"p   = {p:.4e}")

    # ========================================================
    # MIXED EFFECTS MODEL
    # ========================================================
    try:

        # pair_table = build_pair_table(
        #     vuniq,
        #     spike_df,
        #     report_df
        # )

        # model = smf.mixedlm(
        #     "SzFreq ~ LogSpikesPerHour + SignedLag_years",
        #     data=pair_table,
        #     groups=pair_table["Patient"]
        # )

        # result = model.fit()

        # print("\n=== MIXED EFFECTS MODEL ===")
        # print(result.summary())
        
        pair_table = build_pair_table(vuniq, spike_df, report_df)

        pair_table = pair_table.dropna(
            subset=["SzFreq", "LogSpikesPerHour", "SignedLag_years", "Patient"]
        ).reset_index(drop=True)

        pair_table["Patient"] = pair_table["Patient"].astype(str)

        model = smf.mixedlm(
            "SzFreq ~ LogSpikesPerHour + SignedLag_years",
            data=pair_table,
            groups=pair_table["Patient"]
        )

        result = model.fit()
        print(result.summary())

    except Exception as e:

        warnings.warn(
            f"Mixed effects model failed:\n{e}"
        )

    # ========================================================
    # FIGURE 1
    # ========================================================
    make_fig1(
        cohort,
        fig1_out,
        EPS_RATE
    )

    # ========================================================
    # FIGURE 2
    # ========================================================
    make_fig2(
        cohort,
        fig2_out
    )

    print("\nDone.")


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

def require_cols(df, cols, name):

    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if len(missing) > 0:

        raise ValueError(
            f"{name} missing columns: {missing}"
        )


# ============================================================
# FILTER VISIT ARRAYS
# ============================================================

def filter_visit_arrays_by_type(df, allowable_visits):

    df = df.copy()

    total_before = 0
    total_after = 0

    for idx in df.index:

        try:

            vt = parse_json_array(
                df.at[idx, "visit_type"]
            )

            dates = parse_json_array(
                df.at[idx, "visit_dates_deid"]
            )

            sz = parse_json_array(
                df.at[idx, "sz_freqs"]
            )

            has_sz = parse_json_array(
                df.at[idx, "visit_hasSz"]
            )

            if not (
                len(vt)
                == len(dates)
                == len(sz)
                == len(has_sz)
            ):
                continue

            total_before += len(vt)

            keep = [
                i
                for i, x in enumerate(vt)
                if x in allowable_visits
            ]

            vt2 = [vt[i] for i in keep]
            d2 = [dates[i] for i in keep]
            sz2 = [sz[i] for i in keep]
            hs2 = [has_sz[i] for i in keep]

            total_after += len(vt2)

            df.at[idx, "visit_type"] = json.dumps(vt2)
            df.at[idx, "visit_dates_deid"] = json.dumps(d2)
            df.at[idx, "sz_freqs"] = json.dumps(sz2)
            df.at[idx, "visit_hasSz"] = json.dumps(hs2)

        except Exception:
            continue

    print(
        f"[Visit filter] "
        f"{total_before} -> {total_after}"
    )

    return df


# ============================================================
# FILTER OUTPATIENT ROUTINE EEGS
# ============================================================

def filter_outpatient_routine(
    spike_df,
    report_df,
    dur_col,
    max_hours
):

    n_patients_total = (
        report_df["patient_id"]
        .nunique()
    )

    acq = (
        report_df["acquired_on"]
        .fillna("")
        .str.lower()
    )

    patient_class = (
        report_df["report_PATIENT_CLASS"]
        .fillna("")
        .str.lower()
    )

    jay = (
        report_df["jay_in_or_out"]
        .fillna("")
        .str.lower()
    )

    is_outpt = (
        acq.str.contains("spe")
        | acq.str.contains("radnor")
        | (patient_class == "outpatient")
        | (jay == "out")
    )

    outpt_keys = report_df.loc[
        is_outpt,
        ["patient_id", "session_number"]
    ].drop_duplicates()

    outpt_keys.columns = [
        "Patient",
        "Session"
    ]

    is_routine = (
        spike_df[dur_col]
        <= max_hours * 3600
    )

    routine_keys = spike_df.loc[
        is_routine,
        ["Patient", "Session"]
    ].drop_duplicates()

    merged_keys = outpt_keys.merge(
        routine_keys,
        on=["Patient", "Session"]
    )

    spike_df = spike_df.merge(
        merged_keys,
        on=["Patient", "Session"]
    )

    report_df = report_df.merge(
        merged_keys,
        left_on=[
            "patient_id",
            "session_number"
        ],
        right_on=[
            "Patient",
            "Session"
        ]
    )

    print(
        f"[Outpatient+routine] "
        f"{len(spike_df)} spike rows"
    )

    return (
        spike_df,
        report_df,
        n_patients_total
    )

# ============================================================
# BUILD VISIT TABLE
# ============================================================

def build_visit_level_table(report_df):

    rows = []

    for _, row in report_df.iterrows():

        pid = row["patient_id"]

        try:

            dates = parse_json_array(
                row["visit_dates_deid"]
            )

            sz = parse_json_array(
                row["sz_freqs"]
            )

            has_sz = parse_json_array(
                row["visit_hasSz"]
            )

        except Exception:
            continue

        if not (
            len(dates)
            == len(sz)
            == len(has_sz)
        ):
            continue

        for d, s, h in zip(
            dates,
            sz,
            has_sz
        ):

            try:

                d = pd.to_datetime(d)

            except Exception:
                continue

            try:

                s = float(s)

            except Exception:
                s = np.nan

            try:

                h = float(h)

            except Exception:
                h = np.nan

            rows.append({
                "Patient": pid,
                "VisitDate": d,
                "Freq": s,
                "HasSz": h
            })

    pv = pd.DataFrame(rows)

    vuniq = (
        pv.groupby(
            ["Patient", "VisitDate"],
            as_index=False
        )
        .agg({
            "Freq": "mean",
            "HasSz": "max"
        })
    )

    vuniq["Freq_R1"] = vuniq["Freq"]

    # Rule 1
    mask = (
        vuniq["Freq_R1"].isna()
        & (vuniq["HasSz"] == 0)
    )

    vuniq.loc[mask, "Freq_R1"] = 0

    return vuniq


# ============================================================
# PATIENT METRICS
# ============================================================

def build_patient_seizure_metrics(vuniq):

    out = (
        vuniq.groupby("Patient")
        .agg(
            MeanSzFreq=("Freq_R1", "mean"),
            FracVisits_HasSz1=(
                "HasSz",
                lambda x: np.mean(x == 1)
            )
        )
        .reset_index()
    )

    return out


# ============================================================
# PATIENT TYPING
# ============================================================

def build_patient_typing(
    report_df,
    canonical3
):

    tmp = report_df[
        [
            "patient_id",
            "epilepsy_type",
            "epilepsy_specific"
        ]
    ].drop_duplicates()

    tmp.columns = [
        "Patient",
        "EpilepsyType",
        "EpilepsySpecific"
    ]

    def determine_type(x):

        x = str(x).lower()

        if "temporal" in x:
            return "Temporal"

        if "frontal" in x:
            return "Frontal"

        return "General"

    tmp["EpiType3"] = (
        tmp["EpilepsySpecific"]
        .apply(determine_type)
    )

    return tmp


# ============================================================
# BUILD PAIR TABLE
# ============================================================

def build_pair_table(
    vuniq,
    spike_df,
    report_df
):

    eeg_dates = report_df[
        [
            "patient_id",
            "session_number",
            "start_time_deid"
        ]
    ].copy()

    eeg_dates.columns = [
        "Patient",
        "Session",
        "EEG_Date"
    ]

    eeg_dates["EEG_Date"] = pd.to_datetime(
        eeg_dates["EEG_Date"],
        errors="coerce"
    )

    eeg_tbl = eeg_dates.merge(
        spike_df[
            [
                "Patient",
                "Session",
                "SpikeRate_perHour"
            ]
        ],
        on=["Patient", "Session"]
    )

    rows = []

    for pid in np.intersect1d(
        eeg_tbl["Patient"].unique(),
        vuniq["Patient"].unique()
    ):

        eeg_rows = eeg_tbl[
            eeg_tbl["Patient"] == pid
        ]

        vis_rows = vuniq[
            vuniq["Patient"] == pid
        ]

        for _, eeg in eeg_rows.iterrows():

            for _, vis in vis_rows.iterrows():

                lag_days = (
                    vis["VisitDate"]
                    - eeg["EEG_Date"]
                ).days

                rows.append({
                    "Patient": pid,
                    "Session": eeg["Session"],
                    "SpikesPerHour":
                        eeg["SpikeRate_perHour"],
                    "SzFreq":
                        vis["Freq_R1"],
                    "HasSz":
                        vis["HasSz"],
                    "SignedLag_days":
                        lag_days
                })

    pair_table = pd.DataFrame(rows)

    pair_table["LogSpikesPerHour"] = np.log(
        pair_table["SpikesPerHour"]
        + 1e-3
    )

    pair_table["SignedLag_years"] = (
        pair_table["SignedLag_days"]
        / 365.25
    )

    return pair_table


# ============================================================
# FIGURE 1
# ============================================================

def make_fig1(
    cohort,
    out_path,
    eps_rate
):

    plt.figure(figsize=(8, 6))

    x_raw = cohort["MeanSpikeRate_perHour"].astype(float)
    y_raw = cohort["MeanSzFreq"].astype(float)

    x = np.log10(np.clip(x_raw + eps_rate, eps_rate, None))
    y = np.log10(np.clip(y_raw + eps_rate, eps_rate, None))

    # x = np.log10(
    #     cohort["MeanSpikeRate_perHour"]
    #     + eps_rate
    # )

    # y = np.log10(
    #     cohort["MeanSzFreq"]
    #     + eps_rate
    # )

    plt.scatter(x, y, alpha=0.6)

    plt.xlabel("Log Spike Rate")
    plt.ylabel("Log Seizure Frequency")

    plt.title(
        "Spike Rate vs Seizure Frequency"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved Fig1: {out_path}")


# ============================================================
# FIGURE 2
# ============================================================

def make_fig2(
    cohort,
    out_path
):

    plt.figure(figsize=(8, 6))

    groups = cohort.groupby("EpiType3")

    for name, grp in groups:

        plt.scatter(
            grp["MeanSpikeRate_perHour"],
            grp["MeanSzFreq"],
            label=name,
            alpha=0.7
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Spike Rate")
    plt.ylabel("Seizure Frequency")

    plt.legend()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved Fig2: {out_path}")


# ============================================================
# JSON ARRAY PARSER
# ============================================================

def parse_json_array(x):

    if pd.isna(x):
        return []

    x = str(x).strip()

    if x in ["", "[]", "[null]", "null"]:
        return []

    try:

        return json.loads(x)

    except Exception:

        return []


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_spike_sz_pipeline_clean()