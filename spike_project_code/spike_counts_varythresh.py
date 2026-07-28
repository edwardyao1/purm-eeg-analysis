from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd


# ===== Config =====
MAIN_DIR = Path("/mnt/sauce/littlab/users/erinconr/projects/routine_eeg/output/spikenet/")
OUT_CSV = Path("../data/SN_counts/spike_counts_summary_multiThresh.csv")

OVERWRITE = False  # True = rebuild from scratch

FS = 16  # Hz
WIN_SEC = 1.0
WIN_SAMP = round(WIN_SEC * FS)
HALF_WIN = WIN_SAMP // 2


# ===== Threshold list =====
base_thresh = np.arange(0.05, 1.00, 0.05)
extra_thresh = np.array([0.43, 0.46])

thresh_list = np.unique(np.concatenate([base_thresh, extra_thresh]))
n_thresh = len(thresh_list)

# Create column names like count_0_43
thresh_var_names = [
    f"count_{thr:.2f}".replace(".", "_")
    for thr in thresh_list
]


# ===== Find all probability CSVs =====
pattern1 = list(
    MAIN_DIR.glob(
        "sub-Penn*/ses-*/sub-Penn*_ses-*_task-EEG_eeg_*.csv"
    )
)

pattern2 = list(
    MAIN_DIR.glob(
        "sub-Penn*/ses-*/sub-Penn*_ses-*_combined_*.csv"
    )
)

files_all = pattern1 + pattern2

if len(files_all) == 0:
    print(f"No CSVs found under: {MAIN_DIR}")

# Remove duplicates while preserving order
seen = set()
files = []

for f in files_all:
    fstr = str(f.resolve())

    if fstr not in seen:
        seen.add(fstr)
        files.append(f)


# ===== Schema =====
base_columns = [
    "EEG_Name",
    "Patient",
    "Session",
    "Duration_sec"
]

all_columns = base_columns + thresh_var_names

summary = pd.DataFrame(columns=all_columns)

done_pairs = set()


# ===== Load existing summary =====
if not OVERWRITE and OUT_CSV.exists():

    try:
        existing = pd.read_csv(OUT_CSV)

        # Add missing columns
        for col in all_columns:
            if col not in existing.columns:

                if col == "EEG_Name":
                    existing[col] = ""
                else:
                    existing[col] = np.nan

        # Reorder columns
        existing = existing[all_columns]

        # Deduplicate by Patient + Session
        existing["pair_key"] = (
            existing["Patient"].astype(str)
            + "_"
            + existing["Session"].astype(str)
        )

        before = len(existing)

        existing = existing.drop_duplicates(
            subset="pair_key",
            keep="first"
        )

        after = len(existing)

        if after < before:
            print(
                f"Note: found {before - after} duplicate "
                f"(Patient, Session) rows; keeping first instances."
            )

        existing = existing.drop(columns=["pair_key"])

        summary = existing

        done_pairs = set(
            summary["Patient"].astype(str)
            + "_"
            + summary["Session"].astype(str)
        )

        print(
            f"Loaded existing summary "
            f"({len(summary)} rows)."
        )

    except Exception as e:

        warnings.warn(
            f"Could not read existing summary:\n{e}"
        )

else:

    if OVERWRITE:
        print("OVERWRITE=True: rebuilding summary from scratch.")
    else:
        print("No existing summary found. Starting a new one.")


# ===== Spike counting function =====
def count_spikes_wholefile(sn2, threshold, win_samp):
    """
    Count clustered detections over the whole file.

    Uses a 1-second skip rule after each detection.
    """

    total_count = 0
    n = len(sn2)

    i = 0

    while i < n:

        if sn2[i] > threshold:

            total_count += 1

            # Skip ahead ~1 second
            i += win_samp

        else:
            i += 1

    return total_count


# ===== Process each CSV =====
for k, fpath in enumerate(files):

    fname = fpath.name

    # ===== Parse patient/session =====
    match = re.search(
        r"sub-Penn(\d+).*?ses-(\d+)",
        str(fpath)
    )

    patient_num = np.nan
    session_num = np.nan

    if match:

        patient_num = int(match.group(1))
        session_num = int(match.group(2))

    else:

        match2 = re.search(
            r"sub-Penn(\d+)_ses-(\d+)",
            fname
        )

        if match2:

            patient_num = int(match2.group(1))
            session_num = int(match2.group(2))

    # Skip if parsing failed
    if np.isnan(patient_num) or np.isnan(session_num):

        warnings.warn(
            f"Skipping {fname} "
            f"(could not parse Patient/Session)."
        )

        continue

    pair_key = f"{patient_num}_{session_num}"

    # ===== Skip duplicates =====
    if not OVERWRITE and pair_key in done_pairs:

        print(
            f"Skipping (Patient={patient_num}, "
            f"Session={session_num}) already summarized. "
            f"({fname})"
        )

        continue

    # ===== Read CSV =====
    try:

        T = pd.read_csv(fpath)

    except Exception as e:

        warnings.warn(
            f"Failed to read {fname}. Skipping.\n{e}"
        )

        continue

    # ===== Check required column =====
    if "SN2" not in T.columns:

        warnings.warn(
            f"Skipping {fname}: required column SN2 not found."
        )

        continue

    # ===== Extract SN2 =====
    sn2 = pd.to_numeric(
        T["SN2"],
        errors="coerce"
    ).to_numpy()

    # Treat NaNs as below threshold
    sn2[~np.isfinite(sn2)] = -np.inf

    N = len(sn2)

    duration_sec = N / FS

    # ===== Count spikes for all thresholds =====
    counts = []

    for thr in thresh_list:

        count = count_spikes_wholefile(
            sn2,
            thr,
            WIN_SAMP
        )

        counts.append(count)

    # ===== Build row =====
    row = {
        "EEG_Name": fname,
        "Patient": patient_num,
        "Session": session_num,
        "Duration_sec": duration_sec
    }

    # Add threshold counts
    for col, val in zip(thresh_var_names, counts):
        row[col] = val

    # Append row
    summary = pd.concat(
        [summary, pd.DataFrame([row])],
        ignore_index=True
    )

    done_pairs.add(pair_key)

    # ===== Print status =====
    def get_count(target_thr):

        idx = np.where(
            np.isclose(thresh_list, target_thr)
        )[0]

        if len(idx) == 0:
            return np.nan

        return counts[idx[0]]

    print(
        f"Processed (Patient={patient_num}, "
        f"Session={session_num}) "
        f"{fname:<40} "
        f"dur={duration_sec:.1f}s  "
        f"count@0.10={get_count(0.10):.0f}  "
        f"count@0.50={get_count(0.50):.0f}  "
        f"count@0.90={get_count(0.90):.0f}"
    )


# ===== Sort for readability =====
if len(summary) > 0:

    try:

        summary = summary.sort_values(
            by=["Patient", "Session", "EEG_Name"]
        )

    except Exception:
        pass


# ===== Save summary =====
OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(OUT_CSV, index=False)

print(
    f"Saved summary to: {OUT_CSV} "
    f"(rows={len(summary)})"
)