# Pythonic Refactor of `run_spike_sz_pipeline_clean_April72026.m`

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    spearmanr,
    mannwhitneyu,
    kruskal,
)

# =========================================================
# Configuration
# =========================================================


@dataclass
class PipelineConfig:
    spike_summary_csv: Path
    report_csv: Path
    output_dir: Path

    max_routine_hours: float = 4

    nesd_label: str = "Non-Epileptic Seizure Disorder"

    bad_types: Tuple[str, ...] = (
        "uncertain if epilepsy",
        "unknown or mrn not found",
        "",
    )

    canonical_subtypes: Tuple[str, ...] = (
        "General",
        "Temporal",
        "Frontal",
    )

    eps_rate: float = 30e-3

    n_boot: int = 5000
    alpha: float = 0.05

    count_col: str = "count_0_46"
    duration_col: str = "Duration_sec"

    allowable_visits: Tuple[str, ...] = (
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
        "TELEHEALTH VIDEO VISIT RETURN",
    )


# =========================================================
# Utility Functions
# =========================================================


class ValidationError(Exception):
    pass


class TableValidator:
    @staticmethod
    def require_columns(df: pd.DataFrame, cols: List[str], name: str):
        missing = set(cols) - set(df.columns)

        if missing:
            raise ValidationError(
                f"{name} missing required columns: {sorted(missing)}"
            )

    @staticmethod
    def assert_unique_keys(
        df: pd.DataFrame,
        patient_col: str,
        session_col: str,
        name: str,
    ):
        duplicates = df.duplicated([patient_col, session_col])

        if duplicates.any():
            n_dup = duplicates.sum()
            raise ValidationError(
                f"{name} contains {n_dup} duplicated patient/session keys"
            )


# =========================================================
# JSON Parsing Utilities
# =========================================================


class JsonArrayParser:
    @staticmethod
    def parse_string_array(value: str) -> List[str]:
        if pd.isna(value) or value in ["", "[]"]:
            return []

        return list(json.loads(value))

    @staticmethod
    def parse_float_array(value: str) -> List[float]:
        if pd.isna(value) or value in ["", "[]"]:
            return []

        value = value.replace("null", "NaN")
        arr = json.loads(value)

        output = []

        for item in arr:
            try:
                x = float(item)
            except Exception:
                x = np.nan

            output.append(x)

        return output


# =========================================================
# Bootstrap Utilities
# =========================================================


class Bootstrap:
    @staticmethod
    def median_ci(
        values: np.ndarray,
        n_boot: int = 5000,
        alpha: float = 0.05,
    ):
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan, np.nan, np.nan

        rng = np.random.default_rng(1)

        boot = []

        for _ in range(n_boot):
            sample = rng.choice(values, size=len(values), replace=True)
            boot.append(np.median(sample))

        lower = np.percentile(boot, 100 * alpha / 2)
        upper = np.percentile(boot, 100 * (1 - alpha / 2))

        return np.median(values), lower, upper

    @staticmethod
    def spearman_ci(
        x: np.ndarray,
        y: np.ndarray,
        n_boot: int = 5000,
        alpha: float = 0.05,
    ):
        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        rng = np.random.default_rng(1)

        rho_boot = []

        for _ in range(n_boot):
            idx = rng.choice(len(x), size=len(x), replace=True)
            rho, _ = spearmanr(x[idx], y[idx])
            rho_boot.append(rho)

        rho, _ = spearmanr(x, y)

        lower = np.percentile(rho_boot, 100 * alpha / 2)
        upper = np.percentile(rho_boot, 100 * (1 - alpha / 2))

        return rho, lower, upper


# =========================================================
# Data Loading
# =========================================================


class DataLoader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_spike_summary(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.spike_summary_csv)

        TableValidator.require_columns(
            df,
            [
                "Patient",
                "Session",
                self.config.count_col,
                self.config.duration_col,
            ],
            "SpikeSummaryTable",
        )

        df["SpikeRate_perHour"] = (
            df[self.config.count_col]
            / df[self.config.duration_col]
            * 3600
        )

        return df

    def load_report_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.report_csv)

        required = [
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
            "epilepsy_specific",
            "nlp_gender",
            "deid_birth_date",
            "start_time_deid",
            "report_SPORADIC_EPILEPTIFORM_DISCHARGES",
            "jay_focal_epi",
            "jay_multifocal_epi",
            "jay_gen_epi",
        ]

        TableValidator.require_columns(
            df,
            required,
            "ReportTable",
        )

        return df


# =========================================================
# Visit Filtering
# =========================================================


class VisitFilter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def filter_visit_arrays_by_type(
        self,
        report_df: pd.DataFrame,
    ) -> pd.DataFrame:
        report_df = report_df.copy()

        allowable = set(self.config.allowable_visits)

        for idx, row in report_df.iterrows():
            visit_types = JsonArrayParser.parse_string_array(
                str(row["visit_type"])
            )

            visit_dates = JsonArrayParser.parse_string_array(
                str(row["visit_dates_deid"])
            )

            seizure_freqs = JsonArrayParser.parse_float_array(
                str(row["sz_freqs"])
            )

            visit_has_sz = JsonArrayParser.parse_float_array(
                str(row["visit_hasSz"])
            )

            if not (
                len(visit_types)
                == len(visit_dates)
                == len(seizure_freqs)
                == len(visit_has_sz)
            ):
                raise ValidationError(
                    f"Mismatched visit arrays for row {idx}"
                )

            keep_idx = [
                i
                for i, vt in enumerate(visit_types)
                if vt in allowable
            ]

            visit_types = [visit_types[i] for i in keep_idx]
            visit_dates = [visit_dates[i] for i in keep_idx]
            seizure_freqs = [seizure_freqs[i] for i in keep_idx]
            visit_has_sz = [visit_has_sz[i] for i in keep_idx]

            report_df.at[idx, "visit_type"] = json.dumps(visit_types)
            report_df.at[idx, "visit_dates_deid"] = json.dumps(visit_dates)
            report_df.at[idx, "sz_freqs"] = json.dumps(seizure_freqs)
            report_df.at[idx, "visit_hasSz"] = json.dumps(visit_has_sz)

        return report_df


# =========================================================
# Outpatient + Routine EEG Filtering
# =========================================================


class EEGFilter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def filter_outpatient_routine(
        self,
        spike_df: pd.DataFrame,
        report_df: pd.DataFrame,
    ):
        acquired = report_df["acquired_on"].str.lower().fillna("")

        outpatient_site = (
            acquired.str.contains("spe")
            | acquired.str.contains("radnor")
        )

        outpatient_class = (
            report_df["report_PATIENT_CLASS"]
            .str.lower()
            .fillna("")
            == "outpatient"
        )

        outpatient_jay = (
            report_df["jay_in_or_out"]
            .str.lower()
            .fillna("")
            == "out"
        )

        outpatient_mask = (
            outpatient_site
            | outpatient_class
            | outpatient_jay
        )

        outpatient_keys = report_df.loc[
            outpatient_mask,
            ["patient_id", "session_number"],
        ].drop_duplicates()

        outpatient_keys.columns = ["Patient", "Session"]

        routine_mask = (
            spike_df[self.config.duration_col]
            <= self.config.max_routine_hours * 3600
        )

        routine_keys = spike_df.loc[
            routine_mask,
            ["Patient", "Session"],
        ].drop_duplicates()

        valid_keys = outpatient_keys.merge(
            routine_keys,
            on=["Patient", "Session"],
        )

        filtered_spikes = spike_df.merge(
            valid_keys,
            on=["Patient", "Session"],
        )

        filtered_reports = report_df.merge(
            valid_keys,
            left_on=["patient_id", "session_number"],
            right_on=["Patient", "Session"],
        )

        return filtered_spikes, filtered_reports


# =========================================================
# Visit-Level Table Construction
# =========================================================


class VisitLevelBuilder:
    @staticmethod
    def build(report_df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in report_df.iterrows():
            patient = row["patient_id"]

            visit_dates = JsonArrayParser.parse_string_array(
                str(row["visit_dates_deid"])
            )

            seizure_freqs = JsonArrayParser.parse_float_array(
                str(row["sz_freqs"])
            )

            has_sz = JsonArrayParser.parse_float_array(
                str(row["visit_hasSz"])
            )

            for d, f, h in zip(
                visit_dates,
                seizure_freqs,
                has_sz,
            ):
                rows.append(
                    {
                        "Patient": patient,
                        "VisitDate": pd.to_datetime(d),
                        "Freq": f,
                        "HasSz": h,
                    }
                )

        visit_df = pd.DataFrame(rows)

        grouped = (
            visit_df.groupby(["Patient", "VisitDate"])
            .agg(
                {
                    "Freq": "mean",
                    "HasSz": "max",
                }
            )
            .reset_index()
        )

        grouped["Freq_R1"] = grouped["Freq"]

        mask = (
            grouped["Freq_R1"].isna()
            & (grouped["HasSz"] == 0)
        )

        grouped.loc[mask, "Freq_R1"] = 0

        return grouped


# =========================================================
# Patient-Level Seizure Metrics
# =========================================================


class SeizureMetricsBuilder:
    @staticmethod
    def build(visit_df: pd.DataFrame):
        output = (
            visit_df.groupby("Patient")
            .agg(
                MeanSzFreq=("Freq_R1", "mean"),
                FracVisits_HasSz1=(
                    "HasSz",
                    lambda x: np.mean(x == 1),
                ),
            )
            .reset_index()
        )

        return output


# =========================================================
# Epilepsy Typing
# =========================================================


class EpilepsyTypingBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build(self, report_df: pd.DataFrame):
        df = (
            report_df[
                [
                    "patient_id",
                    "epilepsy_type",
                    "epilepsy_specific",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

        df.columns = [
            "Patient",
            "EpilepsyType",
            "EpilepsySpecific",
        ]

        df["EpiType3"] = ""

        spec = df["EpilepsySpecific"].str.lower().fillna("")
        etype = df["EpilepsyType"].str.lower().fillna("")

        df.loc[
            spec.str.contains("temporal"),
            "EpiType3",
        ] = "Temporal"

        df.loc[
            spec.str.contains("frontal"),
            "EpiType3",
        ] = "Frontal"

        df.loc[
            (df["EpiType3"] == "")
            & (etype == "general"),
            "EpiType3",
        ] = "General"

        return df


# =========================================================
# Analysis Views
# =========================================================


class CohortBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build(
        self,
        spike_df: pd.DataFrame,
        report_df: pd.DataFrame,
        typing_df: pd.DataFrame,
        seizure_metrics_df: pd.DataFrame,
    ):
        patient_spike_rates = (
            spike_df.groupby("Patient")
            .agg(
                MeanSpikeRate_perHour=(
                    "SpikeRate_perHour",
                    "mean",
                )
            )
            .reset_index()
        )

        patient_spike_rates = patient_spike_rates.merge(
            typing_df,
            on="Patient",
            how="left",
        )

        epilepsy_type = (
            patient_spike_rates["EpilepsyType"]
            .str.lower()
            .fillna("")
        )

        is_nesd = epilepsy_type == self.config.nesd_label.lower()

        is_bad = epilepsy_type.isin(self.config.bad_types)

        is_epilepsy = ~(is_nesd | is_bad)

        patient_spike_sz = patient_spike_rates.merge(
            seizure_metrics_df,
            on="Patient",
        )

        patient_spike_sz = patient_spike_sz.loc[
            is_epilepsy.values
        ]

        patient_spike_sz = patient_spike_sz.loc[
            np.isfinite(patient_spike_sz["MeanSzFreq"])
        ]

        return {
            "patient_spike_rates": patient_spike_rates,
            "patient_spike_sz": patient_spike_sz,
            "typing": typing_df,
        }


# =========================================================
# Figure Generation
# =========================================================


class FigureBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def make_spearman_plot(
        self,
        patient_spike_sz: pd.DataFrame,
        output_path: Path,
    ):
        x = patient_spike_sz["MeanSpikeRate_perHour"].to_numpy()
        y = patient_spike_sz["MeanSzFreq"].to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        rho, p = spearmanr(x, y)

        rho_boot, rho_lo, rho_hi = Bootstrap.spearman_ci(
            x,
            y,
            n_boot=self.config.n_boot,
            alpha=self.config.alpha,
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(x, y, alpha=0.35)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Spikes/hour")
        ax.set_ylabel("Seizures/month")

        ax.set_title(
            (
                f"Spearman rho={rho:.2f}, "
                f"95% CI [{rho_lo:.2f}, {rho_hi:.2f}], "
                f"p={p:.3g}"
            )
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)

        plt.close(fig)


# =========================================================
# Summary Writer
# =========================================================


class ResultsWriter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def write_summary(
        self,
        cohort_views: Dict,
    ):
        output_path = self.config.output_dir / "results_summary.txt"

        patient_spike_sz = cohort_views["patient_spike_sz"]

        median_spike_rate = np.median(
            patient_spike_sz["MeanSpikeRate_perHour"]
        )

        median_seizure_freq = np.median(
            patient_spike_sz["MeanSzFreq"]
        )

        with open(output_path, "w") as f:
            f.write("Spike-Seizure Analysis Summary\n")
            f.write("================================\n\n")

            f.write(
                f"Patients: {len(patient_spike_sz)}\n"
            )

            f.write(
                f"Median spike rate: {median_spike_rate:.2f}\n"
            )

            f.write(
                f"Median seizure frequency: "
                f"{median_seizure_freq:.2f}\n"
            )


# =========================================================
# Main Pipeline
# =========================================================


class SpikeSeizurePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        self.loader = DataLoader(config)
        self.visit_filter = VisitFilter(config)
        self.eeg_filter = EEGFilter(config)
        self.typing_builder = EpilepsyTypingBuilder(config)
        self.cohort_builder = CohortBuilder(config)
        self.figure_builder = FigureBuilder(config)
        self.results_writer = ResultsWriter(config)

    def run(self):
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Loading data...")

        spike_df = self.loader.load_spike_summary()
        report_df = self.loader.load_report_table()

        print("Filtering outpatient clinic visits...")

        report_df = self.visit_filter.filter_visit_arrays_by_type(
            report_df
        )

        print("Filtering outpatient routine EEGs...")

        spike_df, report_df = (
            self.eeg_filter.filter_outpatient_routine(
                spike_df,
                report_df,
            )
        )

        print("Validating patient/session uniqueness...")

        TableValidator.assert_unique_keys(
            spike_df,
            "Patient",
            "Session",
            "SpikeSummaryTable",
        )

        TableValidator.assert_unique_keys(
            report_df,
            "patient_id",
            "session_number",
            "ReportTable",
        )

        print("Building visit-level table...")

        visit_df = VisitLevelBuilder.build(report_df)

        print("Building seizure metrics...")

        seizure_metrics_df = (
            SeizureMetricsBuilder.build(visit_df)
        )

        print("Building epilepsy typing table...")

        typing_df = self.typing_builder.build(report_df)

        print("Building cohort views...")

        cohort_views = self.cohort_builder.build(
            spike_df,
            report_df,
            typing_df,
            seizure_metrics_df,
        )

        print("Generating figures...")

        self.figure_builder.make_spearman_plot(
            cohort_views["patient_spike_sz"],
            self.config.output_dir / "fig2_python.png",
        )

        print("Writing summary...")

        self.results_writer.write_summary(cohort_views)

        print("Pipeline complete.")


# =========================================================
# Entrypoint
# =========================================================


if __name__ == "__main__":
    config = PipelineConfig(
        spike_summary_csv=Path("../data/spike_counts.csv"),
        report_csv=Path(
            "../data/Routineeegpec-Deidreport_DATA_LABELS_2026-03-11_0911.csv"
        ),
        output_dir=Path("../output"),
    )

    pipeline = SpikeSeizurePipeline(config)
    pipeline.run()
