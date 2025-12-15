#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fit and evaluate a Hopf (neurolib `PhenoHopfModel`) on a single subject from a MATLAB `.mat`.

Reuses this repo's existing preprocessing/metrics:
- Bandpass filtering: `BOLDFilters.BandPassFilter` (configured by CLI TR/flp/fhi)
- FC: `my_functions.fc` + similarity `my_functions.matrix_correlation`
- phFCD: `phFCD.phFCD` + similarity `my_functions.matrix_kolmogorov` (KS distance)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.io

from mat_timeseries import extract_subject_timeseries, infer_n_nodes, load_mat


@dataclass(frozen=True)
class SimSampling:
    dt_ms: float = 0.1
    sampling_dt_ms: float = 10.0


def _load_sc_matrix(sc_path: Optional[Path]) -> np.ndarray:
    if sc_path is None:
        # Try petTOAD's AAL SC loader (requires repo data layout).
        try:
            from petTOAD_load import load_norm_aal_sc

            return np.asarray(load_norm_aal_sc(), dtype=float)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not load SC automatically. Provide --sc (CSV/NPY/MAT)."
            ) from e

    if sc_path.suffix.lower() == ".npy":
        return np.load(sc_path)

    if sc_path.suffix.lower() in {".csv", ".tsv"}:
        delim = "," if sc_path.suffix.lower() == ".csv" else "\t"
        return np.genfromtxt(sc_path, delimiter=delim)

    if sc_path.suffix.lower() == ".mat":
        m = scipy.io.loadmat(sc_path, squeeze_me=True, struct_as_record=False)
        # Try common names
        for key in ("timeseries_all", "sc", "SC", "Cmat", "C"):
            if key in m:
                return np.asarray(m[key], dtype=float)
        raise KeyError(f"No SC variable found in {sc_path}. Tried sc/SC/Cmat/C")

    raise ValueError(f"Unsupported --sc file type: {sc_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default="data/ts_young_TR0.72.mat", help="Path to MATLAB .mat containing timeseries_all")
    parser.add_argument("--timeseries-var", default="timeseries_all", help="Variable name for per-subject timeseries")
    parser.add_argument("--fc-mean-var", default="FC_mean", help="Variable name for mean FC (optional)")
    parser.add_argument("--fc-all-var", default="FC_all", help="Variable name for per-subject FC (optional)")
    parser.add_argument("--subject-index", type=int, default=0, help="0-based subject index")

    parser.add_argument("--tr", type=float, default=0.72, help="Empirical TR in seconds")
    parser.add_argument("--flp", type=float, default=0.04, help="Bandpass low frequency (Hz)")
    parser.add_argument("--fhi", type=float, default=0.07, help="Bandpass high frequency (Hz)")

    parser.add_argument("--warmup-sec", type=float, default=120.0, help="Warm-up seconds to discard")

    parser.add_argument("--kgl-min", type=float, default=0.0)
    parser.add_argument("--kgl-max", type=float, default=3.5)
    parser.add_argument("--kgl-step", type=float, default=0.02)

    parser.add_argument("--a", type=float, default=-0.02, help="Hopf bifurcation parameter a (fixed)")
    parser.add_argument("--sigma", type=float, default=0.02, help="Noise strength")

    parser.add_argument("--out", type=Path, default=Path("hopf_fit_out"), help="Output directory")
    parser.add_argument("--save-bold", action="store_true", help="Also save simulated BOLD per run (large)")

    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    mat = load_mat(str(args.mat))
    n_nodes_hint = infer_n_nodes(mat, args.fc_mean_var, args.fc_all_var)
    ts = extract_subject_timeseries(
        mat,
        args.timeseries_var,
        args.subject_index,
        n_nodes=n_nodes_hint,
    )
    n_nodes = int(ts.shape[0])

    # Local imports (so `py_compile` works even if neurolib isn't installed).
    import BOLDFilters
    import filteredPowerSpectralDensity as filtPowSpectr
    import my_functions as my_func
    from phFCD import phFCD as calc_phfcd

    # Configure filter globals.
    BOLDFilters.TR = float(args.tr)
    BOLDFilters.flp = float(args.flp)
    BOLDFilters.fhi = float(args.fhi)

    ts_filt = BOLDFilters.BandPassFilter(ts)

    # Empirical metrics (same code-path as the rest of the repo)
    emp_fc = my_func.fc(ts_filt)
    emp_phfcd = calc_phfcd(ts_filt)

    # Frequency per node for Hopf
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(ts_filt, float(args.tr))
    f_diff = np.asarray(f_diff, dtype=float)
    f_diff[f_diff == 0] = np.mean(f_diff[f_diff != 0]) if np.any(f_diff != 0) else 0.05

    # Load SC
    sc = _load_sc_matrix(args.mat)
    if sc.shape[0] != n_nodes or sc.shape[1] != n_nodes:
        raise ValueError(f"SC shape {sc.shape} does not match N={n_nodes}.")

    # Neurolib imports
    from neurolib.models.pheno_hopf import PhenoHopfModel
    from neurolib.optimize.exploration import BoxSearch
    from neurolib.utils import paths
    from neurolib.utils import pypetUtils as pu
    from neurolib.utils.parameterSpace import ParameterSpace

    sampling = SimSampling()

    # Sim duration: warmup + empirical length
    t_emp = ts_filt.shape[1] * float(args.tr)
    duration_ms = (float(args.warmup_sec) + t_emp) * 1000.0

    # Prepare Hopf model
    Dmat = np.zeros_like(sc)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    model.params["Dmat"] = None
    model.params["duration"] = duration_ms
    model.params["signalV"] = 0
    model.params["dt"] = sampling.dt_ms
    model.params["sampling_dt"] = sampling.sampling_dt_ms
    model.params["sigma"] = float(args.sigma)
    model.params["w"] = 2 * np.pi * f_diff
    model.params["a"] = np.ones(n_nodes) * float(args.a)

    k_values = np.round(
        np.arange(float(args.kgl_min), float(args.kgl_max) + float(args.kgl_step) / 2.0, float(args.kgl_step)),
        6,
    )
    parameters = ParameterSpace({"K_gl": [float(v) for v in k_values]}, kind="grid")

    filename = f"subj-{args.subject_index}_hopf_fit_Kgl.hdf"
    paths.HDF_DIR = str(args.out)

    # Use a module-level global like the existing scripts (neurolib calls evaluate(traj)).
    global search  # noqa: PLW0603

    def evaluate(traj):  # noqa: ANN001
        m = search.getModelFromTraj(traj)
        m.randomICs()
        m.run(chunkwise=True, chunksize=60000, append=True)

        warmup_idx = int(round(float(args.warmup_sec) * 1000.0 / sampling.sampling_dt_ms))
        step = int(round(float(args.tr) * 1000.0 / sampling.sampling_dt_ms))
        if step <= 0:
            raise ValueError("Invalid TR/sampling_dt leading to non-positive step.")

        sim_ts = m.outputs.x[:, warmup_idx::step]
        sim_ts_filt = BOLDFilters.BandPassFilter(sim_ts)

        # Metrics
        sim_fc = my_func.fc(sim_ts_filt)
        fc_pearson = float(my_func.matrix_correlation(sim_fc, emp_fc))

        sim_phfcd = calc_phfcd(sim_ts_filt)
        phfcd_ks = float(my_func.matrix_kolmogorov(emp_phfcd, sim_phfcd))

        result_dict: Dict[str, Any] = {
            "fc_pearson": float(fc_pearson),
            "phfcd_ks": float(phfcd_ks),
        }
        if args.save_bold:
            result_dict["BOLD"] = np.asarray(sim_ts_filt)

        search.saveToPypet(result_dict, traj)

    # Run search (single repetition)
    search = BoxSearch(model=model, evalFunction=evaluate, parameterSpace=parameters, filename=filename)
    search.run(chunkwise=True, chunksize=60000, append=True)

    # Collect results
    hdf_path = args.out / filename
    traj_names = pu.getTrajectorynamesInFile(str(hdf_path))
    if len(traj_names) == 0:
        raise RuntimeError("No trajectories found in output HDF.")

    # For BoxSearch, there is typically one trajectory with multiple runs.
    tr = pu.loadPypetTrajectory(str(hdf_path), traj_names[0])
    run_names = tr.f_get_run_names()

    rows = []
    for run_i in range(len(run_names)):
        r = pu.getRun(run_i, tr)
        # r contains the saved results and explored parameters
        try:
            kgl = float(r["K_gl"])
        except Exception:  # noqa: BLE001
            kgl = float("nan")
        try:
            fc_p = float(r["fc_pearson"])
        except Exception:  # noqa: BLE001
            fc_p = float("nan")
        try:
            ks = float(r["phfcd_ks"])
        except Exception:  # noqa: BLE001
            ks = float("nan")
        rows.append({"K_gl": kgl, "fc_pearson": fc_p, "phfcd_ks": ks})

    df = pd.DataFrame(rows).sort_values("K_gl").reset_index(drop=True)
    out_csv = args.out / f"subj-{args.subject_index}_hopf_fit_results.csv"
    df.to_csv(out_csv, index=False)

    best_idx = int(df["phfcd_ks"].astype(float).argmin())
    best = df.iloc[best_idx].to_dict()

    summary = {
        "mat": str(args.mat),
        "subject_index": int(args.subject_index),
        "tr": float(args.tr),
        "n_nodes": int(n_nodes),
        "t_points": int(ts_filt.shape[1]),
        "metrics": {
            "best_by": "min_phfcd_ks",
            "best": best,
        },
        "outputs": {
            "results_csv": str(out_csv),
            "pypet_hdf": str(hdf_path),
        },
    }

    (args.out / f"subj-{args.subject_index}_hopf_fit_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"Wrote: {out_csv}")
    print(f"Best (min phfcd_ks): K_gl={best.get('K_gl')} phfcd_ks={best.get('phfcd_ks')} fc_pearson={best.get('fc_pearson')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
