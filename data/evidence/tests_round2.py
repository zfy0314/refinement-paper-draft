import numpy as np
import pandas as pd
import traceback
from scipy import stats
import datetime

OUTPUT_FILE = "test_results.txt"
LOOKAHEAD = 6  # model default as given in RacecarPolicy

# small tolerances for numeric comparisons
RTOL = 1e-5
ATOL = 1e-6

def _log(msg: str):
    """Append a message to the result file with timestamp"""
    ts = datetime.datetime.now().isoformat()
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

def _safe_cols_exist(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

def _first_examples(df, idxs, cols, n=5):
    rows = df.loc[idxs, cols].head(n)
    return rows.to_string(index=True)

def test_basic_info(df):
    """Summarize dataset: rows, episodes, rows per episode"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_basic_info ===\n")
            f.write(f"Dataset rows: {len(df)}\n")
            if "episodeID" in df.columns:
                unique_eps = df["episodeID"].nunique()
                f.write(f"Unique episodeID: {unique_eps}\n")
                counts = df["episodeID"].value_counts()
                f.write(f"Rows per episode (min, median, max): {counts.min()}, {counts.median()}, {counts.max()}\n")
            else:
                f.write("No episodeID column found.\n")
    except Exception as e:
        _log("ERROR in test_basic_info: " + str(e))
        _log(traceback.format_exc())

def test_latent_consistency(df):
    """Verify latents are computed from observed obs_* columns"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_latent_consistency ===\n")
            # required obs columns
            obs_theta_cols = [f"obs_tile_{i}_theta" for i in range(LOOKAHEAD)]
            obs_border_cols = [f"obs_tile_{i}_border" for i in range(LOOKAHEAD)]
            obs_tile0_x = "obs_tile_0_x"
            wheel_cols = ["obs_abs_front_left", "obs_abs_front_right", "obs_abs_rear_left", "obs_abs_rear_right"]
            req_cols = obs_theta_cols + obs_border_cols + [obs_tile0_x, "obs_current_speed", "obs_current_angle", "obs_current_omega"] + wheel_cols
            _safe_cols_exist(df, req_cols)

            # compute derived
            look_theta = df[obs_theta_cols].astype(float)
            look_theta_mean = look_theta.mean(axis=1).to_numpy()
            look_theta_abs_mean = look_theta.abs().mean(axis=1).to_numpy()
            sharp_vals = df[obs_border_cols].max(axis=1).astype(float).to_numpy()
            closest_x = df[obs_tile0_x].to_numpy().astype(float)
            wheel_abs_mean = df[wheel_cols].astype(float).mean(axis=1).to_numpy()
            speed = df["obs_current_speed"].to_numpy().astype(float)
            heading_indicator = df["obs_current_angle"].to_numpy().astype(float)
            omega = df["obs_current_omega"].to_numpy().astype(float)

            # collected latents to compare
            latent_checks = {
                "latent_lookahead_theta_mean": look_theta_mean,
                "latent_lookahead_theta_abs_mean": look_theta_abs_mean,
                "latent_sharp_ahead": sharp_vals,
                "latent_closest_x": closest_x,
                "latent_wheel_abs_mean": wheel_abs_mean,
                "latent_speed": speed,
                "latent_heading_indicator": heading_indicator,
                "latent_omega": omega,
            }
            # wheel slip derived
            wheel_slip_rel = wheel_abs_mean - speed
            wheel_slip_pos = np.maximum(wheel_slip_rel, 0.0)
            latent_checks["latent_wheel_slip_rel"] = wheel_slip_rel
            latent_checks["latent_wheel_slip_pos"] = wheel_slip_pos

            missing_latents = [k for k in latent_checks.keys() if k not in df.columns]
            if missing_latents:
                f.write(f"Missing latent columns in dataframe: {missing_latents}\n")
            # compare each latent
            for name, arr in latent_checks.items():
                if name not in df.columns:
                    continue
                logged = df[name].to_numpy().astype(float)
                is_close = np.isclose(logged, arr, rtol=RTOL, atol=ATOL)
                n_bad = (~is_close).sum()
                pct_bad = 100.0 * n_bad / len(df)
                f.write(f"{name}: mismatches {n_bad}/{len(df)} ({pct_bad:.3f}%)\n")
                if n_bad > 0:
                    bad_idx = np.where(~is_close)[0][:5]
                    cols_show = ["Unnamed: 0", "episodeID"] if "Unnamed: 0" in df.columns else []
                    cols_show += [name]
                    sample = df.iloc[bad_idx][cols_show].to_string(index=True)
                    f.write("Sample mismatches (index, logged value):\n")
                    f.write(sample + "\n")
    except Exception as e:
        _log("ERROR in test_latent_consistency: " + str(e))
        _log(traceback.format_exc())

def test_steering_computation(df):
    """Reconstruct steering_raw, steering_gain_effective and steer_clamped and compare"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_steering_computation ===\n")
            required = [
                "weight_steer_closest_x", "weight_steer_lookahead_theta", "weight_steer_heading_indicator",
                "weight_steer_omega", "weight_steer_omega_speed_scale", "weight_steering_gain", "weight_steer_speed_scale",
                "latent_closest_x", "latent_lookahead_theta_mean", "latent_heading_indicator", "latent_omega", "latent_speed",
                "latent_steering_raw", "latent_steering_gain_effective", "latent_steer_clamped", "latent_steer_abs",
                "output_clipped_steer"
            ]
            _safe_cols_exist(df, required)

            # gather arrays
            w_cx = df["weight_steer_closest_x"].astype(float).to_numpy()
            w_ltheta = df["weight_steer_lookahead_theta"].astype(float).to_numpy()
            w_heading = df["weight_steer_heading_indicator"].astype(float).to_numpy()
            w_omega = df["weight_steer_omega"].astype(float).to_numpy()
            w_omega_speed = df["weight_steer_omega_speed_scale"].astype(float).to_numpy()
            w_gain = df["weight_steering_gain"].astype(float).to_numpy()
            w_speed_scale = df["weight_steer_speed_scale"].astype(float).to_numpy()

            closest_x = df["latent_closest_x"].astype(float).to_numpy()
            lookahead_theta_mean = df["latent_lookahead_theta_mean"].astype(float).to_numpy()
            heading_indicator = df["latent_heading_indicator"].astype(float).to_numpy()
            omega = df["latent_omega"].astype(float).to_numpy()
            speed = df["latent_speed"].astype(float).to_numpy()

            # reconstruct
            centering = w_cx * closest_x
            theta_contrib = w_ltheta * lookahead_theta_mean
            heading_contrib = w_heading * heading_indicator
            omega_damping = -np.abs(w_omega) * np.abs(omega) * (1.0 + np.abs(w_omega_speed) * speed)
            steering_raw_calc = centering + theta_contrib + heading_contrib + omega_damping

            steering_raw_logged = df["latent_steering_raw"].astype(float).to_numpy()
            eq_raw = np.isclose(steering_raw_calc, steering_raw_logged, rtol=RTOL, atol=ATOL)
            f.write(f"steering_raw mismatches: { (~eq_raw).sum() } / {len(df)}\n")
            if (~eq_raw).any():
                idx = np.where(~eq_raw)[0][:5]
                f.write("Examples of steering_raw mismatches (index, calc, logged):\n")
                for i in idx:
                    f.write(f"{i}: {steering_raw_calc[i]:.6f} vs {steering_raw_logged[i]:.6f}\n")

            # gain effective
            steering_gain_eff_calc = w_gain / (1.0 + np.abs(w_speed_scale) * speed)
            steering_gain_eff_logged = df["latent_steering_gain_effective"].astype(float).to_numpy()
            eq_gain = np.isclose(steering_gain_eff_calc, steering_gain_eff_logged, rtol=RTOL, atol=ATOL)
            f.write(f"steering_gain_effective mismatches: { (~eq_gain).sum() } / {len(df)}\n")
            if (~eq_gain).any():
                idx = np.where(~eq_gain)[0][:5]
                for i in idx:
                    f.write(f"{i}: {steering_gain_eff_calc[i]:.6f} vs {steering_gain_eff_logged[i]:.6f}\n")

            # steer clamped
            steer_calc = steering_raw_calc * steering_gain_eff_calc
            steer_clamped_calc = np.clip(steer_calc, -1.0, 1.0)
            steer_logged = df["latent_steer_clamped"].astype(float).to_numpy()
            eq_steer = np.isclose(steer_clamped_calc, steer_logged, rtol=RTOL, atol=ATOL)
            f.write(f"steer_clamped mismatches: { (~eq_steer).sum() } / {len(df)}\n")
            if (~eq_steer).any():
                idx = np.where(~eq_steer)[0][:5]
                for i in idx:
                    f.write(f"{i}: calc {steer_clamped_calc[i]:.6f} logged {steer_logged[i]:.6f}\n")

            # compare to actual action output
            if "output_clipped_steer" in df.columns:
                out_steer = df["output_clipped_steer"].astype(float).to_numpy()
                eq_out = np.isclose(out_steer, steer_clamped_calc, rtol=RTOL, atol=ATOL)
                f.write(f"output_clipped_steer matches steer_clamped calc: {eq_out.sum()} / {len(df)}\n")
                if (~eq_out).any():
                    idx = np.where(~eq_out)[0][:5]
                    for i in idx:
                        f.write(f"{i}: out {out_steer[i]:.6f} calc {steer_clamped_calc[i]:.6f}\n")
    except Exception as e:
        _log("ERROR in test_steering_computation: " + str(e))
        _log(traceback.format_exc())

def test_desired_speed_and_actuators(df):
    """Recompute desired_speed, accel, brake and compare to latents and outputs"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_desired_speed_and_actuators ===\n")
            required = [
                "weight_speed_base", "weight_speed_curv", "weight_speed_lat", "weight_speed_slip", "weight_speed_omega",
                "weight_speed_lookahead_theta_abs", "weight_speed_lookahead_theta_abs_speed", "weight_speed_steer_interaction",
                "weight_accel_gain", "weight_brake_gain",
                "latent_sharp_ahead", "latent_lookahead_theta_abs_mean", "latent_closest_x", "latent_wheel_slip_pos",
                "latent_omega", "latent_speed", "latent_steer_abs",
                "latent_desired_speed_preclamp", "latent_desired_speed", "latent_speed_error",
                "latent_accel_raw", "latent_brake_raw",
                "output_clipped_accelerate", "output_clipped_brake"
            ]
            _safe_cols_exist(df, required)

            # arrays
            w_base = df["weight_speed_base"].astype(float).to_numpy()
            w_curv = df["weight_speed_curv"].astype(float).to_numpy()
            w_lat = df["weight_speed_lat"].astype(float).to_numpy()
            w_slip = df["weight_speed_slip"].astype(float).to_numpy()
            w_omega = df["weight_speed_omega"].astype(float).to_numpy()
            w_lookabs = df["weight_speed_lookahead_theta_abs"].astype(float).to_numpy()
            w_lookabs_speed = df["weight_speed_lookahead_theta_abs_speed"].astype(float).to_numpy()
            w_steer_inter = df["weight_speed_steer_interaction"].astype(float).to_numpy()

            w_accel_gain = df["weight_accel_gain"].astype(float).to_numpy()
            w_brake_gain = df["weight_brake_gain"].astype(float).to_numpy()

            sharp = df["latent_sharp_ahead"].astype(float).to_numpy()
            lookabs = df["latent_lookahead_theta_abs_mean"].astype(float).to_numpy()
            clos_x = df["latent_closest_x"].astype(float).to_numpy()
            wheel_slip_pos = df["latent_wheel_slip_pos"].astype(float).to_numpy()
            omega = df["latent_omega"].astype(float).to_numpy()
            speed = df["latent_speed"].astype(float).to_numpy()
            steer_abs = df["latent_steer_abs"].astype(float).to_numpy()

            desired_precalc = (
                w_base
                + w_curv * sharp
                + w_lat * np.abs(clos_x)
                + w_slip * wheel_slip_pos
                + w_omega * np.abs(omega)
                + w_lookabs * lookabs
                + w_lookabs_speed * lookabs * speed
                + w_steer_inter * steer_abs * speed
            )
            logged_pre = df["latent_desired_speed_preclamp"].astype(float).to_numpy()
            eq_pre = np.isclose(desired_precalc, logged_pre, rtol=RTOL, atol=ATOL)
            f.write(f"desired_preclamp mismatches: { (~eq_pre).sum() } / {len(df)}\n")
            if (~eq_pre).any():
                idx = np.where(~eq_pre)[0][:5]
                for i in idx:
                    f.write(f"{i}: calc {desired_precalc[i]:.6f} logged {logged_pre[i]:.6f}\n")

            desired_clamped = np.clip(desired_precalc, 0.0, 1.0)
            logged_desired = df["latent_desired_speed"].astype(float).to_numpy()
            eq_des = np.isclose(desired_clamped, logged_desired, rtol=RTOL, atol=ATOL)
            f.write(f"desired_speed (post clamp) mismatches: { (~eq_des).sum() } / {len(df)}\n")

            speed_error_calc = desired_clamped - speed
            logged_error = df["latent_speed_error"].astype(float).to_numpy()
            eq_err = np.isclose(speed_error_calc, logged_error, rtol=RTOL, atol=ATOL)
            f.write(f"speed_error mismatches: { (~eq_err).sum() } / {len(df)}\n")

            accel_raw = np.maximum(speed_error_calc * w_accel_gain, 0.0)
            logged_accel_raw = df["latent_accel_raw"].astype(float).to_numpy()
            eq_acc_raw = np.isclose(accel_raw, logged_accel_raw, rtol=RTOL, atol=ATOL)
            f.write(f"accel_raw mismatches: { (~eq_acc_raw).sum() } / {len(df)}\n")

            accel_clamped = np.clip(accel_raw, 0.0, 1.0)
            out_accel = df["output_clipped_accelerate"].astype(float).to_numpy()
            eq_out_acc = np.isclose(accel_clamped, out_accel, rtol=RTOL, atol=ATOL)
            f.write(f"output_clipped_accelerate matches clamped accel: {eq_out_acc.sum()} / {len(df)}\n")

            brake_raw = np.maximum(-speed_error_calc * w_brake_gain, 0.0)
            logged_brake_raw = df["latent_brake_raw"].astype(float).to_numpy()
            eq_br_raw = np.isclose(brake_raw, logged_brake_raw, rtol=RTOL, atol=ATOL)
            f.write(f"brake_raw mismatches: { (~eq_br_raw).sum() } / {len(df)}\n")

            brake_clamped = np.clip(brake_raw, 0.0, 1.0)
            out_brake = df["output_clipped_brake"].astype(float).to_numpy()
            eq_out_brake = np.isclose(brake_clamped, out_brake, rtol=RTOL, atol=ATOL)
            f.write(f"output_clipped_brake matches clamped brake: {eq_out_brake.sum()} / {len(df)}\n")

            # check mutually exclusive accel / brake
            both_positive = np.logical_and(accel_clamped > 1e-8, brake_clamped > 1e-8)
            n_both = int(both_positive.sum())
            f.write(f"Rows with both accel>0 and brake>0: {n_both} / {len(df)}\n")
            if n_both > 0:
                idx = np.where(both_positive)[0][:5]
                f.write("Examples where both accel and brake are positive:\n")
                f.write(_first_examples(df, idx, ["output_clipped_accelerate", "output_clipped_brake", "latent_speed_error"]) + "\n")
    except Exception as e:
        _log("ERROR in test_desired_speed_and_actuators: " + str(e))
        _log(traceback.format_exc())

def test_variance_and_cov(df):
    """Recompute diagonal stds and check clamps and distributions"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_variance_and_cov ===\n")
            req = [
                "weight_var_steer", "weight_var_steer_sharp", "weight_var_steer_speed",
                "weight_var_accel", "weight_var_accel_sharp",
                "weight_var_brake", "weight_var_brake_sharp",
                "latent_sharp_ahead", "latent_speed"
            ]
            _safe_cols_exist(df, req)
            w_vs = df["weight_var_steer"].astype(float).to_numpy()
            w_vs_sh = df["weight_var_steer_sharp"].astype(float).to_numpy()
            w_vs_speed = df["weight_var_steer_speed"].astype(float).to_numpy()
            w_va = df["weight_var_accel"].astype(float).to_numpy()
            w_va_sh = df["weight_var_accel_sharp"].astype(float).to_numpy()
            w_vb = df["weight_var_brake"].astype(float).to_numpy()
            w_vb_sh = df["weight_var_brake_sharp"].astype(float).to_numpy()
            sharp = df["latent_sharp_ahead"].astype(float).to_numpy()
            speed = df["latent_speed"].astype(float).to_numpy()

            diag_std_steer = np.abs(w_vs) + np.abs(w_vs_sh) * sharp + np.abs(w_vs_speed) * speed
            diag_std_accel = np.abs(w_va) + np.abs(w_va_sh) * sharp
            diag_std_brake = np.abs(w_vb) + np.abs(w_vb_sh) * sharp

            # clamps
            std_min = 1e-3
            std_max = 1.0
            diag_std_steer_clamped = np.clip(diag_std_steer, std_min, std_max)
            diag_std_accel_clamped = np.clip(diag_std_accel, std_min, std_max)
            diag_std_brake_clamped = np.clip(diag_std_brake, std_min, std_max)

            f.write(f"Steer std min/max (pre-clamp): {diag_std_steer.min():.6f} / {diag_std_steer.max():.6f}\n")
            f.write(f"Accel std min/max (pre-clamp): {diag_std_accel.min():.6f} / {diag_std_accel.max():.6f}\n")
            f.write(f"Brake std min/max (pre-clamp): {diag_std_brake.min():.6f} / {diag_std_brake.max():.6f}\n")

            # how often clamped at min or max
            f.write(f"Steer std clamped to min { (diag_std_steer_clamped <= std_min + 1e-12).sum() } rows\n")
            f.write(f"Steer std clamped to max { (diag_std_steer_clamped >= std_max - 1e-12).sum() } rows\n")
    except Exception as e:
        _log("ERROR in test_variance_and_cov: " + str(e))
        _log(traceback.format_exc())

def test_weight_stability_and_signs(df):
    """Report weight ranges and flag sign mismatches from expected semantic signs"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_weight_stability_and_signs ===\n")
            # expected sign mapping based on model initialization (positive => +1, negative => -1, 0 => 0 if ambiguous)
            expected_signs = {
                "weight_steer_closest_x": +1,
                "weight_steer_lookahead_theta": -1,
                "weight_steer_heading_indicator": -1,
                "weight_steer_omega": +1,  # magnitude used (abs), but init +0.05
                "weight_steer_omega_speed_scale": 0,  # init 0
                "weight_steering_gain": +1,
                "weight_steer_speed_scale": +1,
                "weight_speed_base": +1,
                "weight_speed_curv": -1,
                "weight_speed_lat": -1,
                "weight_speed_slip": -1,
                "weight_speed_omega": -1,
                "weight_speed_lookahead_theta_abs": -1,
                "weight_speed_lookahead_theta_abs_speed": -1,
                "weight_speed_steer_interaction": -1,
                "weight_accel_gain": +1,
                "weight_brake_gain": +1,
                "weight_var_steer": +1,
                "weight_var_steer_sharp": +1,
                "weight_var_steer_speed": +1,
                "weight_var_accel": +1,
                "weight_var_accel_sharp": +1,
                "weight_var_brake": +1,
                "weight_var_brake_sharp": +1,
            }
            all_weights = [k for k in expected_signs.keys() if k in df.columns]
            for w in all_weights:
                vals = df[w].astype(float).to_numpy()
                v_min, v_max = float(np.min(vals)), float(np.max(vals))
                v_mean, v_std = float(np.mean(vals)), float(np.std(vals))
                f.write(f"{w}: mean={v_mean:.6f}, std={v_std:.6g}, min={v_min:.6f}, max={v_max:.6f}\n")
                expect = expected_signs[w]
                if expect != 0:
                    # determine sign by median/mean
                    med = np.median(vals)
                    if med == 0:
                        f.write(f"  WARNING: median value is zero\n")
                    else:
                        sign_med = 1 if med > 0 else -1
                        if sign_med != expect:
                            f.write(f"  SIGN MISMATCH (expected {expect}, median sign {sign_med})\n")
                # stability: if std is tiny, stable; if large, changing
                if v_std > 1e-3:
                    f.write(f"  NOTE: parameter shows non-trivial variability (std {v_std:.6g}) across dataset\n")
    except Exception as e:
        _log("ERROR in test_weight_stability_and_signs: " + str(e))
        _log(traceback.format_exc())

def test_action_saturation_and_simultaneous(df):
    """Check saturation rates and any simultaneous accel+brake"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_action_saturation_and_simultaneous ===\n")
            req = ["latent_steer_clamped", "output_clipped_accelerate", "output_clipped_brake"]
            _safe_cols_exist(df, req)
            steer = df["latent_steer_clamped"].astype(float).to_numpy()
            accel = df["output_clipped_accelerate"].astype(float).to_numpy()
            brake = df["output_clipped_brake"].astype(float).to_numpy()
            n = len(df)
            sat_steer = np.abs(steer) >= 0.98
            sat_accel = accel >= 0.999
            sat_brake = brake >= 0.999
            f.write(f"Fraction steer saturated (|steer|>=0.98): {sat_steer.sum()}/{n} = {100*sat_steer.sum()/n:.3f}%\n")
            f.write(f"Fraction accel saturated (>=0.999): {sat_accel.sum()}/{n} = {100*sat_accel.sum()/n:.3f}%\n")
            f.write(f"Fraction brake saturated (>=0.999): {sat_brake.sum()}/{n} = {100*sat_brake.sum()/n:.3f}%\n")

            both_pos = np.logical_and(accel > 1e-8, brake > 1e-8)
            f.write(f"Rows with both accel>0 and brake>0: {both_pos.sum()} / {n}\n")
            if both_pos.sum() > 0:
                idx = np.where(both_pos)[0][:5]
                f.write("Examples (first few):\n")
                f.write(_first_examples(df, idx, ["output_clipped_accelerate", "output_clipped_brake", "latent_speed_error"]) + "\n")
    except Exception as e:
        _log("ERROR in test_action_saturation_and_simultaneous: " + str(e))
        _log(traceback.format_exc())

def test_sharp_and_slip_effects(df):
    """Check that sharp ahead reduces desired_speed and slip reduces desired_speed"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_sharp_and_slip_effects ===\n")
            req = ["latent_sharp_ahead", "latent_desired_speed", "latent_wheel_slip_pos", "weight_speed_slip"]
            _safe_cols_exist(df, req)
            sharp = df["latent_sharp_ahead"].astype(float).to_numpy()
            desired = df["latent_desired_speed"].astype(float).to_numpy()
            slip = df["latent_wheel_slip_pos"].astype(float).to_numpy()
            # sharp effect
            idx_sharp = sharp == 1.0
            if idx_sharp.sum() >= 5:
                mean_sharp = desired[idx_sharp].mean()
                mean_not = desired[~idx_sharp].mean()
                tstat, pval = stats.ttest_ind(desired[idx_sharp], desired[~idx_sharp], equal_var=False)
                f.write(f"Desired speed when sharp ahead (n={idx_sharp.sum()}): mean={mean_sharp:.4f}; otherwise mean={mean_not:.4f}; t={tstat:.3f}, p={pval:.3e}\n")
            else:
                f.write(f"Not enough sharp samples ({idx_sharp.sum()}) to run t-test.\n")

            # slip effect correlation
            if np.unique(slip).size > 1:
                corr, pval = stats.pearsonr(slip, desired)
                f.write(f"Correlation desired_speed vs wheel_slip_pos: r={corr:.4f}, p={pval:.3e}\n")
                # check weight sign recorded
                w_slip = df["weight_speed_slip"].astype(float).to_numpy()
                median_w_slip = np.median(w_slip)
                f.write(f"Median weight_speed_slip in dataset: {median_w_slip:.6f}\n")
            else:
                f.write("Wheel slip has no variance in dataset (all zeros?).\n")
    except Exception as e:
        _log("ERROR in test_sharp_and_slip_effects: " + str(e))
        _log(traceback.format_exc())

def test_lookahead_and_steer_correlation(df):
    """Correlate lookahead theta mean with steer and check sign consistency with weight"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_lookahead_and_steer_correlation ===\n")
            req = ["latent_lookahead_theta_mean", "latent_steer_clamped", "weight_steer_lookahead_theta"]
            _safe_cols_exist(df, req)
            look = df["latent_lookahead_theta_mean"].astype(float).to_numpy()
            steer = df["latent_steer_clamped"].astype(float).to_numpy()
            corr, pval = stats.pearsonr(look, steer)
            f.write(f"Pearson correlation lookahead_theta_mean vs steer_clamped: r={corr:.4f}, p={pval:.3e}\n")
            # check weight sign
            w = df["weight_steer_lookahead_theta"].astype(float).to_numpy()
            wmed = np.median(w)
            f.write(f"Median weight_steer_lookahead_theta: {wmed:.6f}\n")
            if wmed < 0 and corr < 0:
                f.write("Sign consistent: negative weight and negative correlation.\n")
            elif wmed > 0 and corr > 0:
                f.write("Sign consistent: positive weight and positive correlation.\n")
            else:
                f.write("Sign potentially inconsistent between weight and empirical correlation (investigate other contributors to steering).\n")
    except Exception as e:
        _log("ERROR in test_lookahead_and_steer_correlation: " + str(e))
        _log(traceback.format_exc())

def test_omega_damping_and_steer_gain_speed_dependence(df):
    """Check omega damping sign and that steering gain decreases with speed"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_omega_damping_and_steer_gain_speed_dependence ===\n")
            req = ["weight_steer_omega", "weight_steer_omega_speed_scale", "latent_omega", "latent_speed", "latent_steering_gain_effective"]
            _safe_cols_exist(df, req)
            w_omega = df["weight_steer_omega"].astype(float).to_numpy()
            w_omega_speed = df["weight_steer_omega_speed_scale"].astype(float).to_numpy()
            omega = df["latent_omega"].astype(float).to_numpy()
            speed = df["latent_speed"].astype(float).to_numpy()
            w_speed_scale = df["weight_steer_speed_scale"].astype(float).to_numpy() if "weight_steer_speed_scale" in df.columns else np.zeros(len(df))

            omega_damping = -np.abs(w_omega) * np.abs(omega) * (1.0 + np.abs(w_omega_speed) * speed)
            n_pos = int((omega_damping > 1e-8).sum())
            f.write(f"omega_damping positive (>1e-8) in {n_pos} rows (should be 0 or near 0)\n")

            # steering gain vs speed correlation
            gain_eff = df["latent_steering_gain_effective"].astype(float).to_numpy()
            if np.unique(speed).size > 1:
                corr, pval = stats.pearsonr(speed, gain_eff)
                f.write(f"Correlation speed vs steering_gain_effective: r={corr:.4f}, p={pval:.3e}\n")
                f.write("Expected negative correlation (gain decreases with speed).\n")
            else:
                f.write("No variance in speed to check gain-speed relation.\n")
    except Exception as e:
        _log("ERROR in test_omega_damping_and_steer_gain_speed_dependence: " + str(e))
        _log(traceback.format_exc())

def test_offtrack_and_oscillation(df):
    """Check off-track frequency and steering oscillations per episode"""
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n=== test_offtrack_and_oscillation ===\n")
            req = ["obs_tile_0_x", "latent_steer_clamped"]
            if "episodeID" in df.columns:
                _safe_cols_exist(df, req + ["episodeID"])
            else:
                _safe_cols_exist(df, req)
            tile0x = df["obs_tile_0_x"].astype(float).to_numpy()
            steer = df["latent_steer_clamped"].astype(float).to_numpy()
            n = len(df)
            offtrack = np.abs(tile0x) > 0.2
            f.write(f"Off-track (|obs_tile_0_x|>0.2) rows: {offtrack.sum()} / {n} ({100.0*offtrack.sum()/n:.3f}%)\n")

            # oscillation per episode: count sign flips of steer
            if "episodeID" in df.columns:
                flips_per_episode = []
                for ep, g in df.groupby("episodeID"):
                    s = g["latent_steer_clamped"].astype(float).to_numpy()
                    if len(s) < 2:
                        continue
                    # map small values near 0 to 0 to avoid spurious flips
                    s_sign = np.sign(s)
                    s_sign[np.abs(s) < 1e-3] = 0.0
                    flips = np.sum((s_sign[:-1] * s_sign[1:]) < 0)
                    flips_per_episode.append((ep, flips, len(s)))
                if len(flips_per_episode) == 0:
                    f.write("No episodes with sufficient length to compute flips.\n")
                else:
                    flip_rates = [flips / max(1, (length - 1)) for (_, flips, length) in flips_per_episode]
                    mean_rate = float(np.mean(flip_rates))
                    f.write(f"Mean sign-flip rate per step across episodes: {mean_rate:.4f}\n")
                    # list worst episodes
                    worst = sorted(flips_per_episode, key=lambda x: x[1]/max(1, x[2]-1), reverse=True)[:5]
                    f.write("Top episodes by flip-rate (episodeID, flips, length):\n")
                    for ep, flips, length in worst:
                        f.write(f"  {ep}, flips={flips}, len={length}, rate={flips/max(1,length-1):.4f}\n")
            else:
                f.write("No episodeID column present; cannot compute per-episode oscillation stats.\n")
    except Exception as e:
        _log("ERROR in test_offtrack_and_oscillation: " + str(e))
        _log(traceback.format_exc())

def run_all(df):
    """Run all tests and write outputs to test_results.txt"""
    # initialize / clear output file
    with open(OUTPUT_FILE, "w") as f:
        f.write("Test results for RacecarPolicy\'s rollouts\n")
        f.write(f"Started at {datetime.datetime.now().isoformat()}\n\n")
    try:
        test_basic_info(df)
        test_latent_consistency(df)
        test_steering_computation(df)
        test_desired_speed_and_actuators(df)
        test_variance_and_cov(df)
        test_weight_stability_and_signs(df)
        test_action_saturation_and_simultaneous(df)
        test_sharp_and_slip_effects(df)
        test_lookahead_and_steer_correlation(df)
        test_omega_damping_and_steer_gain_speed_dependence(df)
        test_offtrack_and_oscillation(df)
        _log("All tests completed successfully.")
    except Exception as e:
        _log("ERROR in run_all: " + str(e))
        _log(traceback.format_exc())