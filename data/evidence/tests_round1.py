import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import traceback

def _to_float_array(series):
    """Convert pandas Series-ish to numpy float array, handling non-numeric gracefully."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    # don't convert NaNs to zeros here; keep NaN for comparisons and masking
    return arr

def _write(msg):
    with open("test_results.txt", "a") as f:
        f.write(msg + "\n")

def test_weights_stability(df):
    """Check whether model weights logged in df are (nearly) constant across rows."""
    try:
        _write("=== test_weights_stability ===")
        weight_cols = [
            "weight_steer_closest_x",
            "weight_steer_lookahead_theta",
            "weight_steer_heading_indicator",
            "weight_steer_omega",
            "weight_steering_gain",
            "weight_steer_speed_scale",
            "weight_speed_base",
            "weight_speed_curv",
            "weight_speed_lat",
            "weight_speed_slip",
            "weight_speed_omega",
            "weight_speed_lookahead_theta_abs",
            "weight_accel_gain",
            "weight_brake_gain",
            "weight_var_steer",
            "weight_var_steer_sharp",
            "weight_var_steer_speed",
            "weight_var_accel",
            "weight_var_accel_sharp",
            "weight_var_brake",
            "weight_var_brake_sharp",
        ]
        any_missing = False
        for c in weight_cols:
            if c not in df.columns:
                _write(f"Missing column: {c}")
                any_missing = True
        if any_missing:
            _write("Some expected weight columns are missing; subsequent weight checks will skip missing columns.")
        for w in weight_cols:
            if w not in df.columns:
                continue
            vals = _to_float_array(df[w])
            valid = np.isfinite(vals)
            if valid.sum() == 0:
                _write(f"{w}: no finite values")
                continue
            std = float(np.nanstd(vals[valid]))
            unique_vals_rounded = np.unique(np.round(vals[valid], 8))
            _write(f"{w}: std={std:.3e}, unique_count={len(unique_vals_rounded)}")
            if std > 1e-6:
                _write(f"  WARNING: {w} varies across rows (std>1e-6). This may indicate multiple model snapshots in one log or logging inconsistency.")
        _write("")  # newline
    except Exception:
        _write("ERROR in test_weights_stability:\n" + traceback.format_exc())

def test_latents_consistency(df):
    """Verify latent_* columns are consistent with obs_tile_* and obs_* columns.
    Also attempt to infer the lookahead used by matching latent_lookahead_theta_mean."""
    try:
        _write("=== test_latents_consistency ===")
        n_rows = len(df)
        # Basic direct checks
        checks = [
            ("latent_closest_x", "obs_tile_0_x"),
            ("latent_closest_theta", "obs_tile_0_theta"),
            ("latent_speed", "obs_current_speed"),
            ("latent_heading_indicator", "obs_current_angle"),
            ("latent_omega", "obs_current_omega"),
        ]
        for latent_col, obs_col in checks:
            if latent_col in df.columns and obs_col in df.columns:
                latent = _to_float_array(df[latent_col])
                obs = _to_float_array(df[obs_col])
                diff = np.abs(latent - obs)
                valid = np.isfinite(diff)
                if valid.sum() == 0:
                    _write(f"{latent_col} vs {obs_col}: no finite entries")
                    continue
                _write(
                    f"{latent_col} vs {obs_col}: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}"
                )
            else:
                _write(f"Skipping check {latent_col} vs {obs_col}: column missing")

        # Infer lookahead L by comparing latent_lookahead_theta_mean to mean of obs_tile_{0..L-1}_theta
        theta_cols = [c for c in df.columns if c.startswith("obs_tile_") and c.endswith("_theta")]
        # sort by index number to ensure order
        def tile_index(colname):
            # extract number between obs_tile_ and _theta
            try:
                return int(colname.split("_")[2])
            except Exception:
                return 999
        theta_cols = sorted(theta_cols, key=tile_index)
        if "latent_lookahead_theta_mean" in df.columns and len(theta_cols) >= 1:
            latent_mean = _to_float_array(df["latent_lookahead_theta_mean"])
            min_mae = float("inf")
            best_L = None
            max_L = min(12, len(theta_cols))
            obs_theta_matrix = np.vstack([_to_float_array(df[c]) for c in theta_cols]).T  # shape (n_rows, num_tiles)
            for L in range(1, max_L + 1):
                candidate = np.nanmean(obs_theta_matrix[:, :L], axis=1)
                diff = np.abs(candidate - latent_mean)
                valid = np.isfinite(diff)
                if valid.sum() == 0:
                    continue
                mae = float(np.nanmean(diff[valid]))
                if mae < min_mae:
                    min_mae = mae
                    best_L = L
            if best_L is not None:
                _write(f"Inferred lookahead L = {best_L} (MAE={min_mae:.4e}) comparing latent_lookahead_theta_mean to obs_tile_0..obs_tile_{best_L-1}")
            else:
                _write("Could not infer lookahead L due to NaN data")
            # Also compare lookahead abs mean
            if "latent_lookahead_theta_abs_mean" in df.columns:
                latent_abs = _to_float_array(df["latent_lookahead_theta_abs_mean"])
                candidate_abs = np.nanmean(np.abs(obs_theta_matrix[:, :best_L]), axis=1)
                diff_abs = np.abs(candidate_abs - latent_abs)
                valid = np.isfinite(diff_abs)
                if valid.sum() > 0:
                    _write(
                        f"latent_lookahead_theta_abs_mean vs computed (L={best_L}): mean_abs_diff={np.nanmean(diff_abs[valid]):.4e}, max_abs_diff={np.nanmax(diff_abs[valid]):.4e}"
                    )
        else:
            _write("Skipping lookahead inference: missing latent_lookahead_theta_mean or obs tile theta columns")

        # sharp ahead (max border)
        border_cols = [c for c in df.columns if c.startswith("obs_tile_") and c.endswith("_border")]
        border_cols = sorted(border_cols, key=tile_index)
        if "latent_sharp_ahead" in df.columns and len(border_cols) >= 1:
            # use same best_L if found above, else use 6 or available
            Luse = best_L if ("best_L" in locals() and best_L is not None) else min(6, len(border_cols))
            obs_borders = np.vstack([_to_float_array(df[c]) for c in border_cols[:Luse]]).T
            max_border = np.nanmax(obs_borders, axis=1)
            latent_sharp = _to_float_array(df["latent_sharp_ahead"])
            diff = np.abs(max_border - latent_sharp)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"latent_sharp_ahead vs max(obs_tile_{0}..obs_tile_{Luse-1}_border): mean_abs_diff={np.nanmean(diff[valid]):.4e}, mismatches={(diff[valid] > 1e-6).sum()}/{valid.sum()}")
        else:
            _write("Skipping sharp ahead check: missing columns")

        # wheel means and slip
        wheel_cols = ["obs_abs_front_left", "obs_abs_front_right", "obs_abs_rear_left", "obs_abs_rear_right"]
        if all(c in df.columns for c in wheel_cols) and "latent_wheel_abs_mean" in df.columns:
            wheels = np.vstack([_to_float_array(df[c]) for c in wheel_cols]).T
            wheel_mean = np.nanmean(wheels, axis=1)
            latent_wheel_mean = _to_float_array(df["latent_wheel_abs_mean"])
            diff = np.abs(wheel_mean - latent_wheel_mean)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"latent_wheel_abs_mean vs obs mean: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")
            # slip
            if "latent_wheel_slip_rel" in df.columns and "obs_current_speed" in df.columns:
                speed = _to_float_array(df["obs_current_speed"])
                slip = wheel_mean - speed
                latent_slip = _to_float_array(df["latent_wheel_slip_rel"])
                diff = np.abs(slip - latent_slip)
                valid = np.isfinite(diff)
                if valid.sum() > 0:
                    _write(f"latent_wheel_slip_rel vs computed: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")
        else:
            _write("Skipping wheel mean/slip checks: missing wheel or latent columns")

        _write("")  # newline
    except Exception:
        _write("ERROR in test_latents_consistency:\n" + traceback.format_exc())

def test_steering_computation(df):
    """Recompute steering_raw, steering_gain_effective, and final steer and compare to logged latents and outputs."""
    try:
        _write("=== test_steering_computation ===")
        required = [
            "latent_closest_x", "latent_lookahead_theta_mean", "latent_heading_indicator",
            "latent_omega", "latent_steering_raw", "latent_steering_gain_effective", "output_clipped_steer"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            _write(f"Skipping steering computation tests, missing columns: {missing}")
            return

        # Extract arrays
        cx = _to_float_array(df["latent_closest_x"])
        look_theta = _to_float_array(df["latent_lookahead_theta_mean"])
        heading = _to_float_array(df["latent_heading_indicator"])
        omega = _to_float_array(df["latent_omega"])
        steer_raw_logged = _to_float_array(df["latent_steering_raw"])
        steer_gain_logged = _to_float_array(df["latent_steering_gain_effective"])
        steer_out = _to_float_array(df["output_clipped_steer"])

        # weights (may be per-row)
        w_cx = _to_float_array(df["weight_steer_closest_x"]) if "weight_steer_closest_x" in df.columns else np.full_like(cx, np.nan)
        w_la = _to_float_array(df["weight_steer_lookahead_theta"]) if "weight_steer_lookahead_theta" in df.columns else np.full_like(cx, np.nan)
        w_hd = _to_float_array(df["weight_steer_heading_indicator"]) if "weight_steer_heading_indicator" in df.columns else np.full_like(cx, np.nan)
        w_omega = _to_float_array(df["weight_steer_omega"]) if "weight_steer_omega" in df.columns else np.full_like(cx, np.nan)
        w_gain = _to_float_array(df["weight_steering_gain"]) if "weight_steering_gain" in df.columns else np.full_like(cx, np.nan)
        w_speed_scale = _to_float_array(df["weight_steer_speed_scale"]) if "weight_steer_speed_scale" in df.columns else np.full_like(cx, np.nan)
        speed = _to_float_array(df["latent_speed"]) if "latent_speed" in df.columns else np.full_like(cx, np.nan)

        # Recompute steering_raw
        centering = w_cx * cx
        theta_contrib = w_la * look_theta
        heading_contrib = w_hd * heading
        omega_damping = -np.abs(w_omega) * np.abs(omega)
        steer_raw_recon = centering + theta_contrib + heading_contrib + omega_damping

        # Compare steer_raw
        diff_raw = np.abs(steer_raw_recon - steer_raw_logged)
        valid = np.isfinite(diff_raw)
        if valid.sum() > 0:
            _write(f"steering_raw: mean_abs_diff={np.nanmean(diff_raw[valid]):.4e}, max_abs_diff={np.nanmax(diff_raw[valid]):.4e}")
            n_mismatch = np.sum(diff_raw[valid] > 1e-5)
            _write(f"steering_raw: rows with abs(diff)>1e-5 : {n_mismatch}/{valid.sum()}")
        else:
            _write("No finite steering_raw comparisons available")

        # Recompute steering_gain_effective
        gain_eff_recon = w_gain / (1.0 + np.abs(w_speed_scale) * speed)
        diff_gain = np.abs(gain_eff_recon - steer_gain_logged)
        valid = np.isfinite(diff_gain)
        if valid.sum() > 0:
            _write(f"steering_gain_effective: mean_abs_diff={np.nanmean(diff_gain[valid]):.4e}, max_abs_diff={np.nanmax(diff_gain[valid]):.4e}")
        else:
            _write("No finite steering_gain comparisons available")

        # Recompute final steer and compare to logged output (clamped)
        steer_pre_gain = steer_raw_recon * gain_eff_recon
        steer_clamped = np.clip(steer_pre_gain, -1.0, 1.0)
        diff_final = np.abs(steer_clamped - steer_out)
        valid = np.isfinite(diff_final)
        if valid.sum() > 0:
            _write(f"final steer (after gain and clamp): mean_abs_diff={np.nanmean(diff_final[valid]):.4e}, max_abs_diff={np.nanmax(diff_final[valid]):.4e}")
            n_mismatch = np.sum(diff_final[valid] > 1e-5)
            _write(f"final steer: rows with abs(diff)>1e-5 : {n_mismatch}/{valid.sum()}")
        else:
            _write("No finite final steer comparisons available")

        # Quick sign checks with simple correlations
        def _corr(a, b):
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 3:
                return (np.nan, np.nan)
            try:
                r, p = stats.pearsonr(a[valid], b[valid])
                return (float(r), float(p))
            except Exception:
                return (np.nan, np.nan)

        r1, p1 = _corr(cx, steer_raw_logged)
        r2, p2 = _corr(look_theta, steer_raw_logged)
        r3, p3 = _corr(heading, steer_raw_logged)
        r4, p4 = _corr(np.abs(omega), steer_raw_logged)
        _write(f"Pearson(steering_raw, closest_x) = r={r1:.4f}, p={p1:.3e}")
        _write(f"Pearson(steering_raw, lookahead_theta_mean) = r={r2:.4f}, p={p2:.3e}")
        _write(f"Pearson(steering_raw, heading_indicator) = r={r3:.4f}, p={p3:.3e}")
        _write(f"Pearson(steering_raw, abs(omega)) = r={r4:.4f}, p={p4:.3e}")
        _write("")
    except Exception:
        _write("ERROR in test_steering_computation:\n" + traceback.format_exc())

def test_speed_and_actuation_consistency(df):
    """Recompute desired speed, accel_raw, and brake_raw and compare to logged latents and outputs."""
    try:
        _write("=== test_speed_and_actuation_consistency ===")
        required = [
            "latent_desired_speed_preclamp", "latent_desired_speed", "latent_speed_error",
            "latent_accel_raw", "latent_brake_raw", "output_clipped_accelerate", "output_clipped_brake"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            _write(f"Skipping speed/actuation tests, missing columns: {missing}")
            # still try to compute some partial checks below if possible

        # Prepare inputs and weights
        speed = _to_float_array(df["latent_speed"]) if "latent_speed" in df.columns else _to_float_array(df["obs_current_speed"])
        closest_x = _to_float_array(df.get("latent_closest_x", df.get("obs_tile_0_x", pd.Series(np.nan))))
        sharp = _to_float_array(df.get("latent_sharp_ahead", pd.Series(np.nan)))
        wheel_slip = _to_float_array(df.get("latent_wheel_slip_rel", df.get("obs_wheel_slip_rel", pd.Series(np.nan))))
        lookahead_abs = _to_float_array(df.get("latent_lookahead_theta_abs_mean", pd.Series(np.nan)))
        omega = _to_float_array(df.get("latent_omega", _to_float_array(df.get("obs_current_omega", pd.Series(np.nan)))))

        # weights
        wb = _to_float_array(df.get("weight_speed_base", pd.Series(np.nan)))
        wc = _to_float_array(df.get("weight_speed_curv", pd.Series(np.nan)))
        wl = _to_float_array(df.get("weight_speed_lat", pd.Series(np.nan)))
        ws = _to_float_array(df.get("weight_speed_slip", pd.Series(np.nan)))
        wo = _to_float_array(df.get("weight_speed_omega", pd.Series(np.nan)))
        wlta = _to_float_array(df.get("weight_speed_lookahead_theta_abs", pd.Series(np.nan)))
        w_acc = _to_float_array(df.get("weight_accel_gain", pd.Series(np.nan)))
        w_brk = _to_float_array(df.get("weight_brake_gain", pd.Series(np.nan)))

        # Recompute desired_preclamp
        desired_pre = wb + wc * sharp + wl * np.abs(closest_x) + ws * wheel_slip + wo * np.abs(omega) + wlta * lookahead_abs
        # Compare to logged latent if present
        if "latent_desired_speed_preclamp" in df.columns:
            logged_pre = _to_float_array(df["latent_desired_speed_preclamp"])
            diff = np.abs(desired_pre - logged_pre)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"desired_preclamp: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")
        # Clamp and compare desired_speed
        desired_clamped = np.clip(desired_pre, 0.0, 1.0)
        if "latent_desired_speed" in df.columns:
            logged_clamped = _to_float_array(df["latent_desired_speed"])
            diff = np.abs(desired_clamped - logged_clamped)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"desired_speed (after clamp): mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")

        # speed_error
        speed_error = desired_clamped - speed
        if "latent_speed_error" in df.columns:
            logged_err = _to_float_array(df["latent_speed_error"])
            diff = np.abs(speed_error - logged_err)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"speed_error: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")

        # accel and brake raw
        accel_raw = np.maximum(speed_error * w_acc, 0.0)
        brake_raw = np.maximum(-speed_error * w_brk, 0.0)

        if "latent_accel_raw" in df.columns:
            logged_accel_raw = _to_float_array(df["latent_accel_raw"])
            diff = np.abs(accel_raw - logged_accel_raw)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"accel_raw: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")
        if "latent_brake_raw" in df.columns:
            logged_brake_raw = _to_float_array(df["latent_brake_raw"])
            diff = np.abs(brake_raw - logged_brake_raw)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"brake_raw: mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")

        # Compare to clipped outputs if present
        if "output_clipped_accelerate" in df.columns:
            accel_out = _to_float_array(df["output_clipped_accelerate"])
            diff = np.abs(np.clip(accel_raw, 0.0, 1.0) - accel_out)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"final accel (clipped): mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")
        if "output_clipped_brake" in df.columns:
            brake_out = _to_float_array(df["output_clipped_brake"])
            diff = np.abs(np.clip(brake_raw, 0.0, 1.0) - brake_out)
            valid = np.isfinite(diff)
            if valid.sum() > 0:
                _write(f"final brake (clipped): mean_abs_diff={np.nanmean(diff[valid]):.4e}, max_abs_diff={np.nanmax(diff[valid]):.4e}")

        # Check exclusivity: accel and brake should not both be > eps
        eps = 1e-6
        if "output_clipped_accelerate" in df.columns and "output_clipped_brake" in df.columns:
            accel_out = _to_float_array(df["output_clipped_accelerate"])
            brake_out = _to_float_array(df["output_clipped_brake"])
            both_positive = np.sum((accel_out > eps) & (brake_out > eps))
            total = np.sum(np.isfinite(accel_out) & np.isfinite(brake_out))
            _write(f"Rows where accel>eps and brake>eps: {both_positive}/{total}")
            # Check alignment with speed_error sign
            speed_err_logged = _to_float_array(df.get("latent_speed_error", speed_error))
            mismatch_accel = np.sum((accel_out > eps) & (speed_err_logged <= 0))
            mismatch_brake = np.sum((brake_out > eps) & (speed_err_logged >= 0))
            _write(f"accel engaged while speed_error<=0: {mismatch_accel}/{total}")
            _write(f"brake engaged while speed_error>=0: {mismatch_brake}/{total}")

        # Check slip behaviour: correlation between wheel_slip_rel and desired_pre (we expected negative coef)
        if np.any(np.isfinite(wheel_slip)) and np.any(np.isfinite(desired_pre)):
            valid = np.isfinite(wheel_slip) & np.isfinite(desired_pre)
            if valid.sum() > 3:
                r, p = stats.pearsonr(wheel_slip[valid], desired_pre[valid])
                _write(f"Correlation desired_preclamp vs wheel_slip_rel: r={r:.4f}, p={p:.3e} (expected negative if weight_speed_slip < 0)")
                # Test separately positive and negative slip
                idx_pos = valid & (wheel_slip > 0)
                idx_neg = valid & (wheel_slip < 0)
                if idx_pos.sum() > 3:
                    rpos, ppos = stats.pearsonr(wheel_slip[idx_pos], desired_pre[idx_pos])
                    _write(f"  positive slip subset: r={rpos:.4f}, p={ppos:.3e}")
                if idx_neg.sum() > 3:
                    rneg, pneg = stats.pearsonr(wheel_slip[idx_neg], desired_pre[idx_neg])
                    _write(f"  negative slip subset: r={rneg:.4f}, p={pneg:.3e} (if positive, indicates model increases desired speed when wheels are slower than vehicle)")
        _write("")
    except Exception:
        _write("ERROR in test_speed_and_actuation_consistency:\n" + traceback.format_exc())

def test_regression_steering(df):
    """Fit linear regression: latent_steering_raw ~ [closest_x, lookahead_theta_mean, heading_indicator, abs(omega)]
    and compare coefficients to the weight columns."""
    try:
        _write("=== test_regression_steering ===")
        cols_needed = ["latent_steering_raw", "latent_closest_x", "latent_lookahead_theta_mean", "latent_heading_indicator", "latent_omega"]
        if not all(c in df.columns for c in cols_needed):
            _write(f"Skipping regression test: missing columns among {cols_needed}")
            return
        y = _to_float_array(df["latent_steering_raw"])
        X1 = _to_float_array(df["latent_closest_x"])
        X2 = _to_float_array(df["latent_lookahead_theta_mean"])
        X3 = _to_float_array(df["latent_heading_indicator"])
        X4 = np.abs(_to_float_array(df["latent_omega"]))
        valid = np.isfinite(y) & np.isfinite(X1) & np.isfinite(X2) & np.isfinite(X3) & np.isfinite(X4)
        if valid.sum() < 10:
            _write(f"Not enough valid rows ({valid.sum()}) for regression")
            return
        X = np.vstack([X1, X2, X3, X4]).T[valid]
        yv = y[valid]
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, yv)
        coefs = reg.coef_
        r2 = reg.score(X, yv)
        _write(f"Regression (no intercept) steering_raw = c1*closest_x + c2*look_theta + c3*heading + c4*abs(omega)")
        _write(f"  Coefficients: closest_x={coefs[0]:.6f}, lookahead_theta={coefs[1]:.6f}, heading={coefs[2]:.6f}, abs(omega)={coefs[3]:.6f}")
        _write(f"  R^2 = {r2:.4f}, n_samples={X.shape[0]}")

        # Compare to mean of the logged weights
        def mean_col(c):
            if c in df.columns:
                arr = _to_float_array(df[c])
                validc = np.isfinite(arr)
                return float(np.nanmean(arr[validc])) if validc.sum() > 0 else np.nan
            return np.nan

        w_cx = mean_col("weight_steer_closest_x")
        w_la = mean_col("weight_steer_lookahead_theta")
        w_hd = mean_col("weight_steer_heading_indicator")
        w_om = mean_col("weight_steer_omega")
        expected = [w_cx, w_la, w_hd, -abs(w_om) if not np.isnan(w_om) else np.nan]
        _write(f"Mean logged weights (expected coefficients): closest_x={expected[0]}, lookahead_theta={expected[1]}, heading={expected[2]}, abs(omega)={expected[3]}")
        diffs = [coefs[i] - expected[i] if not np.isnan(expected[i]) else np.nan for i in range(4)]
        _write(f"Coefficient differences (regression - logged_mean): {diffs}")
        _write("")
    except Exception:
        _write("ERROR in test_regression_steering:\n" + traceback.format_exc())

def test_speed_behavior_stats(df):
    """Statistical checks: desired speed when sharp ahead, saturation, and variance-related tests."""
    try:
        _write("=== test_speed_behavior_stats ===")
        if "latent_desired_speed" not in df.columns or "latent_sharp_ahead" not in df.columns:
            _write("Skipping some speed behavior stats: latent_desired_speed or latent_sharp_ahead missing")
        else:
            desired = _to_float_array(df["latent_desired_speed"])
            sharp = _to_float_array(df["latent_sharp_ahead"])
            valid = np.isfinite(desired) & np.isfinite(sharp)
            if valid.sum() > 3:
                d_sharp = desired[valid & (sharp == 1)]
                d_nsharp = desired[valid & (sharp == 0)]
                if d_sharp.size > 1 and d_nsharp.size > 1:
                    tstat, p = stats.ttest_ind(d_sharp, d_nsharp, equal_var=False, nan_policy='omit')
                    _write(f"Desired speed when sharp=1: mean={np.nanmean(d_sharp):.4f}, sharp=0 mean={np.nanmean(d_nsharp):.4f}, t={tstat:.4f}, p={p:.3e}")
                else:
                    _write("Not enough samples in sharp/non-sharp groups for t-test")
            else:
                _write("Not enough valid desired speed/sharp rows")

        # Saturation of desired_preclamp
        if "latent_desired_speed_preclamp" in df.columns:
            pre = _to_float_array(df["latent_desired_speed_preclamp"])
            valid = np.isfinite(pre)
            if valid.sum() > 0:
                frac_below = np.sum(pre[valid] < 0.0) / valid.sum()
                frac_above = np.sum(pre[valid] > 1.0) / valid.sum()
                _write(f"desired_preclamp out of [0,1]: frac_below={frac_below:.3f}, frac_above={frac_above:.3f} (total rows={valid.sum()})")
        else:
            _write("latent_desired_speed_preclamp missing; skipping saturation check")

        # Variance stats
        if all(c in df.columns for c in ["weight_var_steer", "weight_var_steer_sharp", "weight_var_steer_speed", "latent_sharp_ahead", "latent_speed"]):
            base = _to_float_array(df["weight_var_steer"])
            sharpp = _to_float_array(df["weight_var_steer_sharp"])
            spdcoeff = _to_float_array(df["weight_var_steer_speed"])
            sharp = _to_float_array(df["latent_sharp_ahead"])
            spd = _to_float_array(df["latent_speed"])
            diag = np.abs(base) + np.abs(sharpp) * sharp + np.abs(spdcoeff) * spd
            valid = np.isfinite(diag)
            if valid.sum() > 0:
                _write(f"steer diag std summary: mean={np.nanmean(diag[valid]):.4f}, median={np.nanmedian(diag[valid]):.4f}, 90p={np.nanpercentile(diag[valid],90):.4f}")
                n_large = np.sum(diag[valid] > 0.5)
                _write(f"  Rows with steer std > 0.5: {n_large}/{valid.sum()}")
            else:
                _write("No finite diag std entries for steer")
        else:
            _write("Skipping variance stats: missing columns")

        _write("")
    except Exception:
        _write("ERROR in test_speed_behavior_stats:\n" + traceback.format_exc())

def test_dynamics_response(df):
    """Simple causal sanity checks: speed_delta vs accel/brake, heading_delta vs steer."""
    try:
        _write("=== test_dynamics_response ===")
        if not ("episodeID" in df.columns and "latent_speed" in df.columns and "output_clipped_accelerate" in df.columns and "output_clipped_brake" in df.columns):
            _write("Skipping dynamics response checks: missing episodeID, latent_speed, or output_clipped_accelerate/brake")
            return
        # Make sure df is ordered; if Unnamed: 0 corresponds to time index we can use sort, else assume already ordered.
        # We will group by episodeID and compute next-step delta
        speed = _to_float_array(df["latent_speed"])
        acc = _to_float_array(df["output_clipped_accelerate"])
        brk = _to_float_array(df["output_clipped_brake"])
        steer = _to_float_array(df["output_clipped_steer"]) if "output_clipped_steer" in df.columns else _to_float_array(df.get("latent_steering_raw", pd.Series(np.nan)))
        angle = _to_float_array(df.get("obs_current_angle", df.get("latent_closest_theta", pd.Series(np.nan))))
        ep = df["episodeID"].to_numpy() if "episodeID" in df.columns else None

        # Build shifted arrays
        speed_next = df.groupby("episodeID")["latent_speed"].shift(-1).to_numpy() if "episodeID" in df.columns else np.full_like(speed, np.nan)
        angle_next = df.groupby("episodeID")["obs_current_angle"].shift(-1).to_numpy() if "episodeID" in df.columns and "obs_current_angle" in df.columns else np.full_like(angle, np.nan)

        speed_delta = speed_next - speed
        angle_delta = angle_next - angle

        # speed_delta vs accel/brake
        valid_speed = np.isfinite(speed_delta) & np.isfinite(acc)
        if np.sum(valid_speed) > 10:
            r_acc, p_acc = stats.pearsonr(speed_delta[valid_speed], acc[valid_speed])
            _write(f"Correlation speed_delta vs accel: r={r_acc:.4f}, p={p_acc:.3e}")
        else:
            _write("Not enough valid rows for speed_delta vs accel correlation")

        valid_speed_b = np.isfinite(speed_delta) & np.isfinite(brk)
        if np.sum(valid_speed_b) > 10:
            r_brk, p_brk = stats.pearsonr(speed_delta[valid_speed_b], brk[valid_speed_b])
            _write(f"Correlation speed_delta vs brake: r={r_brk:.4f}, p={p_brk:.3e} (expected negative or small)")
        else:
            _write("Not enough valid rows for speed_delta vs brake correlation")

        # heading/angle delta vs steering
        valid_angle = np.isfinite(angle_delta) & np.isfinite(steer)
        if np.sum(valid_angle) > 10:
            r_steer, p_steer = stats.pearsonr(angle_delta[valid_angle], steer[valid_angle])
            _write(f"Correlation angle_delta vs steer: r={r_steer:.4f}, p={p_steer:.3e} (sign depends on angle convention; magnitude indicates responsiveness)")
        else:
            _write("Not enough valid rows for angle_delta vs steer correlation")

        _write("")
    except Exception:
        _write("ERROR in test_dynamics_response:\n" + traceback.format_exc())


def run_all(df):
    """Run all tests and write a consolidated test_results.txt"""
    # overwrite the results file
    with open("test_results.txt", "w") as f:
        f.write("Policy structure tests - results\n")
        f.write("================================\n\n")

    # Run tests
    test_weights_stability(df)
    test_latents_consistency(df)
    test_steering_computation(df)
    test_speed_and_actuation_consistency(df)
    test_regression_steering(df)
    test_speed_behavior_stats(df)
    test_dynamics_response(df)

    _write("All tests completed.")