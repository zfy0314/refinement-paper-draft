import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import traceback
import time

def _append(msg):
    """Append a message to test_results.txt with timestamp."""
    with open("test_results.txt", "a") as f:
        f.write(msg + "\n")

def _safe_get(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")
    return df[col].astype(float).values

def test_1_column_consistency(df):
    """Verify model latents correspond to raw observation columns."""
    try:
        _append("=== test_1_column_consistency ===")
        lookahead = 6  # Matches RacecarPolicy default
        n = len(df)
        # Check core latent <-> obs equality
        pairs = [
            ("latent_closest_x", "obs_tile_0_x"),
            ("latent_closest_theta", "obs_tile_0_theta"),
            ("latent_speed", "obs_current_speed"),
            ("latent_heading_indicator", "obs_current_angle"),
            ("latent_omega", "obs_current_omega"),
        ]
        for latent_col, obs_col in pairs:
            latent = _safe_get(df, latent_col)
            obs = _safe_get(df, obs_col)
            abs_diff = np.abs(latent - obs)
            _append(f"{latent_col} vs {obs_col}: mean_abs_diff={abs_diff.mean():.6e}, max_abs_diff={abs_diff.max():.6e}, frac_mismatch={(abs_diff>1e-6).mean():.3f}")

        # lookahead theta mean
        theta_cols = [f"obs_tile_{i}_theta" for i in range(lookahead)]
        for c in theta_cols:
            if c not in df.columns:
                raise KeyError(f"Missing obs tile theta column: {c}")
        theta_matrix = np.vstack([_safe_get(df, c) for c in theta_cols]).T  # (n, lookahead)
        theta_mean_obs = np.mean(theta_matrix, axis=1)
        latent_lookahead = _safe_get(df, "latent_lookahead_theta_mean")
        abs_diff = np.abs(latent_lookahead - theta_mean_obs)
        _append(f"latent_lookahead_theta_mean: mean_abs_diff={abs_diff.mean():.6e}, max_abs_diff={abs_diff.max():.6e}, frac_mismatch={(abs_diff>1e-6).mean():.3f}")

        # sharp ahead vs max border in lookahead
        border_cols = [f"obs_tile_{i}_border" for i in range(lookahead)]
        border_matrix = np.vstack([_safe_get(df, c) for c in border_cols]).T
        border_max = np.max(border_matrix, axis=1)
        latent_sharp = _safe_get(df, "latent_sharp_ahead")
        eq_frac = (np.abs(latent_sharp - border_max) < 1e-6).mean()
        _append(f"latent_sharp_ahead equality fraction with max(obs borders[0:{lookahead}]): {eq_frac:.3f}")

        # wheel slip
        wheel_cols = ["obs_abs_front_left", "obs_abs_front_right", "obs_abs_rear_left", "obs_abs_rear_right"]
        wheel_matrix = np.vstack([_safe_get(df, c) for c in wheel_cols]).T
        wheel_mean = wheel_matrix.mean(axis=1)
        latent_wheel_slip = _safe_get(df, "latent_wheel_slip")
        abs_diff = np.abs(latent_wheel_slip - wheel_mean)
        _append(f"latent_wheel_slip: mean_abs_diff={abs_diff.mean():.6e}, max_abs_diff={abs_diff.max():.6e}, frac_mismatch={(abs_diff>1e-6).mean():.3f}")

        _append("test_1_column_consistency completed successfully.\n")
    except Exception as e:
        _append("test_1_column_consistency ERROR:")
        _append(traceback.format_exc())

def test_2_action_consistency(df):
    """Recompute steering and speed-related outputs from weights+latents and compare to logged outputs."""
    try:
        _append("=== test_2_action_consistency ===")
        # pull arrays
        latent_closest_x = _safe_get(df, "latent_closest_x")
        latent_lookahead_theta = _safe_get(df, "latent_lookahead_theta_mean")
        latent_heading = _safe_get(df, "latent_heading_indicator")
        latent_omega = _safe_get(df, "latent_omega")
        steering_raw_logged = _safe_get(df, "latent_steering_raw")
        steer_out = _safe_get(df, "output_clipped_steer")

        # weights
        w_cx = _safe_get(df, "weight_steer_closest_x")
        w_lt = _safe_get(df, "weight_steer_lookahead_theta")
        w_hi = _safe_get(df, "weight_steer_heading_indicator")
        w_om = _safe_get(df, "weight_steer_omega")
        w_gain = _safe_get(df, "weight_steering_gain")

        # predicted steering_raw from weight columns (per-row)
        steering_pred = w_cx * latent_closest_x + w_lt * latent_lookahead_theta + w_hi * latent_heading + w_om * latent_omega
        abs_diff_raw = np.abs(steering_pred - steering_raw_logged)
        _append(f"steering_raw reconstruction: mean_abs_diff={abs_diff_raw.mean():.6e}, max_abs_diff={abs_diff_raw.max():.6e}, frac_mismatch={(abs_diff_raw>1e-6).mean():.3f}")

        # predicted steer after gain and clamp
        steer_scaled = steering_raw_logged * w_gain
        steer_clamped_pred = np.clip(steer_scaled, -1.0, 1.0)
        abs_diff_steer = np.abs(steer_clamped_pred - steer_out)
        _append(f"steer after gain+clamp: mean_abs_diff={abs_diff_steer.mean():.6e}, max_abs_diff={abs_diff_steer.max():.6e}, frac_mismatch={(abs_diff_steer>1e-6).mean():.3f}")

        # speed-related reconstructions
        latent_speed = _safe_get(df, "latent_speed")
        latent_sharp = _safe_get(df, "latent_sharp_ahead")
        latent_wheel_slip = _safe_get(df, "latent_wheel_slip")
        desired_logged = _safe_get(df, "latent_desired_speed")
        speed_error_logged = _safe_get(df, "latent_speed_error")

        ws_base = _safe_get(df, "weight_speed_base")
        ws_curv = _safe_get(df, "weight_speed_curv")
        ws_lat = _safe_get(df, "weight_speed_lat")
        ws_slip = _safe_get(df, "weight_speed_slip")
        ws_omega = _safe_get(df, "weight_speed_omega")

        desired_pred = ws_base + ws_curv * latent_sharp + ws_lat * np.abs(latent_closest_x) + ws_slip * latent_wheel_slip + ws_omega * latent_omega
        abs_diff_desired = np.abs(desired_pred - desired_logged)
        _append(f"desired_speed reconstruction: mean_abs_diff={abs_diff_desired.mean():.6e}, max_abs_diff={abs_diff_desired.max():.6e}, frac_mismatch={(abs_diff_desired>1e-6).mean():.3f}")

        speed_error_pred = desired_pred - latent_speed
        abs_diff_speed_error = np.abs(speed_error_pred - speed_error_logged)
        _append(f"speed_error reconstruction: mean_abs_diff={abs_diff_speed_error.mean():.6e}, max_abs_diff={abs_diff_speed_error.max():.6e}, frac_mismatch={(abs_diff_speed_error>1e-6).mean():.3f}")

        # accel / brake
        w_acc_gain = _safe_get(df, "weight_accel_gain")
        w_brake_gain = _safe_get(df, "weight_brake_gain")
        accel_raw_pred = np.maximum(0.0, speed_error_pred * w_acc_gain)
        brake_raw_pred = np.maximum(0.0, -speed_error_pred * w_brake_gain)
        accel_logged = _safe_get(df, "latent_accel_raw")
        brake_logged = _safe_get(df, "latent_brake_raw")
        out_accel = _safe_get(df, "output_clipped_accelerate")
        out_brake = _safe_get(df, "output_clipped_brake")
        _append(f"accel_raw reconstruction: mean_abs_diff={(np.abs(accel_raw_pred-accel_logged)).mean():.6e}, max_abs_diff={(np.abs(accel_raw_pred-accel_logged)).max():.6e}")
        _append(f"brake_raw reconstruction: mean_abs_diff={(np.abs(brake_raw_pred-brake_logged)).mean():.6e}, max_abs_diff={(np.abs(brake_raw_pred-brake_logged)).max():.6e}")

        # clamped actions
        accel_clamped_pred = np.clip(accel_raw_pred, 0.0, 1.0)
        brake_clamped_pred = np.clip(brake_raw_pred, 0.0, 1.0)
        _append(f"accel clamp mismatch fraction: {(np.abs(accel_clamped_pred - out_accel) > 1e-6).mean():.3f}")
        _append(f"brake clamp mismatch fraction: {(np.abs(brake_clamped_pred - out_brake) > 1e-6).mean():.3f}")

        # desired_speed bounds
        outside01_frac = ((desired_logged < 0.0) | (desired_logged > 1.0)).mean()
        _append(f"fraction of latent_desired_speed outside [0,1]: {outside01_frac:.3f}")

        _append("test_2_action_consistency completed successfully.\n")
    except Exception as e:
        _append("test_2_action_consistency ERROR:")
        _append(traceback.format_exc())

def test_3_regressions(df):
    """Fit regressions to recover weights and measure fit quality."""
    try:
        _append("=== test_3_regressions ===")
        n = len(df)
        # Steering regression (no intercept)
        Xs = np.vstack([
            _safe_get(df, "latent_closest_x"),
            _safe_get(df, "latent_lookahead_theta_mean"),
            _safe_get(df, "latent_heading_indicator"),
            _safe_get(df, "latent_omega"),
        ]).T  # (n,4)
        y_steer = _safe_get(df, "latent_steering_raw")
        reg = LinearRegression(fit_intercept=False)
        reg.fit(Xs, y_steer)
        pred = reg.predict(Xs)
        r2 = r2_score(y_steer, pred)
        coef = reg.coef_
        mean_weights = {
            "weight_steer_closest_x": _safe_get(df, "weight_steer_closest_x").mean(),
            "weight_steer_lookahead_theta": _safe_get(df, "weight_steer_lookahead_theta").mean(),
            "weight_steer_heading_indicator": _safe_get(df, "weight_steer_heading_indicator").mean(),
            "weight_steer_omega": _safe_get(df, "weight_steer_omega").mean(),
        }
        _append(f"Steering OLS (no intercept) R^2={r2:.4f}")
        _append(f"Estimated coefs: closest_x={coef[0]:.6f}, lookahead_theta={coef[1]:.6f}, heading={coef[2]:.6f}, omega={coef[3]:.6f}")
        _append("Mean recorded weights: " + ", ".join([f"{k}={v:.6f}" for k,v in mean_weights.items()]))

        # Desired speed regression (intercept allowed)
        Xd = np.vstack([
            _safe_get(df, "latent_sharp_ahead"),
            np.abs(_safe_get(df, "latent_closest_x")),
            _safe_get(df, "latent_wheel_slip"),
            _safe_get(df, "latent_omega"),
        ]).T
        y_des = _safe_get(df, "latent_desired_speed")
        regd = LinearRegression(fit_intercept=True)
        regd.fit(Xd, y_des)
        predd = regd.predict(Xd)
        r2d = r2_score(y_des, predd)
        _append(f"Desired speed OLS R^2={r2d:.4f}")
        _append(f"Estimated intercept (base)={regd.intercept_:.6f}, coefs (curv, lat, slip, omega)={regd.coef_}")
        mean_speed_weights = {
            "weight_speed_base": _safe_get(df, "weight_speed_base").mean(),
            "weight_speed_curv": _safe_get(df, "weight_speed_curv").mean(),
            "weight_speed_lat": _safe_get(df, "weight_speed_lat").mean(),
            "weight_speed_slip": _safe_get(df, "weight_speed_slip").mean(),
            "weight_speed_omega": _safe_get(df, "weight_speed_omega").mean(),
        }
        _append("Mean recorded speed weights: " + ", ".join([f"{k}={v:.6f}" for k,v in mean_speed_weights.items()]))

        _append("test_3_regressions completed successfully.\n")
    except Exception as e:
        _append("test_3_regressions ERROR:")
        _append(traceback.format_exc())

def test_4_signs_and_bounds(df):
    """Check whether learned weights match expected signs and simple bounds."""
    try:
        _append("=== test_4_signs_and_bounds ===")
        checks = {
            "weight_steer_closest_x": ("positive", lambda v: v > 0.0),
            "weight_steer_lookahead_theta": ("negative", lambda v: v < 0.0),
            "weight_steer_heading_indicator": ("negative", lambda v: v < 0.0),
            "weight_steer_omega": ("negative", lambda v: v < 0.0),
            "weight_steering_gain": ("positive", lambda v: v > 0.0),
            "weight_speed_base": ("in_[0,1]", lambda v: (v >= 0.0) and (v <= 1.0)),
            "weight_speed_curv": ("negative", lambda v: v < 0.0),
            "weight_speed_lat": ("negative", lambda v: v < 0.0),
            "weight_speed_slip": ("negative", lambda v: v < 0.0),
            "weight_speed_omega": ("negative", lambda v: v < 0.0),
            "weight_accel_gain": ("positive", lambda v: v > 0.0),
            "weight_brake_gain": ("positive", lambda v: v > 0.0),
        }
        for col, (expect, fn) in checks.items():
            if col not in df.columns:
                _append(f"{col}: MISSING")
                continue
            mean_val = _safe_get(df, col).mean()
            ok = fn(mean_val)
            _append(f"{col}: mean={mean_val:.6f}, expected {expect} -> {'PASS' if ok else 'FAIL'}")
        # variance params
        for col in ["weight_var_steer", "weight_var_accel", "weight_var_brake"]:
            if col in df.columns:
                mean_v = _safe_get(df, col).mean()
                _append(f"{col}: mean raw param={mean_v:.6f}, abs used in forward={abs(mean_v):.6f}")
            else:
                _append(f"{col}: MISSING")
        _append("test_4_signs_and_bounds completed successfully.\n")
    except Exception as e:
        _append("test_4_signs_and_bounds ERROR:")
        _append(traceback.format_exc())

def test_5_saturation_and_conflicts(df):
    """Measure actuator saturations and simultaneous accel/brake usage."""
    try:
        _append("=== test_5_saturation_and_conflicts ===")
        steer = _safe_get(df, "output_clipped_steer")
        accel = _safe_get(df, "output_clipped_accelerate")
        brake = _safe_get(df, "output_clipped_brake")
        n = len(df)
        steer_sat = (np.isclose(steer, 1.0) | np.isclose(steer, -1.0)).mean()
        accel_sat = np.isclose(accel, 1.0).mean()
        brake_sat = np.isclose(brake, 1.0).mean()
        both_nonzero = ((accel > 1e-3) & (brake > 1e-3)).mean()
        _append(f"steer saturation fraction (|steer|==1): {steer_sat:.3f}")
        _append(f"accel saturation fraction (==1): {accel_sat:.3f}")
        _append(f"brake saturation fraction (==1): {brake_sat:.3f}")
        _append(f"fraction with both accel>1e-3 and brake>1e-3: {both_nonzero:.3f}")
        _append("test_5_saturation_and_conflicts completed successfully.\n")
    except Exception as e:
        _append("test_5_saturation_and_conflicts ERROR:")
        _append(traceback.format_exc())

def test_6_speed_tracking_and_correlations(df):
    """Check correlation between speed_error and accel/brake, and general speed_error distribution."""
    try:
        _append("=== test_6_speed_tracking_and_correlations ===")
        speed_error = _safe_get(df, "latent_speed_error")
        accel = _safe_get(df, "output_clipped_accelerate")
        brake = _safe_get(df, "output_clipped_brake")
        # correlations
        try:
            r_acc, p_acc = stats.pearsonr(speed_error, accel)
            r_brk, p_brk = stats.pearsonr(speed_error, brake)
            _append(f"Pearson(speed_error, accel) = {r_acc:.4f} (p={p_acc:.3e}) expected positive")
            _append(f"Pearson(speed_error, brake) = {r_brk:.4f} (p={p_brk:.3e}) expected negative")
        except Exception as e:
            _append("Pearson correlation failed (constant array?). Details: " + str(e))
        _append(f"speed_error mean={speed_error.mean():.6f}, std={speed_error.std():.6f}")
        # fraction of steps with large error
        frac_large = (np.abs(speed_error) > 0.1).mean()
        _append(f"fraction with |speed_error|>0.1: {frac_large:.3f}")
        _append("test_6_speed_tracking_and_correlations completed successfully.\n")
    except Exception as e:
        _append("test_6_speed_tracking_and_correlations ERROR:")
        _append(traceback.format_exc())

def test_7_off_track_behavior(df):
    """Evaluate behavior when vehicle is near or beyond track edges."""
    try:
        _append("=== test_7_off_track_behavior ===")
        closest_x = _safe_get(df, "latent_closest_x")
        steer = _safe_get(df, "output_clipped_steer")
        desired = _safe_get(df, "latent_desired_speed")
        on_edge_mask = np.abs(closest_x) >= 0.2  # at/over edge
        frac_off = on_edge_mask.mean()
        _append(f"fraction of steps at/over track edge (|closest_x|>=0.2): {frac_off:.3f}")
        if frac_off > 0:
            # Expect steering sign matches closest_x sign (positive x -> positive steer)
            sign_agree = (np.sign(closest_x[on_edge_mask]) * np.sign(steer[on_edge_mask]) >= 0.0).mean()
            _append(f"fraction of off-edge steps where sign(steer) aligns with sign(closest_x) (expected correction towards center): {sign_agree:.3f}")
            desired_on = desired[~on_edge_mask]
            desired_off = desired[on_edge_mask]
            _append(f"desired_speed on-track mean={desired_on.mean():.4f}, off-track mean={desired_off.mean():.4f}")
            try:
                tstat, pval = stats.ttest_ind(desired_on, desired_off, equal_var=False)
                _append(f"t-test for desired_speed on vs off track: t={tstat:.3f}, p={pval:.3e}")
            except Exception as e:
                _append("t-test failed: " + str(e))
        _append("test_7_off_track_behavior completed successfully.\n")
    except Exception as e:
        _append("test_7_off_track_behavior ERROR:")
        _append(traceback.format_exc())

def test_8_episode_oscillations(df):
    """Compute per-episode oscillation statistics (steering sign changes, speed variance)."""
    try:
        _append("=== test_8_episode_oscillations ===")
        if "episodeID" not in df.columns:
            _append("episodeID column missing; skipping episode-level tests.")
            return
        grouped = df.groupby("episodeID")
        steer_change_rates = []
        speed_vars = []
        episodes = []
        for ep, g in grouped:
            steer = g["output_clipped_steer"].astype(float).values
            speed = g["latent_speed"].astype(float).values
            # convert to -1/0/1 sign with small threshold
            sgn = np.sign(np.where(np.abs(steer) < 1e-4, 0.0, steer))
            if len(sgn) < 2:
                continue
            sign_changes = np.sum(sgn[1:] != sgn[:-1])
            rate = sign_changes / max(1, (len(sgn)-1))
            steer_change_rates.append(rate)
            speed_vars.append(np.var(speed))
            episodes.append(ep)
        steer_change_rates = np.array(steer_change_rates)
        speed_vars = np.array(speed_vars)
        if len(steer_change_rates) == 0:
            _append("No episodes with enough length for oscillation analysis.")
            return
        _append(f"Median steer sign-change rate per step across episodes: {np.median(steer_change_rates):.4f}")
        _append(f"Top 5 episodes by steer sign-change rate:")
        top_idx = np.argsort(-steer_change_rates)[:5]
        for idx in top_idx:
            _append(f"  episode {episodes[idx]}: rate={steer_change_rates[idx]:.4f}, speed_var={speed_vars[idx]:.6f}")
        # Correlate episode-level oscillation with mean weight_steer_omega (if weight varies per row take mean per episode)
        try:
            # compute per-episode mean of weight_steer_omega
            w_omega_ep = grouped["weight_steer_omega"].mean().reindex(episodes).astype(float).values
            if len(w_omega_ep) == len(steer_change_rates):
                r, p = stats.pearsonr(steer_change_rates, w_omega_ep)
                _append(f"Pearson(steer_change_rate, mean_weight_steer_omega) = {r:.4f} (p={p:.3e}) -- expected negative (more damping -> fewer sign changes)")
        except Exception:
            _append("Could not correlate episode oscillation with weight_steer_omega (maybe column missing or constant).")
        _append("test_8_episode_oscillations completed successfully.\n")
    except Exception as e:
        _append("test_8_episode_oscillations ERROR:")
        _append(traceback.format_exc())

def test_9_variance_vs_observed(df):
    """Compare learned variance parameters to observed action variability."""
    try:
        _append("=== test_9_variance_vs_observed ===")
        steer = _safe_get(df, "output_clipped_steer")
        accel = _safe_get(df, "output_clipped_accelerate")
        brake = _safe_get(df, "output_clipped_brake")
        obs_std = {
            "steer": np.std(steer),
            "accel": np.std(accel),
            "brake": np.std(brake),
        }
        var_steer = np.abs(_safe_get(df, "weight_var_steer")).mean()
        var_accel = np.abs(_safe_get(df, "weight_var_accel")).mean()
        var_brake = np.abs(_safe_get(df, "weight_var_brake")).mean()
        _append(f"observed stds: steer={obs_std['steer']:.6f}, accel={obs_std['accel']:.6f}, brake={obs_std['brake']:.6f}")
        _append(f"learned std params (mean abs): steer={var_steer:.6f}, accel={var_accel:.6f}, brake={var_brake:.6f}")
        for name, obs in obs_std.items():
            learned = {"steer": var_steer, "accel": var_accel, "brake": var_brake}[name]
            ratio = learned / (obs + 1e-12)
            _append(f"{name}: learned_std / observed_std = {ratio:.3f}")
        _append("test_9_variance_vs_observed completed successfully.\n")
    except Exception as e:
        _append("test_9_variance_vs_observed ERROR:")
        _append(traceback.format_exc())

def test_10_sharp_ahead_distance(df):
    """When latent_sharp_ahead==1, compute how far the first sharp tile is within lookahead."""
    try:
        _append("=== test_10_sharp_ahead_distance ===")
        lookahead = 6
        if any(f"obs_tile_{i}_border" not in df.columns for i in range(lookahead)):
            _append("Missing some tile border columns; skipping sharp-ahead distance test.")
            return
        borders = np.vstack([_safe_get(df, f"obs_tile_{i}_border") for i in range(lookahead)]).T  # (n, lookahead)
        sharp_mask = _safe_get(df, "latent_sharp_ahead") > 0.5
        if sharp_mask.sum() == 0:
            _append("No rows with latent_sharp_ahead==1; skipping.")
            return
        distances = []
        for row in borders[sharp_mask]:
            # find first index where border==1
            idxs = np.where(row > 0.5)[0]
            distances.append(int(idxs[0]) if len(idxs) > 0 else np.nan)
        distances = np.array(distances, dtype=float)
        _append(f"sharp_ahead distances (first sharp tile index within lookahead): mean={np.nanmean(distances):.3f}, median={np.nanmedian(distances):.3f}")
        counts = {i: np.sum(distances==i) for i in range(lookahead)}
        _append("counts by distance: " + ", ".join([f"{i}:{counts.get(i,0)}" for i in range(lookahead)]))
        _append("test_10_sharp_ahead_distance completed successfully.\n")
    except Exception as e:
        _append("test_10_sharp_ahead_distance ERROR:")
        _append(traceback.format_exc())

def run_all(df):
    """Run all tests and write results to test_results.txt (overwrites existing file)."""
    # overwrite file and write header
    with open("test_results.txt", "w") as f:
        f.write(f"Test results generated at {time.ctime()}\n\n")
    # call tests
    tests = [
        test_1_column_consistency,
        test_2_action_consistency,
        test_3_regressions,
        test_4_signs_and_bounds,
        test_5_saturation_and_conflicts,
        test_6_speed_tracking_and_correlations,
        test_7_off_track_behavior,
        test_8_episode_oscillations,
        test_9_variance_vs_observed,
        test_10_sharp_ahead_distance,
    ]
    for t in tests:
        try:
            t(df)
        except Exception:
            _append(f"Unhandled error running {t.__name__}:")
            _append(traceback.format_exc())
    _append("\nAll tests completed.\n")