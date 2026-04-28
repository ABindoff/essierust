use extendr_api::prelude::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RAD: f64 = std::f64::consts::PI / 180.0;

/// Earth radius (km) — matches the value hard-coded in SGAT::gcDist.
const R_EARTH_KM: f64 = 6378.137;

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

/// Great-circle distance (km) using the spherical law of cosines.
///
/// Matches SGAT::gcDist exactly:
///   6378.137 * acos( min(cos_lat1 * cos_lat2 * cos(Δlon) + sin_lat1 * sin_lat2, 1) )
///
/// Accepts pre-computed sin/cos of the latitudes to avoid re-computing inside
/// inner loops.
#[inline(always)]
fn gc_dist(
    lon1: f64, sin_lat1: f64, cos_lat1: f64,
    lon2: f64, sin_lat2: f64, cos_lat2: f64,
) -> f64 {
    let cos_angle = (cos_lat1 * cos_lat2 * (RAD * (lon2 - lon1)).cos()
                     + sin_lat1 * sin_lat2)
                    .min(1.0_f64);
    R_EARTH_KM * cos_angle.acos()
}

/// Log-density of the gamma distribution (shape / rate parameterisation).
///
/// log f(x) = (shape − 1) * ln(x) − rate * x + log_norm
///
/// where `log_norm = shape * ln(rate) − lgamma(shape)` is precomputed once
/// in R and passed in to avoid computing lgamma inside the inner loop.
#[inline(always)]
fn log_dgamma(x: f64, shape_m1: f64, rate: f64, log_norm: f64) -> f64 {
    shape_m1 * x.ln() - rate * x + log_norm
}

/// Normalise a mutable slice in-place.  No-op when the sum is zero.
#[inline]
fn normalize(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s != 0.0 {
        v.iter_mut().for_each(|x| *x /= s);
    }
}

// ---------------------------------------------------------------------------
// Main kernel: one Essie sweep step with the gamma movement model
// ---------------------------------------------------------------------------

/// One step of the Essie forward **or** backward sweep using the
/// TwilightFree gamma movement kernel.
///
/// Computes for every destination location j:
///
/// ```text
/// result[j] = Σ_i  weights[i] · exp(log_dgamma(gcDist(i,j) / dt, …))
/// result     ×= ps
/// result      = normalise(result)
/// ```
///
/// Because gcDist is symmetric the same function serves both directions:
/// - **forward**  (k−1 → k): xs0 = previous locs,   xs = current locs
/// - **backward** (k+1 → k): xs0 = next locs,        xs = current locs
///
/// ### Location layout
/// Matrices are passed **column-major** (exactly as R stores them), so a
/// matrix of *n* locations occupies a flat vector of length 2n:
/// `[lon_0, lon_1, …, lon_{n-1},  lat_0, lat_1, …, lat_{n-1}]`
///
/// Pass with `as.double(xs)` from R — **no transpose needed**.
///
/// @param xs0_flat  Source locations, column-major, length `2 * n_src`.
/// @param xs_flat   Destination locations, column-major, length `2 * n_dest`.
/// @param weights   Source weights (unthresholded), length `n_src`.
/// @param ps        Destination likelihoods (from logpk), length `n_dest`.
/// @param dt        Time step between the two frames (hours).
/// @param beta_shape  Gamma shape parameter (beta[1] in R).
/// @param beta_rate   Gamma rate parameter  (beta[2] in R).
/// @param log_norm    Precomputed `beta_shape * log(beta_rate) − lgamma(beta_shape)`.
/// @param epsilon2    Sources whose weight is below `epsilon2 * max(weights)` are skipped.
/// @return Normalised weight vector of length `n_dest`.
#[extendr]
fn step_gamma(
    xs0_flat:   &[f64],
    xs_flat:    &[f64],
    weights:    &[f64],
    ps:         &[f64],
    dt:          f64,
    beta_shape:  f64,
    beta_rate:   f64,
    log_norm:    f64,
    epsilon2:    f64,
) -> Vec<f64> {
    let n_src  = weights.len();
    let n_dest = ps.len();

    // Column-major slices for destination locations.
    let lons_dest = &xs_flat[..n_dest];
    let lats_dest = &xs_flat[n_dest..];

    // Precompute trig for all destination locations — reused for every source.
    let sin_lat_dest: Vec<f64> = lats_dest.iter().map(|&la| (RAD * la).sin()).collect();
    let cos_lat_dest: Vec<f64> = sin_lat_dest.iter().map(|s| (1.0 - s * s).sqrt()).collect();

    let shape_m1 = beta_shape - 1.0;

    let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = epsilon2 * max_w;

    let mut result = vec![0.0_f64; n_dest];

    for i in 0..n_src {
        let w = weights[i];
        if w <= threshold {
            continue;
        }

        // Column-major source location.
        let lon0     = xs0_flat[i];
        let sin_lat0 = (RAD * xs0_flat[n_src + i]).sin();
        let cos_lat0 = (1.0 - sin_lat0 * sin_lat0).sqrt();

        for j in 0..n_dest {
            let dist = gc_dist(
                lon0,         sin_lat0,         cos_lat0,
                lons_dest[j], sin_lat_dest[j], cos_lat_dest[j],
            );
            // pmax(dist, 1e-6) / dt  — matches R's pmax.int(..., 1e-6) / dt[k]
            let spd     = dist.max(1e-6) / dt;
            let log_t   = log_dgamma(spd, shape_m1, beta_rate, log_norm);
            result[j]  += w * log_t.exp();
        }
    }

    for j in 0..n_dest {
        result[j] *= ps[j];
    }

    normalize(&mut result);
    result
}

// ---------------------------------------------------------------------------
// Generic accumulation kernel (for non-gamma / custom movement models)
// ---------------------------------------------------------------------------

/// Weighted accumulation of pre-computed log-transition probabilities.
///
/// Use this when the movement model is **not** the TwilightFree gamma kernel.
/// The R driver pre-computes the log-transition matrix by calling `model$logbk`
/// and passes it here as a flat row-major vector.
///
/// @param weights   Active source weights (already thresholded), length `n_active`.
/// @param logb_flat Log-transition matrix **row-major**, length `n_active × n_dest`.
/// @param ps        Destination likelihoods, length `n_dest`.
/// @return Normalised vector of length `n_dest`.
#[extendr]
fn accumulate_normalize(weights: &[f64], logb_flat: &[f64], ps: &[f64]) -> Vec<f64> {
    let n_active = weights.len();
    let n_dest   = ps.len();

    let mut result = vec![0.0_f64; n_dest];

    for i in 0..n_active {
        let w   = weights[i];
        let row = &logb_flat[i * n_dest..(i + 1) * n_dest];
        for j in 0..n_dest {
            result[j] += w * row[j].exp();
        }
    }

    for j in 0..n_dest {
        result[j] *= ps[j];
    }

    normalize(&mut result);
    result
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod essierust;
    fn step_gamma;
    fn accumulate_normalize;
}
