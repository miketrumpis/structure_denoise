from __future__ import print_function
import numpy as np
from scipy.ndimage import convolve
from scipy.special import expit
from .channel_map import ChannelMap

__all__ = ['clean_frames', 'error_image']

def fenced_out(samps, quantiles=(25,75), thresh=3.0, axis=None, low=True):
    """
    Threshold input sampled based on Tukey's box-plot heuristic. An
    outlier is a value that lies beyond some multiple of of an
    inter-percentile range (3 multiples of the inter-quartile range 
    is default). If the sample has an inter-percentile range of zero, 
    then the sample median is substituted.
    """

    samps = np.asanyarray(samps)
    thresh = float(thresh)
    if isinstance(samps, np.ma.MaskedArray):
        samps = samps.filled(np.nan)

    oshape = samps.shape

    if axis is None:
        # do pooled distribution
        samps = samps.ravel()
    else:
        # roll axis of interest to the end
        samps = np.rollaxis(samps, axis, samps.ndim)

    quantiles = list(map(float, quantiles))
    q_lo, q_hi = np.nanpercentile(samps, quantiles, axis=-1)
    extended_range = thresh * (q_hi - q_lo)
    if (extended_range == 0).any():
        print('Equal percentiles: estimating outlier range from median value')
        m = (extended_range > 0).astype('d')
        md_range = thresh * q_hi
        extended_range = extended_range * m + md_range * (1 - m)
        
    high_cutoff = q_hi + extended_range/2
    low_cutoff = q_lo - extended_range/2
    # don't care about warnings about comparing nans to reals
    with np.errstate(invalid='ignore'):
        out_mask = samps < high_cutoff[...,None]
        if low:
            out_mask &= samps > low_cutoff[...,None]
    # be sure to reject nans as well
    out_mask &= ~np.isnan(samps)
            
    if axis is None:
        out_mask.shape = oshape
    else:
        out_mask = np.rollaxis(out_mask, samps.ndim-1, axis)
    return out_mask


def adapt_bins(bsize, dists, return_map=False):
    """
    Adaptive x-axis binning for variograms (moves N points to be
    at the center of mass of their respective bin)
    """

    bins = [dists.min()]
    while bins[-1] + bsize < dists.max():
        bins.append( bins[-1] + bsize )
    bins = np.array(bins)
    converged = False
    n = 0
    while not converged:
        diffs = np.abs( dists - bins[:, None] )
        bin_assignment = diffs.argmin(0)
        new_bins = [ dists[ bin_assignment==b ].mean()
                     for b in range(len(bins)) ]
        new_bins = np.array(new_bins)
        new_bins = new_bins[ np.isfinite(new_bins) ]
        if len(new_bins) == len(bins):
            dx = np.linalg.norm( bins - new_bins )
            converged = dx < 1e-5
        bins = new_bins
        if n > 20:
            break
        n += 1
    if return_map:
        diffs = np.abs( dists - bins[:, None] )
        bin_assignment = diffs.argmin(0)
        return bins, bins[bin_assignment]
    return bins


def semivariogram(
        F, combs, xbin=None, robust=True,
        trimmed=True, counts=False, se=False
        ):
    """
    Classical semivariogram estimator with option for Cressie's robust
    estimator.

    Parameters
    ----------

    F : ndarray, (N, ...)
        One or more samples of N field values.
    combs : Bunch
        Object representing the site-site pairs between N field
        points. combs.p1 and combs.p2 are (site_i, site_j) indices.
        combs.dist is the distance ||site_i - site_j||. This object
        can be found from the ChannelMap.site_combinations attribute.
    xbin : float (optional)
        Bin site distances with this spacing, rather than the default
        of using all unique distances on the grid.
    robust : Bool
        If True, use Cressie's formula for robust semivariance
        estimation. Else use mean-square difference.
    trimmed : Bool
        Perform outlier detection for extreme values.
    counts : Bool
        If True, then return the bin-counts for observations at each
        lag in x.
    se : Bool
        Return the standard error of the mean.

    Returns
    -------
    x : ndarray
        lags
    sv : ndarray,
        semivariance
    Nd : ndarray
        bin counts (only if counts==True)
    se : ndarray
        standard error (only if se==True)
    
    """
    # F is an n_site field of values
    # combs is a channel combination bunch
    if xbin is None:
        x = np.unique(combs.dist)
        Nd = np.zeros(len(x), 'i')
        sv = np.empty_like(x)
    else:
        x, assignment = adapt_bins(xbin, combs.dist, return_map=True)
        Nd = np.zeros(len(x), 'i')
        sv = np.empty(len(x))
    serr = np.empty(len(x))
    for n in range(len(x)):
        if xbin is None:
            m = combs.dist == x[n]
        else:
            m = assignment == x[n]
        x_s1 = F[ combs.p1[m] ].ravel()
        x_s2 = F[ combs.p2[m] ].ravel()
        diffs = x_s1 - x_s2
        if trimmed:
            # trim outliers from the population of samples at this lag
            t = 4 if isinstance(trimmed, bool) else trimmed
            # mask differences (not raw samples)
            m = fenced_out(diffs, thresh=t)
            diffs = diffs[m]
        Nd[n] = len(diffs)
        if not Nd[n]:
            sv[n] = np.nan
            continue
        if robust:
            avg_var = np.power(np.abs(diffs), 0.5).mean() ** 4
            sv[n] = avg_var / 2 / (0.457 + 0.494 / Nd[n])
        else:
            sv[n] = 0.5 * np.mean(diffs ** 2)
        serr[n] = np.std(0.5 * (diffs ** 2)) / np.sqrt(Nd[n])
    if counts and se:
        return x, sv, Nd, serr
    if se:
        return x, sv, serr
    if counts:
        return x, sv, Nd
    return x, sv


def fast_semivariogram(
        F, combs, xbin=None, trimmed=True,
        cloud=False, counts=False, se=False, **kwargs
        ):
    """
    Semivariogram estimator with stationarity assumptions, enabling
    faster "flipped" covariance computation.

    Parameters
    ----------

    F : ndarray, (N, ...)
        One or more samples of N field values.
    combs : Bunch
        Object representing the site-site pairs between N field
        points. combs.p1 and combs.p2 are (site_i, site_j) indices.
        combs.dist is the distance ||site_i - site_j||. This object
        can be found from the ChannelMap.site_combinations attribute.
    xbin : float (optional)
        Bin site distances with this spacing, rather than the default
        of using all unique distances on the grid.
    trimmed : Bool
        Perform outlier detection for extreme values.
    cloud : Bool
        Return (robust, trimmed) estimates for all pairs.
    counts : Bool
        If True, then return the bin-counts for observations at each
        lag in x. If cloud is True, then Nd is the count of inlier
        differences for each pair.
    se : Bool
        Return the standard error of the mean.

    Returns
    -------
    x : ndarray
        lags
    sv : ndarray,
        semivariance
    Nd : ndarray
        bin counts (only if counts==True)
    se : ndarray
        standard error (only if se==True)
    
    """
    # F is an n_site field of values
    # combs is a channel combination bunch


    sv_matrix = ergodic_semivariogram(F, normed=False,
                                      mask_outliers=trimmed, **kwargs)
    x = combs.dist
    sv = sv_matrix[ np.triu_indices(len(sv_matrix), k=1) ]
    sv = np.ma.masked_invalid(sv)
    x = np.ma.masked_array(x, sv.mask).compressed()
    sv = sv.compressed()
    if cloud:
        if counts:
            return x, sv, 1
        return x, sv

    if xbin is None:
        xb = np.unique(x)
        yb = [ sv[ x == u ] for u in xb ]
    else:
        xb, assignment = adapt_bins(xbin, x, return_map=True)
        yb = [ sv[ assignment == u ] for u in xb ]
    Nd = np.array(list(map(len, yb)))

    semivar = np.array(list(map(np.mean, yb)))
    serr = np.array(list(map(lambda x: np.std(x) / np.sqrt(len(x)), yb)))
    if counts and se:
        return xb, semivar, Nd, serr
    if se:
        return xb, semivar, serr
    if counts:
        return xb, semivar, Nd
    return xb, semivar


def ergodic_semivariogram(data, normed=False, mask_outliers=True, zero_field=True, covar=False):
    #data = data - data.mean(1)[:,None]
    if zero_field:
        data = data - data.mean(0)
    if mask_outliers:
        if isinstance(mask_outliers, bool):
            thresh = 4.0
        else:
            thresh = mask_outliers
        ## pwr = np.apply_along_axis(np.linalg.norm, 0, data)
        ## m = fenced_out(pwr)
        ## data = data[:, m]
        m = fenced_out(data, thresh=thresh)
        dm = np.zeros_like(data)
        np.putmask(dm, m, data)
        data = dm
        
    if normed:
        data = data / np.std(data, axis=1, keepdims=1)
    cxx = np.einsum('ik,jk->ij', data, data)
    if mask_outliers:
        m = m.astype('i')
        N = np.einsum('ik,jk->ij', m, m)
    else:
        N = data.shape[1]
    cxx /= N
    var = cxx.diagonal()
    if covar:
        return cxx
    return 0.5 * (var[:,None] + var) - cxx


def heuristic_length_scale(frames, chan_map, xbin, length_scale_bias, max_distance):
    xb, yb = fast_semivariogram(frames, chan_map.site_combinations, xbin=xbin, trimmed=True)
    yb = yb[xb < max_distance * xb.max()]
    xb = xb[xb < max_distance * xb.max()]
    if yb[0] > yb[1]:
        yb = yb[1:]
        xb = xb[1:]
    y_half = 0.5 * np.percentile(yb, [10, 90]).sum()
    over = np.where(yb > y_half)[0][0]
    under = over - 1
    # check for local monotonicity
    while yb[under] > yb[over]:
        under -= 1
    y_gap = yb[over] - yb[under]
    x_half = xb[under] * (yb[over] - y_half) / y_gap + xb[over] * (y_half - yb[under]) / y_gap
    param = dict(theta=-x_half / np.log(0.5) + length_scale_bias)
    param['x_half'] = x_half
    param['y_half'] = y_half
    param['xb'] = xb
    param['yb'] = yb
    return param


def clean_frames(frames,
                 chan_map,
                 xbin=0.5,
                 resid_var=0.9,
                 use_local_regression=False,
                 return_diagnostics=False,
                 **model_kwargs):
    """
    This method automates the frame denoising based on finding an image subspace
    that represents structured noise. The procedure is

    1) Determine the approximate length scale of the field based on a variogram
    2) Find the field model order for a local regression that maximizes the field-to-noise variance ratio
    3) Obtain the error image for the correct model order
    4) Project the error onto the subspace that preserves 'resid_var' proportion of variance
    5) Return the raw frames minus the projected error frames

    Parameters
    ----------
    frames: ndarray (sites x samps)
    chan_map: ChannelMap
    xbin: float
        Variogram bin size
    resid_var: float
        The proportion of variance to retain in the error image subspace.
    use_local_regression: bool
        If true, skip the error refinement and return the local-regression image.
    return_diagnostics: bool
        Return various estimated parameter info in a Bunch
    model_kwargs: dict
        Keyword arguments for error_image method

    Returns
    -------
    clean_frames: ndarray (sites x samps)
    """

    xb, yb = semivariogram(frames, chan_map.site_combinations, xbin=xbin)
    y_half = 0.5 * np.percentile(yb, [10, 90]).sum()
    x_half = xb[np.where(yb > y_half)[0][0]]
    param = dict(theta=-x_half / np.log(0.5))
    param['x_half'] = x_half
    param['y_half'] = y_half
    param['xb'] = xb
    param['yb'] = yb
    # print(param)
    xcorrs, r_vars, i_vars = image_error_variance(frames, param, chan_map, **model_kwargs)
    # snr = i_vars / r_vars
    # snr = r_vars / xcorrs
    snr = i_vars * r_vars / xcorrs
    # The approximate filters do not change smoothly with order, but jump.
    # So find the first order where SNR is equal to the maximum minus a 1% tolerance
    snr_max = np.max(snr)
    order = np.where(snr > snr_max * 0.99)[0][0]
    param['model_order'] = order
    # print('model order:', order)
    clean_kwargs = model_kwargs.copy()
    clean_kwargs.pop('max_order', None)
    resid_full = error_image(frames, param, order, chan_map, **clean_kwargs)
    if use_local_regression:
        if return_diagnostics:
            return frames - resid_full, param
        return frames - resid_full
    resid_cov = np.cov(resid_full)
    lam, V = np.linalg.eigh(resid_cov + 1e-4 * np.eye(len(frames)))
    pct_var = np.cumsum(lam[::-1]) / np.sum(lam)
    resid_order = np.where(pct_var > resid_var)[0][0]
    resid_order = max(1, resid_order)
    # print('resid order:', resid_order)
    print('model order:', order, 'resid order:', resid_order)
    Vr = V[:, -resid_order:]
    resid_refined = np.dot(Vr, np.dot(Vr.T, resid_full))
    param['resid_basis'] = Vr
    if return_diagnostics:
        return frames - resid_refined, param
    return frames - resid_refined


def clean_frames_quick(frames,
                       chan_map,
                       xbin=0.5,
                       resid_var=0.9,
                       use_local_regression=False,
                       return_diagnostics=False,
                       **model_kwargs):
    """
    This method automates the frame denoising based on finding an image subspace
    that represents structured noise. The procedure is

    1) Determine the approximate length scale of the field based on a variogram
    2) Find the field model order for a local regression that maximizes the field-to-noise variance ratio
    3) Obtain the error image for the correct model order
    4) Project the error onto the subspace that preserves 'resid_var' proportion of variance
    5) Return the raw frames minus the projected error frames

    Parameters
    ----------
    frames: ndarray (sites x samps)
    chan_map: ChannelMap
    xbin: float
        Variogram bin size
    resid_var: float
        The proportion of variance to retain in the error image subspace.
    use_local_regression: bool
        If true, skip the error refinement and return the local-regression image.
    return_diagnostics: bool
        Return various estimated parameter info in a Bunch
    model_kwargs: dict
        Keyword arguments for error_image method

    Returns
    -------
    clean_frames: ndarray (sites x samps)
    """

    ## xb, yb = semivariogram(frames, chan_map.site_combinations, xbin=xbin)
    ## y_half = 0.5 * np.percentile(yb, [10, 90]).sum()
    ## x_half = xb[np.where(yb > y_half)[0][0]]
    ## param = dict(theta=-x_half / np.log(0.5))
    ## param['x_half'] = x_half
    ## param['y_half'] = y_half
    ## param['xb'] = xb
    ## param['yb'] = yb
    ## # print(param)
    ## xcorrs, r_vars, i_vars = image_error_variance(frames, param, chan_map, **model_kwargs)
    ## # snr = i_vars / r_vars
    ## # snr = r_vars / xcorrs
    ## snr = i_vars * r_vars / xcorrs
    ## # The approximate filters do not change smoothly with order, but jump.
    ## # So find the first order where SNR is equal to the maximum minus a 1% tolerance
    ## snr_max = np.max(snr)
    ## order = np.where(snr > snr_max * 0.99)[0][0]
    ## param['model_order'] = order
    ## # print('model order:', order)
    clean_kwargs = model_kwargs.copy()
    clean_kwargs.pop('max_order', None)
    param = dict(theta = 4)
    resid_full = error_image(frames, param, len(chan_map) - 1, chan_map, **clean_kwargs)
    if use_local_regression:
        if return_diagnostics:
            return frames - resid_full, param
        return frames - resid_full
    resid_cov = np.cov(resid_full)
    lam, V = np.linalg.eigh(resid_cov + 1e-4 * np.eye(len(frames)))
    pct_var = np.cumsum(lam[::-1]) / np.sum(lam)
    resid_order = np.where(pct_var > resid_var)[0][0]
    resid_order = max(1, resid_order)
    # print('resid order:', resid_order)
    # print('model order:', order, 'resid order:', resid_order)
    Vr = V[:, -resid_order:]
    resid_refined = np.dot(Vr, np.dot(Vr.T, resid_full))
    param['resid_basis'] = Vr
    if return_diagnostics:
        return frames - resid_refined, param
    return frames - resid_refined


def range_compression(frames):
    rms = frames.std(1)
    q1, q2, q3 = np.percentile(rms, [25, 50, 75])
    iqr = q3 - q1
    # linear range analogous to 3-sigma
    linear_range = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    # squeeze everything past this range with an expit function 1/(1+exp(-x))
    # with x translated and scaled such that
    # T(q3 + 2 * iqr) = 3x
    # T(q1 - 2 * iqr) = -3x

    cx_scale = np.ones_like(rms)
    m_low = rms < q1 - 1.5 * iqr
    # print('linear range:', linear_range)
    # print('3-tau compression range:', 0.5 * iqr)
    if m_low.any():
        last_linear = q1 - 1.5 * iqr
        # shift zero point
        rms_low = rms[m_low] - last_linear
        # scale to 3x ~ 0.5 * iqr (flip to positive)
        rms_low *= -3 / (0.5 * iqr)
        # (subtract positive expit values from last linear range value)
        f_out = last_linear - (2 * expit(rms_low) - 1) * (0.5 * iqr)
        cx_scale[m_low] = f_out / rms[m_low]
    m_high = rms > q3 + 1.5 * iqr
    if m_high.any():
        # shift zero point
        last_linear = q3 + 1.5 * iqr
        rms_high = rms[m_high] - last_linear
        rms_high *= 3 / (0.5 * iqr)
        # output function saturates at q3 + 2 * iqr (1/2 IQR above last linear)
        f_out = last_linear + (2 * expit(rms_high) - 1) * (0.5 * iqr)
        cx_scale[m_high] = f_out / rms[m_high]
    return cx_scale


def clean_frames_quickest(frames,
                          chan_map,
                          xbin=0.5,
                          model_var=0.95,
                          resid_var=0.9,
                          min_resid_rank=1,
                          max_image_rank=None,
                          length_scale_bias=0,
                          compress_range=False,
                          max_variogram_distance=0.7,
                          use_local_regression=False,
                          return_diagnostics=False):
    """
    This method automates the frame denoising based on finding an image subspace
    that represents structured noise. The procedure is

    1) Determine the approximate length scale of the field based on a variogram
    2) Determine the rank of a model image-space projection to preserve model_var variance
    3) Obtain the error image for the correct model order
    4) Project the error onto the subspace that preserves 'resid_var' proportion of variance
    5) Return the raw frames minus the projected error frames

    Parameters
    ----------
    frames: ndarray (sites x samps)
    chan_map: ChannelMap
    xbin: float
        Variogram bin size
    model_var: float,
        Achieve this concentration of "variance" (under the model) in the image projection
    resid_var: float
        The proportion of variance to retain in the error image subspace.
    min_resid_rank: int
        Use this mininum number of basis vectors to project noise
    max_image_rank: int
        Use this maximum number of basis vectors to project image
    use_local_regression: bool
        If true, skip the error refinement and return the local-regression image.
    length_scale_bias: float
        Bias the length scale estimate (add to it) by this amount
    return_diagnostics: bool
        Return various estimated parameter info in a Bunch
    model_kwargs: dict
        Keyword arguments for error_image method

    Returns
    -------
    clean_frames: ndarray (sites x samps)
    """

    # xb, yb = semivariogram(frames, chan_map.site_combinations, xbin=xbin)
    if compress_range:
        cx_scale = range_compression(frames)
        frames = frames * cx_scale[:, None]

    param = heuristic_length_scale(frames, chan_map, xbin, length_scale_bias, max_variogram_distance)

    # Build model covariance eigenvals / vecs
    lam, V = covar_model(param, chan_map)
    pct_var = np.cumsum(lam[::-1]) / np.sum(lam)
    order = np.where(pct_var > model_var)[0][0]
    if max_image_rank:
        order = min(max_image_rank, order)
    param['model_order'] = order
    Vr = V[:, -order:]
    smooth_frames = np.dot(Vr, np.dot(Vr.T, frames))
    resid_full = frames - smooth_frames
    
    if use_local_regression:
        if compress_range:
            smooth_frames /= cx_scale[:, None]
        if return_diagnostics:
            return smooth_frames, param
        return smooth_frames
    resid_cov = np.cov(resid_full)
    lam, V = np.linalg.eigh(resid_cov + 1e-4 * np.eye(len(frames)))
    pct_var = np.cumsum(lam[::-1]) / np.sum(lam)
    resid_order = np.where(pct_var > resid_var)[0][0]
    resid_order = max(min_resid_rank, resid_order)
    Vr = V[:, -resid_order:]
    resid_refined = np.dot(Vr, np.dot(Vr.T, resid_full))
    param['resid_basis'] = Vr
    frames = frames - resid_refined
    if compress_range:
        frames /= cx_scale[:, None]
    if return_diagnostics:
        return frames, param
    return frames


def covar_model(model, chan_map=None, frames=None, decompose=True):
    """Returns a covariance matrix based on the "model" argument"""
    
    if isinstance(model, np.ndarray) and model.ndim == 2:
        model_cov = model
    elif isinstance(model, dict):
        theta = model['theta']
        M = len(chan_map)
        model_cov = np.zeros((M, M))
        covar = np.exp(-chan_map.site_combinations.dist / theta)
        model_cov[np.triu_indices(M, k=1)] = covar
        model_cov = model_cov + model_cov.T + np.eye(M)
    elif model is None:
        model_cov = np.cov(frames)
    if decompose:
        lam, V = np.linalg.eigh(model_cov)
        return lam, V
    return model_cov


def _kriging_predictor(V, lam, i_predict, i_sample, order):
        """Reduced-rank solution of kriging predictor"""
        
        C_full = np.dot(V * lam, V.T)
        C_cutout = C_full[i_sample][:, i_sample]
        # C_xn = C[i_cutout][:, i_center]
        C_xn = np.dot(V[i_sample, -(order + 1):] * lam[-(order + 1):], V[i_predict, -(order + 1):].T)
        # C_xn = np.dot(V[i_sample] * lam, V[i_predict].T)
        # U, S, Vt = np.linalg.svd(C_cutout)
        # (USV') * W = C_xn
        # W = (Vinv(S)U') * C_xn
        # W = np.dot(np.dot(Vt[:order].T / S[:order], U[:, :order].T), C_xn)
        lamc, Vc = np.linalg.eigh(C_cutout)
        Vr = Vc[:, -order:] / np.sqrt(lamc[-order:])
        W = np.dot(Vr, np.dot(Vr.T, C_xn))
        return W


def error_image(frames, model, order, chan_map=None, radius=2.5, dist=2):
    """
    Returns the residual from a local-model autoregression. The regression weights
    are based on an exponential model of spatial covariance, adapted to the
    length scale of the given frames. Only sites within a given radius are used
    for the autoregression. To avoid over-fitting, the covariance matrix is rank-reduced
    to the given order.

    Parameters
    ----------
    frames: ndarray (sites x samps)
    model: dict, covariance matrix, or None
        Either the explicit covariance matrix, or a dictionary with the length-scale value ('theta').
        If None, then use the sample covariance matrix.
    order: int
        Model rank order.
    chan_map: ChannelMap
        Needed if "model" is a dictionary
    radius: float
        Local regression radius
    dist: int or 'inf'
        "radius" is taken as circular (dist == 2) or square (dist == 'inf').
        
    Returns
    -------
        residual: ndarray (sites x samps)
    
    """
    
    lam, V = covar_model(model, chan_map, frames)
    s = np.column_stack(chan_map.to_mat()).astype('d')
    Vr = V[:, -order:]
    Cr = np.dot(Vr, Vr.T)
    auto_regressed = np.zeros_like(frames)
    indx = np.arange(frames.shape[0])
    for n in range(frames.shape[0]):
        i = np.setdiff1d(indx, n)
        ## Cxn = Cr[i][:, n]
        ## Cx_inv = Cr[i][:, i]
        ## W = np.dot(Cx_inv, Cxn)
        if dist == 2:
            i_close = np.sum((s[i] - s[n]) ** 2, axis=1) < radius ** 2
        elif dist == 'inf':
            i_close = np.max(np.abs(s[i] - s[n]), axis=1) < radius
        P = np.diag(i_close.astype('d'))
        W = _kriging_predictor(V, lam, n, i, order)
        W = np.dot(W, P)
        W = W / W.sum()
        auto_regressed[n] = np.dot(W, frames[i])
    return frames - auto_regressed
    

def image_error_variance(frames, model, chan_map, max_order=80, radius=2.5, dist=2):
    """
    Find the error and image variances and covariances for all model orders up to max_order.
    This method approximates the error_image calculation by convolving the image frames
    with "kernelized" regression weights (necessary to speed up calculation).
    """
    
    m, n = chan_map.geometry
    m = m + (1 - m % 2)
    n = n + (1 - n % 2)
    full_chan_map = ChannelMap(np.arange(m * n), (m, n), col_major=False)
    lam, V = covar_model(model, full_chan_map)
    ic = m // 2
    jc = n // 2
    i_center = full_chan_map.lookup(ic, jc)
    i_cutout = np.setdiff1d(np.arange(m * n), [i_center])
    sites = np.column_stack(full_chan_map.to_mat()).astype('d')
    if dist == 2:
        i_far = np.sum((sites[i_cutout] - sites[i_center]) ** 2, axis=1) >= radius ** 2
    elif dist == 'inf':
        i_far = np.max(np.abs(sites[i_cutout] - sites[i_center]), axis=1) >= radius
    ## def _make_kernel(order):
    ##     Vr = V[:, -order:]
    ##     Cxn = np.dot(Vr[i_cutout], Vr[i_center])
    ##     Cx_inv = np.dot(Vr[i_cutout], Vr[i_cutout].T)
    ##     W = np.dot(Cx_inv, Cxn)
    ##     W[i_far] = 0
    ##     W_mapped = full_chan_map.subset(i_cutout).embed(W, fill=0)
    ##     k_size = int(radius)
    ##     return W_mapped[ic - k_size:ic + k_size + 1, jc - k_size:jc + k_size + 1]

    C = np.dot(V * lam, V.T)
    def _make_kernel(order):
        ## C_cutout = C[i_cutout][:, i_cutout]
        ## # C_xn = C[i_cutout][:, i_center]
        ## C_xn = np.dot(V[i_cutout, -order:] * lam[-order:], V[i_center, -order:].T)
        ## U, S, Vt = np.linalg.svd(C_cutout)
        ## # (USV') * W = C_xn
        ## # W = (Vinv(S)U') * C_xn
        ## W = np.dot(np.dot(Vt[:order].T / S[:order], U[:, :order].T), C_xn)
        W = _kriging_predictor(V, lam, i_center, i_cutout, order)
        W_mapped = full_chan_map.subset(i_cutout).embed(W, fill=0)
        k_size = int(radius)
        W_kernel = W_mapped[ic - k_size:ic + k_size + 1, jc - k_size:jc + k_size + 1]
        return W_kernel / W_kernel.sum()

    max_order = min(max_order, len(frames))
    xcorrs = np.zeros(max_order)
    r_vars = np.zeros(max_order)
    i_vars = np.zeros(max_order)
    frames_mapped = chan_map.embed(frames, fill=0)
    
    for order in range(1, max_order + 1):
        w_kernel = _make_kernel(order)
        auto_regressed = convolve(frames_mapped, w_kernel[..., None])
        img = chan_map.as_channels(auto_regressed)
        resid = frames - img
        xc = np.mean((resid - resid.mean(0)) * (img - img.mean(0)), axis=0)
        xcorrs[order - 1] = np.abs(xc).mean()
        r_vars[order - 1] = resid.var(0).mean()
        i_vars[order - 1] = img.var(0).mean()
    return xcorrs, r_vars, i_vars


def image_error_variance_simple(frames, model, chan_map, max_order=200):
    lam, V = covar_model(model, chan_map)
    max_order = min(max_order, len(frames))
    xcorrs = np.zeros(max_order)
    r_vars = np.zeros(max_order)
    i_vars = np.zeros(max_order)
    
    for order in range(1, max_order + 1):
        Vr = V[:, -order:]
        img = np.dot(Vr, np.dot(Vr.T, frames))
        resid = frames - img
        xc = np.mean((resid - resid.mean(0)) * (img - img.mean(0)), axis=0)
        xcorrs[order - 1] = np.abs(xc).mean()
        r_vars[order - 1] = resid.var(0).mean()
        i_vars[order - 1] = img.var(0).mean()
    return xcorrs, r_vars, i_vars


def image_error_variance_simple2(frames, model, chan_map, max_order=200):
    lam, V = covar_model(model, chan_map)
    max_order = min(max_order, len(frames))
    xcorrs = np.zeros(max_order)
    r_vars = np.zeros(max_order)
    i_vars = np.zeros(max_order)
    all_i = np.arange(len(frames))
    sub_i = [np.setdiff1d(all_i, [n]) for n in range(len(frames))]
    
    for order in range(1, max_order + 1):
        Vr = V[:, -order:]
        img = np.dot(Vr, np.dot(Vr.T, frames))
        resid = frames - img
        xc = np.mean((resid - resid.mean(0)) * (img - img.mean(0)), axis=0)
        xcorrs[order - 1] = np.abs(xc).mean()
        r_vars[order - 1] = resid.var(0).mean()
        i_vars[order - 1] = img.var(0).mean()
    return xcorrs, r_vars, i_vars


def drive_error_img2(frames, model, chan_map, max_order=100):
    lam, V = covar_model(model, chan_map)
    s = np.column_stack(chan_map.to_mat()).astype('d')

    max_order = min(max_order, len(frames))
    xcorrs = np.zeros(max_order)
    r_vars = np.zeros(max_order)
    i_vars = np.zeros(max_order)
    all_i = np.arange(len(frames))
    sub_i = [np.setdiff1d(all_i, [n]) for n in range(len(frames))]
    for order in range(1, max_order + 1):
        Vr = V[:, -order:]
        auto_regressed = np.zeros_like(frames)
        for n in range(frames.shape[0]):
            i = sub_i[n]
            # Cr = np.dot(Vr, Vr.T)
            # Cxn = Cr[i][:, n]
            # Cxn = np.dot(Vr[i], Vr[n])

            # Cx_inv = Cr_inv[i][:, i]
            # Cx_inv = np.dot(Vr[i], Vr[i].T)
            # W = np.dot(Cx_inv, Cxn)
            W = _kriging_predictor(V, lam, n, i, order)

            ## i_close = np.sum((s[i] - s[n]) ** 2, axis=1) < 2.5 ** 2
            ## P = np.diag(i_close.astype('d'))
            ## W = np.dot(W, P)
            auto_regressed[n] = np.dot(W, frames[i])

        resid = frames - auto_regressed
        # resid = error_image(frames, model, order, chan_map)
        img = frames - resid
        xc = np.mean((resid - resid.mean(0)) * (img - img.mean(0)), axis=0)
        # xcorrs[order - 1] = np.abs(xc).mean()
        xcorrs[order - 1] = np.abs(xc / np.sqrt(resid.var(0) * img.var(0))).mean()
        r_vars[order - 1] = resid.var(0).mean()
        i_vars[order - 1] = img.var(0).mean()
    return xcorrs, r_vars, i_vars
