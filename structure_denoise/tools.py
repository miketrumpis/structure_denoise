from __future__ import print_function
import os
from datetime import datetime
import warnings
import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from pywt import wavedec, waverec, dwt_max_level
from tqdm import tqdm, trange
from .denoise import clean_frames, clean_frames_quick, clean_frames_quickest


# ------- Headline method -------

def clean_blocks(data_source, block_size, multiresolution=False, wavelet='db2', wave_levels=None, **cleaner_kwargs):
    """
    Take blocks from a DataSource object and clean the frames by removing a subspace
    learned to represent the structured noise.
    """
    
    chan_map = data_source.channel_map
    for block, sl in tqdm(data_source.iter_blocks(block_size, return_slice=True)):
        if multiresolution:
            b_coefs = wavedec(block, wavelet, axis=1, level=wave_levels)
            clean_coefs = [clean_frames_quickest(c, chan_map, **cleaner_kwargs) for c in b_coefs]
            clean_block = waverec(clean_coefs, wavelet, axis=1)
        else:
            clean_block = clean_frames_quickest(block, chan_map, **cleaner_kwargs)
        data_source[sl] = clean_block


# ------- Diagnostics -------

def new_save_dir():
    """Create a new plots directory based on the current time"""
    save_dir = datetime.now().strftime('%y%m%d-%H%M%S')
    save_dir = os.path.join('diagnostic_plots', save_dir)
    n = 0
    while os.path.exists(save_dir + '_{}'.format(n)):
        print('adding to', save_dir)
        n += 1
    if n > 0:
        save_dir = save_dir + '_{}'.format(n)
        print('renamed', save_dir)
    os.makedirs(save_dir)
    return save_dir


def all_wavelets_diagnostics(data_source, block_size=0.5, wavelet='db2', **kwargs):
    L = int(block_size * data_source.samp_rate)
    max_level = dwt_max_level(L, wavelet)
    kwargs['multiresolution'] = True
    if kwargs.get('block', None) is None:
        start = np.random.randint(0, data_source.series_length - L)
        kwargs['block'] = start // L
    coefs = []
    save_dir = new_save_dir()
    for level in range(0, max_level + 1):
        cf = diagnostics(data_source,
                         block_size=block_size,
                         wavelet=wavelet,
                         wave_level=level,
                         return_cleaned=True,
                         save_dir=save_dir,
                         **kwargs)[2]
        coefs.append(cf)
    if kwargs.get('video', True):
        cleaned = waverec(coefs, wavelet, axis=1)
        raw = data_source[start:start + L]
        video_file = os.path.join(save_dir, 'error_image_frames_fullband.mp4')
        make_video(raw, cleaned, data_source.channel_map, video_file)    
    

def diagnostics(data_source,
                block_size=0.5,
                block=None,
                multiresolution=False,
                wavelet='db2',
                wave_level=2,
                video=True,
                save_figs=True,
                return_cleaned=False,
                save_dir='',
                **cleaner_kwargs):
    if (save_figs or video) and not save_dir:
        print('need a savedir')
        save_dir = new_save_dir()
    chan_map = data_source.channel_map
    L = int(block_size * data_source.samp_rate)
    if block is None:
        start = np.random.randint(0, data_source.series_length - L)
    else:
        start = block * L
    block = data_source[start:start + L]
    if multiresolution:
        b_coefs = wavedec(block, wavelet, axis=1)
        clean_coefs = []
        # clean_coefs = [clean_frames(c, chan_map, **cleaner_kwargs) for c in b_coefs]
        # clean_block = waverec(clean_coefs, wavelet, axis=1)
        # hold out the 2nd level for plots?
        if wave_level >= len(b_coefs):
            print('wave level too high -- using highest available')
            wave_level = len(b_coefs) - 1
        for n in range(len(b_coefs)):
            if n == wave_level:
                # c, params = clean_frames(b_coefs[n], chan_map, return_diagnostics=True, **cleaner_kwargs)
                c, params = clean_frames_quickest(b_coefs[n], chan_map, return_diagnostics=True, **cleaner_kwargs)
                clean_coefs.append(c)
                continue
            # clean_coefs[n][:] = 0
            b_coefs[n][:] = 0
            clean_coefs.append(b_coefs[n].copy())
        clean_block = waverec(clean_coefs, wavelet, axis=1)
        block = waverec(b_coefs, wavelet, axis=1)
        ## f1, f2 = make_process_figures(raw_holdout, clean_holdout, chan_map, params)
        ## f1.savefig(os.path.join(save_dir, 'denoising_figure.pdf'))
        ## f2.savefig(os.path.join(save_dir, 'subspaces_figure.pdf'))
        ## if video:
        ##     make_video(raw_holdout, clean_holdout, chan_map, os.path.join(save_dir, 'error_image_frames.mp4'))
        img_tag = 'WL_{}'.format(wave_level)
        print('Model order:', params['model_order'], 'resid order:', params['resid_basis'].shape[1])
        # print 'Resid order:', params['resid_basis'].shape[1]
    else:
        # clean_block, params = clean_frames(block, chan_map, return_diagnostics=True, **cleaner_kwargs)
        clean_block, params = clean_frames_quickest(block, chan_map, return_diagnostics=True, **cleaner_kwargs)
        img_tag = 'full_band'
    cleaner_kwargs.pop('max_order', None)
    cleaner_kwargs.pop('use_local_regression', None)
    f1, f2 = make_process_figures(block, clean_block, chan_map, params, **cleaner_kwargs)
    if save_figs:
        f1.savefig(os.path.join(save_dir, 'denoising_figure_{}.pdf'.format(img_tag)))
        f2.savefig(os.path.join(save_dir, 'subspaces_figure_{}.pdf'.format(img_tag)))
    if video:
        make_video(block, clean_block, chan_map, os.path.join(save_dir, 'error_image_frames_{}.mp4'.format(img_tag)))
    if return_cleaned:
        cleaned = clean_coefs[wave_level] if multiresolution else clean_blocks
        return f1, f2, cleaned
    return f1, f2


def make_process_figures(raw, clean, channel_map, params, **kwargs):
    from .denoise import covar_model, error_image
    import matplotlib.pyplot as pp
    error = raw - clean
    c_model = {'theta': params['theta']}
    lam, Vm = covar_model(c_model, channel_map)

    # figure of low, median, and high error frames

    frame_var = np.argsort(np.var(raw, axis=0))
    n_frames = len(frame_var)
    
    f_frames, axs = pp.subplots(3, 3, figsize=(10, 10))
    row_titles = ['Raw frame', 'Clean frame', 'Error frame']
    clim = np.percentile(raw, [5, 95])
    for r, pt in enumerate([25, 50, 75]):
        i = frame_var[int(pt * n_frames / 100.0)]
        row_images = [raw[:, i], clean[:, i], error[:, i]]
    
        for n in range(3):
            channel_map.image(row_images[n], ax=axs[r, n], cbar=False, clim=clim)
            if r == 0:
                axs[0, n].set_title(row_titles[n])
    f_frames.tight_layout()

    # figure of model AR
    n_row = 3 if 'resid_basis' in params else 2
    f_AR, axs = pp.subplots(n_row, 3, figsize=(10, 10 / 3.0 * n_row))
    # row 1: variogram, eigenvec 1, eigenvec 2
    axs[0, 0].plot(params['xb'], params['yb'], marker='s', ls='--')
    axs[0, 0].set_xlabel('Site-site distance (mm)')
    axs[0, 0].set_ylabel('Semivariane (uV^2)')
    axs[0, 0].axhline(params['y_half'], color='gray')
    axs[0, 0].axvline(params['x_half'], color='gray')
    axs[0, 0].set_title('Length scale: {:0.1f} mm'.format(params['theta']))
    channel_map.image(Vm[:, -1], cbar=False, ax=axs[0, 1])
    axs[0, 1].set_title('Eigenvec 1')
    channel_map.image(Vm[:, -2], cbar=False, ax=axs[0, 2])
    axs[0, 2].set_title('Eigenvec 2')
    
    # row 2: local filter, AR image, error image
    i = frame_var[n_frames // 2]
    ## # AR_resid = error_image(raw[:, i:i+1], c_model, params['model_order'], channel_map, **kwargs).squeeze()
    ## AR_resid = error_image(raw[:, i:i+1], c_model, len(channel_map) - 1, channel_map, **kwargs).squeeze()
    ## AR_image = raw[:, i] - AR_resid
    ## # make a prediction filter for channel 3, 3
    ## i_pred = channel_map.lookup(3, 3)
    ## i_samp = np.setdiff1d(np.arange(len(channel_map)), i_pred)
    ## from denoise import _kriging_predictor
    ## # W = _kriging_predictor(Vm, lam, i_pred, i_samp, params['model_order'])
    ## W = _kriging_predictor(Vm, lam, i_pred, i_samp, len(channel_map) - 1)
    ## ## Vr = Vm[:, -params['model_order']:]
    ## ## C_inv = np.dot(Vr[i_samp], Vr[i_samp].T)
    ## ## C_xn = np.dot(Vr[i_samp], Vr[i_pred])
    ## ## W = np.dot(C_inv, C_xn)
    ## W = channel_map.as_channels(channel_map.subset(i_samp).embed(W))
    ## row_images = [W, AR_image, AR_resid]
    ## row_titles = ['AR filter', 'AR image', 'AR error']
    ## for n in range(3):
    ##     channel_map.image(row_images[n], ax=axs[1, n], cbar=False)
    ##     axs[1, n].set_title(row_titles[n])

    # row 2: raw_image, projected image, residual image
    Vr = Vm[:, -params['model_order']:]
    raw_frame = raw[:, i]
    lowpass = np.dot(Vr, np.dot(Vr.T, raw[:, i]))
    resid = raw_frame - lowpass
    row_images = [raw_frame, lowpass, resid]
    row_titles = ['Raw', 'Lowpass {} modes'.format(params['model_order']), 'Resid']
    for n in range(3):
        channel_map.image(row_images[n], ax=axs[1, n], cbar=False)
        axs[1, n].set_title(row_titles[n])
    
    

    # row 3: eigenimage 1, 2, 3
    if n_row == 3:
        Vn = params['resid_basis']
        # row_images = [Vn[:, -1], Vn[:, -2], Vn[:, -3]]
        row_images = Vn[:, ::-1].T
        row_titles = ['Eigenvec 1', 'Eigenvec 2', 'Eigenvec 3']
        for n in range(3):
            if n >= len(row_images):
                break
            channel_map.image(row_images[n], ax=axs[2, n], cbar=False)
            axs[2, n].set_title(row_titles[n])

    f_AR.tight_layout()

    return f_frames, f_AR


def plot_projections(frames, channel_map, model_var=0.95, deviation=0.5, bias=0,
                     multiresolution=False, wavelet='db2', wave_level=4,
                     projected=True, f_idx=None):
    from .denoise import fast_semivariogram, covar_model, error_image
    import matplotlib.pyplot as pp
    # 3x3 panels:
    # 1st row: raw, "correct" projection (# of eigenvectors), residual
    # 2nd-3rd row: raw, "misjudged" projection, residual

    if multiresolution:
        coefs = wavelet_bandpass(frames, wavelet, wave_level, return_coefs=True)
        xb, yb = fast_semivariogram(coefs[wave_level], channel_map.site_combinations, xbin=0.5, trimmed=True)
        frames = waverec(coefs, wavelet, axis=1)
    else:
        xb, yb = fast_semivariogram(frames, channel_map.site_combinations, xbin=0.5, trimmed=True)

    yb = yb[xb < 0.7 * xb.max()]
    xb = xb[xb < 0.7 * xb.max()]
    y_half = 0.5 * np.percentile(yb, [10, 90]).sum()
    x_half = xb[np.where(yb > y_half)[0][0]]
    theta = x_half / np.log(2) + bias

    if not f_idx:
        f_idx = np.random.randint(0, frames.shape[1])
    F = frames[:, f_idx]
    clim = F.min(), F.max()
    n_rows = 4 if deviation > 0 else 2
    f, axs = pp.subplots(n_rows, 2, figsize=(7, 3 * n_rows))

    channel_map.image(F, cbar=False, clim=clim, ax=axs[0, 0])
    axs[0, 0].set_title('Raw frame')
    axs[0, 1].plot(xb, yb, marker='s', ls='--')
    axs[0, 1].axhline(y_half)
    axs[0, 1].axvline(x_half)
    axs[0, 1].set_title('Scale: {:.1f}'.format(theta))
    
    for row in range(1, n_rows):
        if row == 1:
            scale = 1.0
            theta_ = scale * theta
        elif row == 2:
            scale = 1 + deviation
            theta_ = scale * theta
        elif row == 3:
            scale = 1 - deviation
            theta_ = scale * theta
        lam, V = covar_model({'theta': theta_}, chan_map=channel_map)
        pct_var = np.cumsum(lam[::-1]) / np.sum(lam)
        if model_var >= pct_var.max():
            order = len(lam)
        else:
            order = np.where(pct_var > model_var)[0][0]
        if projected:
            Vr = V[:, -order:]
            smoothed = np.dot(Vr, np.dot(Vr.T, frames))
        else:
            resid = error_image(frames, {'theta': theta_}, order, chan_map=channel_map, radius=3.5, dist='inf')
            smoothed = frames - resid
        p_var = smoothed.var() / frames.var()
        images = [smoothed[:, f_idx], F - smoothed[:, f_idx]]
        for ax, vec, label in zip(axs[row], images, ('projected', 'residual')):
            if label == 'residual':
                channel_map.image(vec, cbar=False, ax=ax)
                ax.set_title(label + ' ({:.2f}x scale)'.format(scale))
            else:
                channel_map.image(vec, cbar=False, clim=clim, ax=ax)
                ax.set_title('{} dim '.format(order) + label + ' ({:.2f} var)'.format(p_var))
    f.tight_layout()
    return f


def wavelet_bandpass(frames, wavelet, wave_level, level=None, return_coefs=False):
    coefs = wavedec(frames, wavelet, level=level, axis=1)
    for n in range(len(coefs)):
        if n == wave_level:
            continue
        coefs[n][:] = 0
    if return_coefs:
        return coefs
    return waverec(coefs, wavelet, axis=1)


def make_video(raw, clean, channel_map, fname):
    error = raw - clean
    frames = np.concatenate([channel_map.embed(raw.T, axis=1),
                             channel_map.embed(clean.T, axis=1),
                             channel_map.embed(error.T, axis=1)], axis=2)
    clim = np.percentile(raw, [2, 98])
    write_frames(frames, fname, quicktime=True, origin='upper', clim=clim, figsize=(10, 3.5))
    

def setup_animated_frames(
        frames, timer='ms', time=(), static_title='', axis_toggle='on',
        figsize=None, colorbar=False, cbar_label='', cbar_orientation='vertical',
        figure_canvas=True,
        **imshow_kw
        ):
    if figure_canvas:
        from matplotlib.pyplot import figure
        f = figure(figsize=figsize)
    else:
        f = Figure(figsize=figsize)
    ax = f.add_subplot(111)
    im = ax.imshow(frames[0], **imshow_kw)
    ax.axis('image')
    ax.axis(axis_toggle)
    if isinstance(time, bool) and time:
        time = np.arange(len(frames))
        timer = 'samp'
    if len(time):
        ttl = ax.set_title('{0:.2f} {1}'.format(time[0], timer))
    elif static_title:
        ax.set_title(static_title)
        ttl = None
    else:
        ttl = None
    def _step_time(num, frames, frame_im):
        frame_im.set_data(frames[num])
        if ttl:
            ttl.set_text('{0:.2f} {1}'.format(time[num], timer))
            return (frame_im, ttl)
        return (frame_im,)
    func = lambda x: _step_time(x, frames, im)
    if colorbar:
        cb = f.colorbar(im, ax=ax, use_gridspec=True, orientation=cbar_orientation)
        cb.set_label(cbar_label)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f.tight_layout(pad=0.2)

    return f, func


def write_anim(
        fname, fig, func, n_frame,
        title='Array Movie', fps=5, quicktime=False, qtdpi=300
        ):

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title=title, artist='ecoglib')
    writer = FFMpegWriter(
        fps=fps, metadata=metadata, codec='h264'
        )
    if quicktime:
        # do arguments that are quicktime compatible
        extra_args = ['-pix_fmt', 'yuv420p', '-qp', '1']
        # yuv420p looks a bit crappy, but upping the res helps
        dpi = qtdpi
    else:
        # yuv422p seems pretty good
        extra_args = ['-pix_fmt', 'yuv422p', '-qp', '0']
        dpi = fig.dpi
    writer.extra_args = extra_args
    fname = fname.split('.mp4')[0]
    with writer.saving(fig, fname+'.mp4', dpi):
        print('Writing {0} frames'.format(n_frame))
        for n in trange(n_frame):
            func(n)
            writer.grab_frame()


def write_frames(
        frames, fname, fps=5, quicktime=False, qtdpi=300,
        title='Array movie', **anim_kwargs
        ):

    f, func = setup_animated_frames(frames, figure_canvas=False, **anim_kwargs)
    write_anim(
        fname, f, func, frames.shape[0], fps=fps, title=title,
        quicktime=quicktime, qtdpi=qtdpi
        )


def animate_frames(frames, fps=5, blit=False, **anim_kwargs):
    f, func = setup_animated_frames(frames, figure_canvas=True, **anim_kwargs)
    anim = animation.FuncAnimation(
        f, func, frames=len(frames), interval=1000.0 / fps, blit=blit
        )
    return anim
