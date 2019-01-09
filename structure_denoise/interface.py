from __future__ import print_function
import argparse
import numpy as np
from .bandpass import plot_bandpasses_for_n
from .tools import plot_projections, all_wavelets_diagnostics, diagnostics, clean_blocks
from .data_io import *


__all__ = ['parser', 'plot_wavelet_mode', 'plot_projection_mode']


class DataSourceError(Exception):
    pass


class CommentArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(CommentArgumentParser, self).__init__(*args, **kwargs)
 
    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


parser = CommentArgumentParser(description='Structure denoising command-line-interface',
                               fromfile_prefix_chars='@')
# Common resources
# Data source spec
parser.add_argument('--data-source', type=str, default='none', help='Path to data resource')
parser.add_argument('--source-type', type=str, default='none', help='pesaran, open ephys, ...')
parser.add_argument('--exclude-channels', type=int, nargs='+', help='Drop these channels from the array')
parser.add_argument('--save-mod', type=str, default='clean', help='Modifier name for cleaned file')
parser.add_argument('--channel-map', type=str, default='', help='Name of channel map (for Intan-sources)')
parser.add_argument('--reraise', action='store_true')

# Block processing spec
parser.add_argument('--multiresolution', action='store_true', help='Denoise wavelet scales')
parser.add_argument('--wavelet-name', type=str, default='db2', help='PyWavelets wavelet name')
parser.add_argument('--wavelet-levels', type=int, default=None,
                    help='Number of decomposition levels (otherwise auto determined)')
parser.add_argument('--block-size', type=float, default=0.5, help='Processing block size in seconds')
parser.add_argument('--skip-lowpass', action='store_true', help='Skip denoising smoothest wavelet scale')

# Image projection spec
parser.add_argument('--model-var', type=float, default=0.9, help='Model covariance proportion')
parser.add_argument('--resid-var', type=float, default=0.9, help='Noise covariance proportion')
parser.add_argument('--length-scale-bias', type=float, default=0, help='Bias length scale estimates')
parser.add_argument('--min-resid-rank', type=int, default=1, help='Minimum dimension of noise space')
parser.add_argument('--max-image-rank', type=int, default=None, help='Maximum dimension of image space')
parser.add_argument('--compress-range', action='store_true', help='Compress RMS range of channels before '
                                                                  'decompositions (avoids attributing variance to '
                                                                  'outlying channels)')

# Basic usage: actually runs the method!
parser.add_argument('--process-mode', action='store_true', help='Do batch artifact subtraction and save results')

# Diagnostic mode (makes before/after plots and residual basis details for current parameters)
parser.add_argument('--diagnostic-mode', action='store_true', help='Make diagnostic plots for current settings')
parser.add_argument('--diagnostic-wavelet-level', type=str, default='2',
                    help='Wavelet level (or "all") for diagnostics (not applicable if multiresolution is not set)')
parser.add_argument('--diagnostic-skip-video', action='store_true', help='Skip (time consuming) video writing')
                    
# Interaction for mode to plot wavelet bandpasses
# also needs: wavelet-name
parser.add_argument('--plot-wavelet-mode', action='store_true', help='Plot wavelet bands')
parser.add_argument('--wavelet-length', type=int, default=1000, help='Sequence length for wavelet decomp')
parser.add_argument('--wavelet-samp-rate', type=float, default=1.0, help='Sampling rate')

# Interaction for mode to test image-space projection
# also needs: data-source-spec + block-processing-spec + image-projection-spec
parser.add_argument('--plot-projection-mode', action='store_true', help='Demo image space projection')
parser.add_argument('--projection-wave-level', type=int, default=3, help='Use this wavelet scale level')
parser.add_argument('--projection-deviation', type=float, default=0.5, help='Show length scale deviation')
parser.add_argument('--projection-frame-start', type=float, default=-1, help='Start frames here (-1 for random)')

def get_data_source(parsed_args, **kwargs):
    dtype = parsed_args.source_type
    path = parsed_args.data_source
    excluded = parsed_args.exclude_channels
    cm = parsed_args.channel_map
    if not cm:
        cm = 'psv_244_intan'
    if not excluded:
        excluded = []
    print('excluded:', excluded)
    save_mod = parsed_args.save_mod
    if dtype.lower() == 'pesaran':
        return PesaranDataSource(path, exclude_channels=excluded, save_mod=save_mod, **kwargs)
    elif dtype.lower() == 'open-ephys':
        return OpenEphysHDFSource(path, exclude_channels=excluded, save_mod=save_mod, electrode_name=cm, **kwargs)

    raise DataSourceError('Unknown data source type {}'.format(dtype))


def get_cleaning_args(parsed_args):
    kwargs = dict()
    kwargs['multiresolution'] = parsed_args.multiresolution
    kwargs['wavelet'] = parsed_args.wavelet_name
    kwargs['wave_levels'] = parsed_args.wavelet_levels
    kwargs['block_size'] = parsed_args.block_size
    kwargs['skip_lowpass'] = parsed_args.skip_lowpass
    kwargs['model_var'] = parsed_args.model_var
    kwargs['resid_var'] = parsed_args.resid_var
    kwargs['max_image_rank'] = parsed_args.max_image_rank
    kwargs['min_resid_rank'] = parsed_args.min_resid_rank
    kwargs['length_scale_bias'] = parsed_args.length_scale_bias
    kwargs['compress_range'] = parsed_args.compress_range
    return kwargs


def processing_mode(parsed_args):
    source = get_data_source(parsed_args, saving=True)
    kwargs = get_cleaning_args(parsed_args)
    block_size = kwargs.pop('block_size')
    clean_blocks(source, block_size, **kwargs)


def plot_wavelet_mode(parsed_args):
    import matplotlib.pyplot as pp
    wavename = parsed_args.wavelet_name
    sequence_length = parsed_args.wavelet_length
    sampling_rate = parsed_args.wavelet_samp_rate
    levels = parsed_args.wavelet_levels
    plot_bandpasses_for_n(wavename, sequence_length, Fs=sampling_rate, levels=levels)
    pp.show()


def plot_projection_mode(parsed_args):
    import matplotlib.pyplot as pp
    source = get_data_source(parsed_args, saving=False)
    block_samps = int(parsed_args.block_size * source.samp_rate)
    start = int(parsed_args.projection_frame_start * source.samp_rate)
    if start < 0:
        start = np.random.randint(0, source.series_length - block_samps)
    frames = source[start:start + block_samps]
    plot_projections(frames,
                     source.channel_map,
                     model_var=parsed_args.model_var,
                     deviation=parsed_args.projection_deviation,
                     bias=parsed_args.length_scale_bias,
                     multiresolution=parsed_args.multiresolution,
                     wavelet=parsed_args.wavelet_name,
                     wave_level=parsed_args.projection_wave_level,
                     compress_range=parsed_args.compress_range)
    pp.show()    


def diagnostic_mode(parsed_args):
    import matplotlib.pyplot as pp
    kwargs = get_cleaning_args(parsed_args)
    multiresolution = kwargs.pop('multiresolution')
    # wave levels not used here
    kwargs.pop('wave_levels')
    # skip-lowpass not used here
    kwargs.pop('skip_lowpass')
    # wavelet = kwargs.pop('wavelet', None)
    video = not parsed_args.diagnostic_skip_video
    source = get_data_source(parsed_args, saving=False)
    wave_level = parsed_args.diagnostic_wavelet_level
    if isinstance(wave_level, str) and wave_level.lower() == 'all':
        all_waves = True
    else:
        wave_level = int(wave_level)
        all_waves = False
    if multiresolution and all_waves:
        all_wavelets_diagnostics(source, video=video, **kwargs)
    else:
        diagnostics(source, multiresolution=multiresolution, wave_level=wave_level, video=video, **kwargs)
    pp.show()

