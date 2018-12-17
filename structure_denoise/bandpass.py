from __future__ import print_function
import scipy.signal as signal
import numpy as np
import pywt

def upsample(x, n):
    ux = np.zeros( n * len(x) )
    ux[::n] = x
    return ux 

def wavelet_frequency_responses(wavename, levels, Fs=1.0, N=1000):
    wl = pywt.Wavelet(wavename)
    max_levels = pywt.dwt_max_level(N, wavename)
    if levels is None:
        levels = max_levels
    elif levels > max_levels:
        print('Number of wavelet levels should be <= {}'.format(max_levels))
    wave = wl.dec_hi #/ np.sqrt(2)
    scale = wl.dec_lo #/ np.sqrt(2)
    om, h = signal.freqz(wave, worN=2**12)
    resps = [h]
    for v in range(1, levels):
        wave = signal.convolve( upsample(wave, 2), scale ) / np.sqrt(2)
        scale = signal.convolve( upsample(scale, 2), scale ) / np.sqrt(2)
        om, h = signal.freqz(wave, worN=2**12)
        resps.append(h)
    om, h = signal.freqz(scale, worN=2**12)
    resps.append(h)
    return om * Fs / 2 / np.pi, np.array(resps)


def plot_bandpasses_for_n(wavename, N, Fs=1.0, levels=None):
    import matplotlib.pyplot as pp
    fx, bands = wavelet_frequency_responses(wavename, levels, N=N, Fs=Fs)
    mag = np.abs(bands)
    fig = pp.figure();
    pp.semilogx(fx, mag.T, color='k')
    center_freqs = np.sum(fx * mag, axis=1) / np.sum(mag, axis=1)
    for n, f in enumerate(center_freqs):
        y = 1.1 * mag[n].max()
        pp.text(f, y, '{} Hz'.format(int(f)), ha='center')
    pp.xlabel('Frequency')
    pp.ylim(ymax=1.2 * mag.max())
    pp.title('Wavelet "{}", {} points'.format(wavename, N))
    fig.tight_layout()
    return fig
    
