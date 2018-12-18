from __future__ import print_function
import os
import numpy as np
import mat4py
import h5py
from .channel_map import ChannelMap, Bunch, get_electrode_map


__all__ = ['DataSource', 'PesaranDataSource', 'OpenEphysHDFSource']


def get_pesaran_metadata(recording):
    """Scan the pesaran-style metadata struct for info"""
    rec_name = os.path.split(recording)[1]
    exp_struct = os.path.join(recording, 'rec{}.experiment.mat'.format(rec_name))
    exp = mat4py.loadmat(exp_struct)['experiment']
    md = Bunch()

    # Raw acquisition binary brick or hdf5 file
    brick = os.path.join(recording, 'rec{}.nspike.dat'.format(rec_name))
    hdf = os.path.join(recording, 'raw_packed.h5')
    # prefer hdf5 over brick
    if os.path.exists(hdf):
        md.raw_type = 'hdf5'
        md.raw_file = hdf
    elif os.path.exists(brick):
        md.raw_file = brick
        md.raw_type = 'brick'
    else:
        md.raw_file = ''
        md.raw_type = 'none'

    # Base name and this particular name
    md.name_base = exp['hardware']['microdrive']['name_base']
    md.rec_name = exp['hardware']['microdrive']['name']
        
    # Sampling rate
    md.fs = exp['hardware']['acquisition']['samplingrate']

    # Binary type (This appears to be wrong?)
    # dtype = exp['hardware']['acquisition']['data_format']
    md.dtype = np.dtype('h')

    # Electrode channels in the brick stream
    md.stream_channels = np.array(exp['hardware']['microdrive']['electrodes']['channelid']) - 1

    # Build the channel map from the electrode matrix positions
    positions = exp['hardware']['microdrive']['electrodes']['position']
    coordinates = [(p['row'], p['col']) for p in positions]
    md.cm = ChannelMap.from_index(coordinates, (16, 16), col_major=False, pitch=0.75)

    # Determine number of samples in the stream to find raw stream shape
    if md.raw_type == 'hdf5':
        with h5py.File(md.raw_file, 'r') as f:
            md.raw_matrix_shape = f['raw_data'].shape
    elif md.raw_type == 'brick':
        n_chan = exp['hardware']['acquisition']['num_channels_streamed']
        samp_length = md.dtype.itemsize
        raw_samps = os.stat(md.raw_file).st_size // samp_length
        chan_samps = raw_samps / float(n_chan)
        if chan_samps != np.floor(chan_samps):
            raise RuntimeError('File size does not divide evenly into stream channels')
        md.raw_matrix_shape = (int(chan_samps), n_chan)
    else:
        md.raw_matrix_shape = None
    
    return md


class DataSource(object):
    """Data source providing access from file-mapped LFP signals.

    >>> batch = data_source[a:b]
    This syntax returns an array timeseries in (channels, samps) shape, including only electrode array channels (but
    excluding any channels rejected by, e.g., manual inspection).

    >>> data_source[a:b] = filtered_batch
    If the DataSource was constructed with "saving=True", then this writes array channels into a file that is
    compatible with the geometry of the original source file.

    >>> data_source.channel_map
    This is a ChannelMap object, providing channel to electrode-site map information and methods.
    """

    def __init__(self, array, channel_map, samp_rate, exclude_channels=[],
                 is_transpose=False, saving=False, save_mod='clean'):
        self.array = array
        self.channel_map = channel_map
        # keep this in case of HDF5 export
        self.__full_channel_map = channel_map[:]
        self.samp_rate = samp_rate
        self.is_transpose = is_transpose
        self.saving = saving
        if len(exclude_channels):
            n_chan = len(channel_map)
            mask = np.ones(n_chan, '?')
            mask[exclude_channels] = False
            self.channel_map = channel_map.subset(mask)
            self._channel_mask = mask
        else:
            self._channel_mask = None
        self.series_length = self.array.shape[1 - int(is_transpose)]
        if saving:
            self._initialize_output(save_mod)


    def _initialize_output(self, file_mod='clean'):
        # Must be overloaded
        # defines self.output_file and self._write_array
        pass


    def _output_channel_subset(self, array_block):
        """Returns the subset of array channels defined by a channel mask"""
        # The subset of data channels that are array channels is defined in particular data source types
        if self._channel_mask is None:
            return array_block
        return array_block[self._channel_mask]


    def _full_channel_set(self, array_block):
        """Writes a subset of array channels into the full set of array channels"""
        if self._channel_mask is None:
            return array_block
        n_chan = len(self._channel_mask)
        full_block = np.zeros((n_chan, array_block.shape[1]))
        full_block[self._channel_mask] = array_block
        return full_block


    def __getitem__(self, slicer):
        """Return the sub-series of samples selected by slicer on (possibly a subset of) all channels"""
        # data goes out as [subset]channels x time
        if self.is_transpose:
            sub_array = self.array[slicer, :].T
        else:
            sub_array = self.array[:, slicer]
        return self._output_channel_subset(sub_array)


    def __setitem__(self, slicer, data):
        """Write the sub-series of samples selected by slicer (from possibly a subset of channels)"""
        # data comes in as [subset]channels x time
        if not self.saving:
            print('This object has no write file')
            return None
        full_data = self._full_channel_set(data)
        if self.is_transpose:
            self._write_array[slicer, :] = full_data.T
        else:
            self._write_array[:, slicer] = full_data


    def iter_blocks(self, block_length, return_slice=True):
        """Yield data blocks with given length (in seconds)"""
        L = int(block_length * self.samp_rate)
        T = self.series_length
        N = T // L
        if L * N < T:
            N += 1
        for i in range(N):
            start = i * L
            if start >= T:
                raise StopIteration
            end = min(T, (i + 1) * L)
            sl = slice(start, end)
            if return_slice:
                yield self[sl], sl
            else:
                yield self[sl]


    def write_parameters(self, **params):
        """Store processing parameters"""
        # must be overloaded!
        pass


    # Incomplete
    # def to_hdf5(self, which='cleaned'):
    #     if which.lower() == 'cleaned':
    #         array = self._write_array
    #     else:
    #         array = self.array
    #     shape = array.shape
    #     if self.is_transpose:
    #         samps, chans = shape
    #     else:
    #         chans, samps = shape
    #
    #     output = os.path.splitext(self.output_file)[0] + '.h5'
    #     hdf = h5py.File(output, 'w')
    #     hdf.create_dataset('samp_rate', shape=(), data=self.samp_rate)
    #     b_pickle = Bunch(channel_map=self.__full_channel_map)
        
        
    
class PesaranDataSource(DataSource):
    """For "Pesaran Lab" style datasets.

    This source currently only uses the "LFP" file as a source.
    """

    def __init__(self, recording_dir, use_lfp=True, downsamp=None, **kwargs):
        metadata = get_pesaran_metadata(recording_dir)
        if use_lfp:
            base_rec = os.path.split(recording_dir)[1]
            lfp_file = os.path.join(recording_dir, 'rec' + base_rec + '.' + metadata.rec_name + '.lfp.dat')
            n_chan = len(metadata.stream_channels)
            n_samp = os.stat(lfp_file).st_size // 4 // n_chan
            array = np.memmap(lfp_file, shape=(n_samp, n_chan), dtype='f', mode='r')
            self.input_file = lfp_file
            samp_rate = 1000.0
        else:
            raise NotImplementedError('You can (so far) only use the LFP file.')
        super(PesaranDataSource, self).__init__(array,
                                                metadata.cm,
                                                samp_rate,
                                                is_transpose=True,
                                                **kwargs)


    def _initialize_output(self, file_mod='clean'):
        input_parts = os.path.splitext(self.input_file)
        self.output_file = input_parts[0] + '.' + file_mod + input_parts[1]
        mm_shape = self.array.shape
        mm_dtype = self.array.dtype
        self._write_array = np.memmap(self.output_file, dtype=mm_dtype, shape=mm_shape, mode='w+')


    def write_parameters(self, **params):
        from json import dump
        params_file = os.path.splitext(self.output_file)[0] + '.params.txt'
        with open(params_file, 'w') as fp:
            dump(params, fp, indent=2, separators=(',', ': '))


class OpenEphysHDFSource(DataSource):
    """For data packed in HDF5 format from ecogdata/ecogana packages."""

    def __init__(self, hdf_file, electrode_name='psv_244_intan', **kwargs):
        h5 = h5py.File(hdf_file, 'r')
        self.input_file = hdf_file
        samp_rate = h5['Fs'].value
        array = h5['chdata']
        chan_map, disconnected = get_electrode_map(electrode_name)[:2]
        self._array_channels = np.setdiff1d(np.arange(array.shape[0]), disconnected)
        super(OpenEphysHDFSource, self).__init__(array, chan_map, samp_rate, **kwargs)


    def _initialize_output(self, file_mod='clean'):
        input_parts = os.path.splitext(self.input_file)
        self.output_file = input_parts[0] + '.' + file_mod + input_parts[1]
        write_file = h5py.File(self.output_file, 'w')
        h5_file = self.array.file
        for k in h5_file.keys():
            if k == 'chdata':
                continue
            h5_file.copy(k, write_file)
        write_file.create_dataset('chdata', shape=self.array.shape, dtype=self.array.dtype, chunks=True)
        self._write_array = write_file['chdata']


    def _output_channel_subset(self, array_block):
        channel_data = array_block[self._array_channels]
        return super(OpenEphysHDFSource, self)._output_channel_subset(channel_data)


    def _full_channel_set(self, array_block):
        full_channels = super(OpenEphysHDFSource, self)._full_channel_set(array_block)
        n_data_chan = self.array.shape[0]
        data_channels = np.zeros((n_data_chan, full_channels.shape[1]))
        data_channels[self._array_channels] = full_channels
        return data_channels


    def write_parameters(self, **params):
        # assume output file is open for business
        h5_file = self._write_array.file
        g = h5_file.create_group('structure_denoise_parameters')
        for k, v in params.items():
            g.create_dataset(k, data=v, shape=())



        
