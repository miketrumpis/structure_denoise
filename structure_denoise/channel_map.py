import itertools
import copy
from enum import Enum
import numpy as np
import scipy.misc as spmisc
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


__all__ = ['Bunch', 'ChannelMap', 'get_electrode_map']


NonSignalChannels = Enum('NonSignalChannels', ['grounded', 'reference', 'other'])
GND = NonSignalChannels.grounded
REF = NonSignalChannels.reference
OTHER = NonSignalChannels.other


def map_intersection(maps):
    geometries = set([m.geometry for m in maps])
    if len(geometries) > 1:
        raise ValueError('cannot intersect maps with different geometries')
    bin_map = maps[0].embed(np.ones(len(maps[0])), fill=0)
    for m in maps[1:]:
        bin_map *= m.embed(np.ones(len(m)), fill=0)
    return bin_map.astype('?')


# ye olde Bunch object
class Bunch(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
        self.__dict__ = self

    def __repr__(self):
        k_rep = self.keys()
        if not len(k_rep):
            return 'an empty Bunch'
        v_rep = [str(type(self[k])) for k in k_rep]
        mx_c1 = max([len(s) for s in k_rep])
        mx_c2 = max([len(s) for s in v_rep])
        
        table = ['{0:<{col1}} : {1:<{col2}}'.format(k, v, col1=mx_c1, col2=mx_c2)
                 for (k, v) in zip(k_rep, v_rep)]
        
        table = '\n'.join(table)
        return table.strip()

    def __copy__(self):
        d = dict([ (k, copy.copy(v)) for k, v in self.items() ])
        return Bunch(**d)
    
    def copy(self):
        return copy.copy(self)

    def __deepcopy__(self, memo):
        d = dict([ (k, copy.deepcopy(v)) for k, v in self.items() ])
        return Bunch(**d)
    
    def deepcopy(self):
        return copy.deepcopy(self)


### Matrix-indexing manipulations
def flat_to_mat(mn, idx, col_major=True):
    idx = np.asarray(idx)
    # convert a flat matrix index into (i,j) style
    (m, n) = mn if col_major else mn[::-1]
    if (idx < 0).any() or (idx >= m*n).any():
        raise ValueError(
            'The flat index does not lie inside the matrix: '+str(mn)
            )
    j = idx // m
    i = (idx - j*m)
    return (i, j) if col_major else (j, i)


def mat_to_flat(mn, i, j, col_major=True):
    i, j = map(np.asarray, (i, j))
    if (i < 0).any() or (i >= mn[0]).any() \
      or (j < 0).any() or (j >= mn[1]).any():
        raise ValueError('The matrix index does not fit the geometry: '+str(mn))
    (i, j) = map(np.asarray, (i, j))
    # covert matrix indexing to a flat (linear) indexing
    (fast, slow) = (i, j) if col_major else (j, i)
    block = mn[0] if col_major else mn[1]
    idx = slow*block + fast
    return idx


def flat_to_flat(mn, idx, col_major=True):
    # convert flat indexing from one convention to another
    i, j = flat_to_mat(mn, idx, col_major=col_major)
    return mat_to_flat(mn, i, j, col_major=not col_major)


class ChannelMap(list):
    "A map of sample vector(s) to a matrix representing 2D sampling space."
    
    def __init__(self, chan_map, geo, col_major=True, pitch=1.0):
        list.__init__(self)
        self[:] = chan_map
        self.col_major = col_major
        self.geometry = geo
        self.pitch = pitch
        self._combs = None


    @staticmethod
    def from_index(ij, shape, col_major=True, pitch=1.0):
        """Return a ChannelMap from a list of matrix index pairs (e.g. [(0, 3), (2, 1), ...])
        and a matrix shape (e.g. (5, 5)).
        """

        i, j = zip(*ij)
        map = mat_to_flat(shape, i, j, col_major=col_major)
        return ChannelMap(map, shape, col_major=col_major, pitch=pitch)


    @staticmethod
    def from_mask(mask, col_major=True, pitch=1.0):
        """Create a ChannelMap from a binary grid. Note: the data channels must be aligned
        with the column-major or row-major raster order of this binary mask
        """

        i, j = mask.nonzero()
        ij = zip(i, j)
        geo = mask.shape
        return ChannelMap.from_index(ij, geo, col_major=col_major, pitch=pitch)


    @property
    def site_combinations(self):
        if self._combs is None:
            self._combs = channel_combinations(self, scale=self.pitch)
        return self._combs
    
    def as_row_major(self):
        if self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:]),
                self.geometry, col_major=False, pitch=self.pitch
                )
        return self

    def as_col_major(self):
        if not self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:], col_major=False),
                self.geometry, col_major=True, pitch=self.pitch
                )
        return self

    def to_mat(self):
        return flat_to_mat(self.geometry, self, col_major=self.col_major)

    def lookup(self, i, j):
        flat_idx = mat_to_flat(self.geometry, i, j, col_major=self.col_major)
        if np.iterable(flat_idx):
            return np.array([self.index(fi) for fi in flat_idx])
        return self.index(flat_idx)

    def rlookup(self, c):
        return flat_to_mat(self.geometry, self[c], col_major=self.col_major)

    def subset(self, sub, as_mask=False, map_intersect=False):
        """
        Behavior depends on the type of "sub":

        Most commonly, sub is a sequence (list, tuple, array) of subset
        indices.
        
        ChannelMap: return the subset map for the intersecting sites

        ndarray: if NOT subset indices (i.e. a binary mask), then the
        mask is converted to indices. If the array is a 2D binary mask,
        then site-lookup is used.
        
        """
        if isinstance(sub, type(self)):
            # check that it's a submap
            submap = map_intersection([self, sub])
            if submap.sum() < len(sub):
                raise ValueError(
                    'The given channel map is not a subset of this map'
                    )
            # get the channels/indices of the subset of sites
            sub = self.lookup(*submap.nonzero())
        elif isinstance(sub, np.ndarray):
            if sub.ndim==2:
                # Get the channels/indices of the subset of sites.
                # This needs to be sorted to look up the subset of
                # channels in sequence
                if map_intersect:
                    # allow 2d binary map to cover missing sites
                    ii, jj = sub.nonzero()
                    sites = []
                    for i, j in zip(ii, jj):
                        try:
                            sites.append(self.lookup(i, j))
                        except ValueError:
                            pass
                else:
                    # if this looks up missing sites, then raise
                    sites = self.lookup(*sub.nonzero())
                sub = np.sort( sites )
            elif sub.ndim==1:
                if sub.dtype.kind in ('b',):
                    sub = sub.nonzero()[0]
            else:
                raise ValueError('Cannot interpret subset array')
        elif not isinstance(sub, (list, tuple)):
            raise ValueError('Unknown subset type')

        if as_mask:
            mask = np.zeros( (len(self),), dtype='?' )
            mask[sub] = True
            return mask

        cls = type(self)
        return cls(
            [self[i] for i in sub], self.geometry,
            col_major=self.col_major, pitch=self.pitch
            )

    def __getslice__(self, i, j):
        cls = type(self)
        new_map = cls(
            super(ChannelMap, self).__getslice__(i,j),
            self.geometry, col_major=self.col_major, pitch=self.pitch
            )
        import sys
        # Keep the pre-computed combinations IFF the entire map is copied
        if i==0 and j==sys.maxint and self._combs is not None:
            new_map._combs = self._combs.copy()
        return new_map

    def embed(self, data, axis=0, fill=np.nan):
        """
        Embed the data in electrode array geometry, mapping channels
        on the given axis
        """
        data = np.atleast_1d(data)
        shape = list(data.shape)
        if shape[axis] != len(self):
            raise ValueError(
                'Data array does not have the correct number of channels'
                )
        shape.pop(axis)
        shape.insert(axis, self.geometry[0]*self.geometry[1])
        array = np.empty(shape, dtype=data.dtype)
        if not isinstance(fill, str):
            array.fill(fill)
        slicing = [slice(None)] * len(shape)
        slicing[axis] = self.as_row_major()[:]
        array[tuple(slicing)] = data
        shape.pop(axis)
        shape.insert(axis, self.geometry[1])
        shape.insert(axis, self.geometry[0])
        array.shape = shape
        if isinstance(fill, str):
            return self.interpolated(array, axis=axis)
        return array

    def as_channels(self, matrix, axis=0):
        """
        Take the elements of a matrix into the "natural" channel ordering.
        """
        m_shape = matrix.shape
        m_flat = m_shape[axis] * m_shape[axis+1]
        c_dims = m_shape[:axis] + (m_flat,) + m_shape[axis+2:]
        matrix = matrix.reshape(c_dims)
        return np.take(matrix, self, axis=axis)

    def inpainted(self, image, axis=0, **kwargs):
        pass

    def interpolated(self, image, axis=0, method='median'):
        # acts in-place
        mask = self.embed(np.zeros(len(self), dtype='?'), fill=1)
        missing = np.where( mask )
        g = self.geometry
        def _slice(i, j, w):
            before = [slice(None)] * axis
            after = [slice(None)] * (image.ndim - axis - 2)
            if w:
                isl = slice( max(0, i-w), min(g[0], i+w+1) )
                jsl = slice( max(0, j-w), min(g[1], j+w+1) )
            else:
                isl = i; jsl = j
            before.extend( [isl, jsl] )
            before.extend( after )
            return tuple(before)

        # first pass, tag all missing sites with nan
        for i, j in zip(*missing):
            image[ _slice(i, j, 0) ] = np.nan
        for i, j in zip(*missing):
            # do a +/- 2 neighborhoods (8 neighbors)
            patch = image[ _slice(i, j, 1) ].copy()
            s = list( patch.shape )
            s = s[:axis] + [ s[axis]*s[axis+1] ] + s[axis+2:]
            patch.shape = s
            fill = np.nanmedian( patch, axis=axis )
            image[ _slice(i, j, 0) ] = fill
        return image

    def image(
            self, arr=None, cbar=True, nan='//',
            fill=np.nan, ax=None, **kwargs
            ):
        kwargs.setdefault('origin', 'upper')
        if ax is None:
            import matplotlib.pyplot as pp
            f = pp.figure()
            ax = pp.subplot(111)
        else:
            f = ax.figure

        if arr is None:
            # image self
            arr = self.embed( np.ones(len(self), 'd'), fill=fill )
            kwargs.setdefault('clim', (0, 1))
            kwargs.setdefault('norm', BoundaryNorm([0, 0.5, 1], 256))
            kwargs.setdefault('cmap', cm.binary)
            
        if arr.shape != self.geometry:
            arr = self.embed(arr, fill=fill)

        nans = zip(*np.isnan(arr).nonzero())
        im = ax.imshow(arr, **kwargs)
        ext = kwargs.pop('extent', ax.get_xlim() + ax.get_ylim())
        dx = abs(float(ext[1] - ext[0])) / arr.shape[1]
        dy = abs(float(ext[3] - ext[2])) / arr.shape[0]
        x0 = min(ext[:2]); y0 = min(ext[2:])
        def s(x):
            return (x[0] * dy + y0, x[1] * dx + x0)
        if len(nan):
            for x in nans:
                r = Rectangle( s(x)[::-1], dx, dy, hatch=nan, fill=False )
                ax.add_patch(r)
        #ax.set_ylim(ext[2:][::-1])
        if cbar:
            cb = f.colorbar(im, ax=ax, use_gridspec=True)
            cb.solids.set_edgecolor('face')
            return f, cb
        return f
    

def channel_combinations(chan_map, scale=1.0, precision=4):
    """Compute tables identifying channel-channel pairs.

    Parameters
    ----------
    chan_map : ChannelMap
    scale : float or pair
        The constant pitch or the (dx, dy) pitch between electrodes
        precision : number of decimals for distance calculation (it seems
        some distances are not uniquely determined in floating point).

    Returns
    -------
    chan_combs : Bunch
        Lists of channel # and grid location of electrode pairs and
        distance between each pair.
    """
    
    combs = itertools.combinations(np.arange(len(chan_map)), 2)
    chan_combs = Bunch()
    npair = spmisc.comb(len(chan_map),2,exact=1)
    chan_combs.p1 = np.empty(npair, 'i')
    chan_combs.p2 = np.empty(npair, 'i')
    chan_combs.idx1 = np.empty((npair, 2), 'd')
    chan_combs.idx2 = np.empty((npair, 2), 'd')
    chan_combs.dist = np.empty(npair)
    ii, jj = chan_map.to_mat()
    # Distances are measured between grid locations (i1,j1) to (i2,j2)
    # Define a (s1,s2) scaling to multiply these distances
    if np.iterable(scale):
        s_ = np.array( scale[::-1] )
    else:
        s_ = np.array( [scale, scale] )
    for n, c in enumerate(combs):
        c0, c1 = c
        chan_combs.p1[n] = c0
        chan_combs.p2[n] = c1
        chan_combs.idx1[n] = ii[c0], jj[c0]
        chan_combs.idx2[n] = ii[c1], jj[c1]

    d = np.abs( chan_combs.idx1 - chan_combs.idx2 ) * s_
    dist = ( d**2 ).sum(1) ** 0.5
    chan_combs.dist = np.round(dist, decimals=precision)
    idx1 = chan_combs.idx1.astype('i')
    if (idx1 == chan_combs.idx1).all():
        chan_combs.idx1 = idx1
    idx2 = chan_combs.idx2.astype('i')
    if (idx2 == chan_combs.idx2).all():
        chan_combs.idx2 = idx2
    return chan_combs


## Define the open-ephys + intan channel map

def _rev(n, coords):
    return [ c if c in NonSignalChannels else (n - c - 1) for c in coords ]


_psv_244_intan = dict(
    geometry = (16, 16),
    
    pitch = 0.75,
    
    rows = [3, GND, 5, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3, 1, 5, 1, 6, 3, 4, 1, 
            0, 2, 2, 0, 1, 2, 3, 0, 5, 0, 4, GND, 0, 1, 2, 1, 1, 2, 3, 0, 
            5, 6, 4, 7, 0, 3, 2, 1, 2, 0, 1, 2, 3, 1, 5, 3, 4, 6, 0, 4, 2,
            0, 1, GND, 8, GND, 8, 5, 7, 5, 7, 5, 7, 4, 6, 4, 6, 3, 6, 2, 7, 
            5, 6, 5, 6, 4, 6, 4, 7, 3, 7, 3, 7, 2, 8, GND, 8, 14, 8, 13, 9, 
            12, 9, 12, 9, 9, 10, 8, 10, 11, 10, 11, 11, 13, 10, 13, 10, 12, 
            10, 12, 9, 8, 9, 11, 9, 11, 8, GND, 12, GND, 10, 11, 11, 15, 15, 
            13, 13, 14, 14, 12, 12, 14, 10, 14, 9, 12, 11, 14, 15, 13, 13, 
            15, 14, 13, 12, 15, 10, 15, 11, GND, 15, 14, 13, 14, 14, 13, 12, 
            15, 10, 9, 11, 8, 15, 12, 13, 14, 13, 15, 14, 13, 12, 14, 10, 
            12, 11, 9, 15, 11, 13, 15, 14, GND, 7, GND, 7, 10, 8, 10, 8, 10,
            8, 11, 9, 11, 9, 12, 9, 13, 8, 10, 9, 10, 9, 11, 9, 11, 8, 12, 
            8, 12, 8, 13, 7, GND, 7, 1, 7, 2, 6, 3, 6, 3, 6, 6, 5, 7, 5, 4, 
            5, 4, 4, 2, 5, 2, 5, 3, 5, 3, 6, 7, 6, 4, 6, 4, 7, GND],

    cols = _rev(16, [7, GND, 7, 10, 8, 10, 8, 10, 8, 11, 9, 11, 9, 12, 9, 
                     13, 8, 10, 9, 10, 9, 11, 9, 11, 8, 12, 8, 12, 8, 13, 
                     7, GND, 7, 1, 7, 2, 6, 3, 6, 3, 6, 6, 5, 7, 5, 4, 5, 4, 
                     4, 2, 5, 2, 5, 3, 5, 3, 6, 7, 6, 4, 6, 4, 7, GND, 3, GND, 
                     5, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3, 1, 5, 1, 6, 3, 4, 1, 
                     0, 2, 2, 0, 1, 2, 3, 0, 5, 0, 4, GND, 0, 1, 2, 1, 1, 2, 
                     3, 0, 5, 6, 4, 7, 0, 3, 2, 1, 2, 0, 1, 2, 3, 1, 5, 3, 
                     4, 6, 0, 4, 2, 0, 1, GND, 8, GND, 8, 5, 7, 5, 7, 5, 7, 4, 
                     6, 4, 6, 3, 6, 2, 7, 5, 6, 5, 6, 4, 6, 4, 7, 3, 7, 3, 7, 
                     2, 8, GND, 8, 14, 8, 13, 9, 12, 9, 12, 9, 9, 10, 8, 10, 
                     11, 10, 11, 11, 13, 10, 13, 10, 12, 10, 12, 9, 8, 9, 11, 
                     9, 11, 8, GND, 12, GND, 10, 11, 11, 15, 15, 13, 13, 14, 
                     14, 12, 12, 14, 10, 14, 9, 12, 11, 14, 15, 13, 13, 15, 
                     14, 13, 12, 15, 10, 15, 11, GND, 15, 14, 13, 14, 14, 13, 
                     12, 15, 10, 9, 11, 8, 15, 12, 13, 14, 13, 15, 14, 13, 
                     12, 14, 10, 12, 11, 9, 15, 11, 13, 15, 14, GND])
    )


map_specs = {'psv_244_intan' : _psv_244_intan}


def get_electrode_map(name):
    try:
        pinouts = map_specs[name]
    except KeyError:
        raise ValueError('electrode name not found: '+name)
    
    rows = pinouts['rows']
    cols = pinouts['cols']
    sig_rows = []
    sig_cols = []
    no_connection = []
    reference = []
    for n in range(len(rows)):
        if rows[n] is REF:
            reference.append(n)
        elif rows[n] in NonSignalChannels:
            no_connection.append(n)
        else:
            sig_rows.append(rows[n])
            sig_cols.append(cols[n])

    geometry = pinouts['geometry']
    pitch = pinouts.get('pitch', 1.0)
    chan_map = ChannelMap.from_index(zip(sig_rows, sig_cols), geometry, pitch=pitch, col_major=False)
    return chan_map, no_connection, reference
