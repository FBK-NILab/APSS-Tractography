import os
import time
import pickle
import numpy as np
import nibabel as nib
from parameters import *
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.streamline import transform_streamlines
from dissimilarity_common import compute_dissimilarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree

def orient2rgb(v):
    """ standard orientation 2 rgb colormap

    v : array, shape (N, 3) of vectors not necessarily normalized

    Returns
    -------

    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    --------

    >>> from dipy.viz import colormap
    >>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> c = colormap.orient2rgb(v)

    """

    if v.ndim == 1:
        orient = v
        orient = np.abs(orient / np.linalg.norm(orient))

    if v.ndim == 2:
        orientn = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        orientn.shape = orientn.shape + (1,)
        orient = np.abs(v / orientn)

    return orient

def streamline2rgb(streamline):
    """Compute orientation of a streamline and retrieve and appropriate RGB
    color to represent it.
    """
    # simplest implementation:
    tmp = orient2rgb(streamline[0] - streamline[-1])
    return tmp


def compute_colors(streamlines, alpha):
    """Compute colors for a list of streamlines.
    """
    # assert(type(streamlines) == type([]))
    tot_vertices = np.sum([len(curve) for curve in streamlines])
    color = np.empty((tot_vertices,4), dtype='f4')
    counter = 0
    for curve in streamlines:
        color[counter:counter+len(curve),:3] = streamline2rgb(curve).astype('f4')
        counter += len(curve)
    color[:,3] = alpha
    return color


def compute_buffers(streamlines, alpha, save=False, filename=None):
    """Compute buffers for GL.
    """
    tmp = streamlines
    #if type(tmp) is not type([]):
    #    tmp = streamlines.tolist()
    streamlines_buffer = np.ascontiguousarray(np.concatenate(tmp).astype('f4'))
    streamlines_colors = np.ascontiguousarray(compute_colors(streamlines, alpha))
    streamlines_count = np.ascontiguousarray(np.array([len(curve) for curve in streamlines],dtype='i4'))
    streamlines_first = np.ascontiguousarray(np.concatenate([[0],np.cumsum(streamlines_count)[:-1]]).astype('i4'))
    tmp = {'buffer': streamlines_buffer,
           'colors': streamlines_colors,
           'count': streamlines_count,
           'first': streamlines_first}
    if save:
        print("saving buffers to", filename)
        np.savez_compressed(filename, **tmp)
    return tmp


def mbkm_wrapper(full_dissimilarity_matrix, n_clusters, streamlines_ids):
    """Wrapper of MBKM with API compatible to the Manipulator.

    streamlines_ids can be set or list.
    """
    sids = np.array(list(streamlines_ids))
    dissimilarity_matrix = full_dissimilarity_matrix[sids]

    print("MBKM clustering time:",)
    init = 'random'
    mbkm = MiniBatchKMeans(init=init, n_clusters=n_clusters, batch_size=1000,
                          n_init=10, max_no_improvement=5, verbose=0)
    t0 = time.time()
    mbkm.fit(dissimilarity_matrix)
    t_mini_batch = time.time() - t0
    print(t_mini_batch)

    print("exhaustive smarter search of the medoids:",)
    medoids_exhs = np.zeros(n_clusters, dtype=np.int)
    t0 = time.time()
    idxs = []
    for i, centroid in enumerate(mbkm.cluster_centers_):
        idx_i = np.where(mbkm.labels_==i)[0]
        if idx_i.size == 0: idx_i = [0]
        tmp = full_dissimilarity_matrix[idx_i] - centroid
        medoids_exhs[i] = sids[idx_i[(tmp * tmp).sum(1).argmin()]]
        idxs.append(set(sids[idx_i].tolist()))
        
    t_exhs_query = time.time() - t0
    print(t_exhs_query, "sec")
    clusters = dict(zip(medoids_exhs, idxs))
    return clusters


def tractome_preprocessing(src_trk_file):
    
    src_trk_dir = os.path.dirname(src_trk_file)
    src_trk_base = os.path.basename(src_trk_file)
    spa_basename = os.path.splitext(src_trk_base)[0] + '.spa'
    out_spa_dir = os.path.join(src_trk_dir, '.temp')
    if not os.path.exists(out_spa_dir):
        os.makedirs(out_spa_dir)
    out_spa_file = os.path.join(out_spa_dir, spa_basename)
    
    tract = nib.streamlines.load(src_trk_file)
    
    # Compute Dissimilarity Representation
    distance = eval(par_prototype_distance)
    dissimilarity_matrix = compute_dissimilarity(tract.streamlines, \
                                                 distance, \
                                                 par_prototype_policy, \
                                                 par_prototype_num)
    
    # Compute Buffers
    affine = tract.affine
    streams = transform_streamlines(tract.streamlines, np.linalg.inv(affine))
    buffers = compute_buffers(streams, alpha=1.0, save=False)
    
    # Compute Clusters
    size_T = len(tract.streamlines)
    n_clusters = 150
    streamlines_ids = np.arange(size_T, dtype=np.int)
    clusters = mbkm_wrapper(dissimilarity_matrix, n_clusters, streamlines_ids)

    # Compute KD-Tree
    #kdtree = KDTree(dissimilarity_matrix)
    
    # Save the SPA file
    info = {'initclusters':clusters, \
            'buff':buffers, \
            'dismatrix':dissimilarity_matrix, \
            'nprot':par_prototype_num}
            #'kdtree':kdtree}
    print("...saving\n")
    pickle_protocol = 2
    #pickle_protocol = pickle.HIGHEST_PROTOCOL
    pickle.dump(info, open(out_spa_file,'wb+'), protocol=pickle_protocol)
