import numpy as np
import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.data import get_sphere
from dipy.direction.peaks import peaks_from_model
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.tracking.streamline import Streamlines
from nibabel.orientations import aff2axcodes
from dipy.tracking import utils
from dipy.tracking.utils import length


def compute_dti_det_tracking(dwi, bval, bvec, mask, fa, trk):

    b0_threshold = 0.0
    bvecs_tol = 0.01
    dwi_data = nib.load(dwi).get_data()
    dwi_mask = nib.load(mask).get_data()
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold, atol=bvecs_tol)
    tenmodel = TensorModel(gtab, fit_method="WLS")

    peaks_sphere = get_sphere('repulsion724')
    peaks_dti = peaks_from_model(model=tenmodel, \
                                 data=dwi_data, \
                                 sphere=peaks_sphere, \
                                 min_separation_angle=0.25, \
                                 relative_peak_threshold=0.5, \
                                 mask=dwi_mask)

    fa_img = nib.load(fa)

    stop = fa_img.get_data()
    stopping_thr = 0.15 #0.25
    classifier = ThresholdTissueClassifier(stop, stopping_thr)

    #seed_threshold = 0.2
    #fa_data = fa_img.get_data()
    #seed_mask = np.zeros(fa_data.shape)
    #seed_bool = fa_data > seed_threshold
    #seed_mask = seed_bool.astype(np.int8)
    seed_mask = nib.load(mask).get_data()
    seed_density = 2
    seeds = utils.seeds_from_mask( \
        seed_mask, \
        density=[seed_density, seed_density, seed_density], \
        affine=fa_img.affine)

    streamlines_gen = LocalTracking(peaks_dti, \
                                    classifier, \
                                    seeds, \
                                    fa_img.affine, \
                                    step_size=1.0)
    streamlines = Streamlines(streamlines_gen)

    min_length = 20.
    max_length = 250.
    streamlines_clean = []
    for s_src in streamlines:
        s_length = list(length([s_src]))[0]
        if (s_length > min_length) and (s_length < max_length):
            streamlines_clean.append(s_src)

    header = nib.streamlines.trk.TrkFile.create_empty_header()
    header['voxel_to_rasmm'] = fa_img.affine.copy()
    header['voxel_sizes'] = fa_img.header.get_zooms()[:3]
    header['dimensions'] = fa_img.shape[:3]
    header['voxel_order'] = "".join(aff2axcodes(fa_img.affine))

    tractogram = nib.streamlines.Tractogram(streamlines_clean,affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, trk, header=header)

