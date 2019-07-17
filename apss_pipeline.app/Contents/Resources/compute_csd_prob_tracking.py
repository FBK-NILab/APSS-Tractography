import numpy as np
import nibabel as nib
from dipy.direction import ProbabilisticDirectionGetter
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.io.peaks import load_peaks
from nibabel.orientations import aff2axcodes
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import length


def compute_csd_prob_tractogram(src_pam, src_fa, src_seed, out_trk):

    pmf_threshold = 0.1
    max_angle = 30
    tracking_method = "det"
    stopping_thr = 0.15 #0.25
    step_size = 1.0
    seed_density = 2

    # Direction Getter
    pam = load_peaks(src_pam, verbose=False)
    dg = ProbabilisticDirectionGetter.from_shcoeff( \
                                                    pam.shm_coeff,
                                                    sphere=pam.sphere,
                                                    max_angle=max_angle,
                                                    pmf_threshold=pmf_threshold)

    # Tissue Classifier
    fa_img = nib.load(src_fa)
    affine = fa_img.affine
    stop = fa_img.get_data()
    classifier = ThresholdTissueClassifier(stop, stopping_thr)

    # Seeding Mask
    seed_img = nib.load(src_seed)
    seed_mask = seed_img.get_data()
    seeds = utils.seeds_from_mask( \
        seed_mask, \
        density=[seed_density, seed_density, seed_density], \
        affine=affine)

    # Tracking 
    streamlines_gen = LocalTracking(dg, classifier, seeds, affine, \
                                step_size=step_size)
    streamlines = Streamlines(streamlines_gen)

    # Cleaning Tractogram
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
    nib.streamlines.save(tractogram, out_trk, header=header)
