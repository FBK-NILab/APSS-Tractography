# -*- coding: utf-8 -*-

"""
Module implementing the main sequence of pipeline analysis to process
diffusion MRI designed according to the requirements from APSS

Copyright (c) 2014, Fondazione Bruno Kessler
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

import os
import sys
import glob
import pickle
import numpy as np
import nibabel as nib
from subprocess import Popen, PIPE
import dipy.reconst.dti as dti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.dpy import Dpy
from dipy.data import get_sphere
from dipy.tracking.eudx import EuDX
from dipy.tracking.metrics import length
from dipy.tracking.distances import bundles_distances_mam
from dissimilarity_common import compute_dissimilarity
from parameters import *
from nibabel.orientations import aff2axcodes
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.utils import length
from compute_dti_det_tracking import compute_dti_det_tracking


def load_nifti(fname, verbose=False):
    img = nib.load(fname)
    data = img.get_data()
    affine = img.get_affine()
    if verbose:
        print('Loading...')
        print(fname)
        print(data.shape)
        print(affine)
        print(img.get_header().get_zooms()[:3])
        print(nib.aff2axcodes(affine))
        print("")
    return data, affine

def save_nifti(fname, data, affine, verbose=False):
    if verbose:
        print('Saving...')
        print(fname)
    nib.save(nib.Nifti1Image(data, affine), fname)

def pipe(cmd, print_sto=True, print_ste=True):
    """Open a pipe to a subprocess where execute an external command.
    """
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    sto = p.stdout.readlines()
    ste = p.stderr.readlines()
    if print_sto :
        print(sto)
    if print_ste :
        print(ste)

def eddy_correct(in_nii, out_nii, ref=0):
    cmd = 'eddy_correct ' + in_nii + ' ' + out_nii + ' ' + str(ref)
    pipe(cmd, print_sto=False, print_ste=False)


def bet(in_nii, out_nii, options=' -F -f .2 -g 0'):
    cmd = 'bet ' + in_nii + ' ' + out_nii + options
    pipe(cmd, print_sto=False, print_ste=False)


def dicom_to_nifti(src_dir, out_dir, subj_name, tag, opt=par_dcm2nii_options):

    if src_dir is not None and out_dir is not None:
        old_file = os.path.join(out_dir, subj_name + '_' + tag + '.nii.gz')
        if os.path.exists(old_file): os.remove(old_file)
        old_file = os.path.join(out_dir, subj_name + '.bval')
        if os.path.exists(old_file): os.remove(old_file)
        old_file = os.path.join(out_dir, subj_name + '.bvec')
        if os.path.exists(old_file): os.remove(old_file)

    try:
        glob.glob(os.path.join(src_dir, '*.dcm'))[0]
        
    except IndexError:
        print("FAIL: dcm2nii - FILE: *.dcm not found")
        sys.exit()
 
    cmd = 'dcm2niix ' + '-f ' + subj_name + ' -z y -o ' + out_dir + ' ' + src_dir
    pipe(cmd, print_sto=True, print_ste=True)
    old_file = os.path.join(out_dir, subj_name + '.nii.gz')
    new_file = os.path.join(out_dir, subj_name + '_' + tag + '.nii.gz')
    os.rename(old_file, new_file)
    old_file = os.path.join(out_dir, subj_name + '.json')
    new_file = os.path.join(out_dir, subj_name + '_' + tag + '.json')
    os.rename(old_file, new_file)
    if tag == 'mri':
        None
    if tag == 'dmri':
        None


def brain_extraction(src_bet, out_dir, subj_name, tag):

    try:
        out_bet_file = os.path.join(out_dir, subj_name + par_bet_tag + ".nii.gz")
        bet_file = os.path.join(src_bet, subj_name + '_' + tag + '.nii.gz')
        src_bet_file = os.path.join(src_bet, bet_file)
        cmd = 'bet ' + src_bet_file + ' ' + out_bet_file + par_bet_options
        pipe(cmd, print_sto=False, print_ste=False)
    except:
        print("FAIL: bet - File: %s" % src_bet_file)
        sys.exit()

        
def brain_dwi_extraction(src_bet, out_dir, subj_name, tag):

    try:
        out_bet_file = os.path.join(out_dir, subj_name + par_bet_tag + ".nii.gz")
        bet_file = os.path.join(src_bet, subj_name + '_' + tag + '.nii.gz')
        src_bet_file = os.path.join(src_bet, bet_file)
        cmd = 'bet ' + src_bet_file + ' ' + out_bet_file + par_bet4dwi_options
        pipe(cmd, print_sto=False, print_ste=False)
    except:
        print("FAIL: bet - File: %s" % src_bet_file)
        sys.exit()


def eddy_current_correction(src_ecc_dir, out_ecc_dir, subj_name):

    try:
        src_ecc_file = os.path.join(src_ecc_dir, subj_name + par_bet_tag + ".nii.gz")
        out_ecc_file = os.path.join(out_ecc_dir, subj_name + par_ecc_tag)
        if src_ecc_file is not None and out_ecc_file is not None:
            cmd = 'eddy_correct ' + src_ecc_file + ' ' + out_ecc_file + \
                  ' ' + str(par_ecc_ref)
            pipe(cmd, print_sto=False, print_ste=False)
    except:
        print("FAIL: eddy_correct - File: %s" % src_ecc_file)
        sys.exit()


def rescaling_isotropic_voxel(src_ecc_dir, out_iso_dir, subj_name):

    tmp_iso_tag = "_iso_nomask.nii.gz"
    src_dwi_tag = "_" + par_dmri_tag + ".nii.gz"
    src_ecc_file = os.path.join(src_ecc_dir, subj_name + par_ecc_tag)
    src_dwi_file = os.path.join(out_iso_dir, subj_name + src_dwi_tag)
    tmp_iso_file = os.path.join(src_ecc_dir, subj_name + tmp_iso_tag)

    if os.path.exists(src_ecc_file):
        src_img = nib.load(src_ecc_file)
    else:
        src_img = nib.load(src_dwi_file)

    src_data = src_img.get_data()
    src_affine = src_img.affine
    src_ecc_size = src_img.get_header().get_zooms()[:3]
    out_iso_size = par_iso_voxel_size
    data, affine = reslice(src_data, src_affine, src_ecc_size,out_iso_size)
    data_img = nib.Nifti1Image(data, affine)
    nib.save(data_img, tmp_iso_file)

    iso_mask_tag = "_iso_mask.nii.gz"
    bet_mask_tag = "_bet_mask.nii.gz"
    src_mask_file = os.path.join(src_ecc_dir, subj_name + bet_mask_tag)
    out_mask_file = os.path.join(out_iso_dir, subj_name + iso_mask_tag)

    if os.path.exists(src_mask_file):
        src_img = nib.load(src_mask_file)
        src_data = src_img.get_data()
        src_affine = src_img.affine
        src_mask_size = src_img.get_header().get_zooms()[:3]
        out_iso_size = par_iso_voxel_size
        data, affine = reslice(src_data, src_affine, src_mask_size, out_iso_size, mode="nearest")
        data_img = nib.Nifti1Image(data, affine)
        nib.save(data_img, out_mask_file)
        fix_wm_mask(out_mask_file, out_mask_file)

    out_iso_file = os.path.join(out_iso_dir, \
                                subj_name + par_iso_tag + ".nii.gz")
    fslmaths_cmd = "fslmaths " + tmp_iso_file \
                   + " -mas " + out_mask_file + " " + out_iso_file
    pipe(fslmaths_cmd)

    src_bet_dir = os.path.join(src_ecc_dir, "../../Structural/Preprocess")
    src_bet_file = os.path.join(src_bet_dir, \
                                subj_name + par_bet_tag + ".nii.gz")
    out_bet_file = os.path.join(src_bet_dir, \
                                subj_name + par_bet_tag + par_iso_tag + ".nii.gz")

    if os.path.exists(src_bet_file):
        src_img = nib.load(src_bet_file)
        src_data = src_img.get_data()
        src_affine = src_img.get_affine()
        src_bet_size = src_img.get_header().get_zooms()[:3]
        out_iso_size = par_iso_voxel_size
        data, affine = reslice(src_data, src_affine, src_bet_size, out_iso_size)
        data_img = nib.Nifti1Image(data, affine)
        nib.save(data_img, out_bet_file)

    
        

def flirt_registration(src_flirt_dir, ref_flirt_dir, out_flirt_dir, aff_flirt_dir, subj_name):

    src_flirt_file = os.path.join(src_flirt_dir, subj_name + par_bet_tag + par_iso_tag + ".nii.gz")
    ref_flirt_file = os.path.join(ref_flirt_dir, subj_name + par_iso_tag + ".nii.gz")
    out_flirt_file = os.path.join(out_flirt_dir, subj_name + par_flirt_tag)
    aff_flirt_file = os.path.join(aff_flirt_dir, subj_name + par_aff_tag)
    pipe('flirt -in ' + src_flirt_file + ' -ref '+ ref_flirt_file +' -out '+ out_flirt_file +' -omat '+ aff_flirt_file +' -dof '+ str(par_flirt_dof))


def atlas_registration(ref_flirt_dir, out_flirt_dir, aff_flirt_dir, subj_name):
    
    try:
        fsl_dir = os.environ['FSLDIR']
        fsl_atlas_file = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(fsl_dir, followlinks=True)
            for f in files if f.endswith(par_atlas_file)][0]
    except IndexError:
        print("FAIL: atlas file not found - File: %s" % par_atlas_file)
        sys.exit()

    ref_flirt_file = os.path.join(ref_flirt_dir, subj_name + par_iso_tag + ".nii.gz")
    out_flirt_file = os.path.join(out_flirt_dir, subj_name + par_atlas_tag)
    aff_flirt_file = os.path.join(aff_flirt_dir, subj_name + par_aff_tag)
    pipe('flirt -in ' + fsl_atlas_file + ' -ref '+ ref_flirt_file +' -out '+ out_flirt_file +' -omat '+ aff_flirt_file + ' ' + par_flirt_opt)


def fix_wm_mask(in_nii, out_nii):
    cmd = 'maskfilter -force -npass 3 ' + in_nii + ' erode ' + out_nii
    pipe(cmd, print_sto=False, print_ste=False)


def decfa(img_orig, scale=False):
    """
    Create a nifti-compliant directional-encoded color FA image.
    Parameters
    ----------
    img_orig : Nifti1Image class instance.
        Contains encoding of the DEC FA image with a 4D volume of data, where
        the elements on the last dimension represent R, G and B components.
    scale: bool.
        Whether to scale the incoming data from the 0-1 to the 0-255 range
        expected in the output.
    Returns
    -------
    img : Nifti1Image class instance with dtype set to store tuples of
        uint8 in (R, G, B) order.
    Notes
    -----
    For a description of this format, see:
    https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html
    """

    dest_dtype = np.dtype([('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')])
    out_data = np.zeros(img_orig.shape[:3], dtype=dest_dtype)

    data_orig = img_orig.get_data()

    if scale:
        data_orig = (data_orig * 255).astype('uint8')

    for ii in np.ndindex(img_orig.shape[:3]):
        val = data_orig[ii]
        out_data[ii] = (val[0], val[1], val[2])

    new_hdr = img_orig.header
    new_hdr['dim'][4] = 1
    new_hdr.set_intent(1001, name='Color FA')
    new_hdr.set_data_dtype(dest_dtype)
    
    return nib.Nifti1Image(out_data, affine=img_orig.affine, header=new_hdr)


def compute_reconstruction(src_dmri_dir, subj_name):

    dwi =  os.path.join(src_dmri_dir, subj_name + par_iso_tag + ".nii.gz")
    bval =  os.path.join(src_dmri_dir, subj_name + ".bval")
    bvec =  os.path.join(src_dmri_dir, subj_name + ".bvec")
    mask =  os.path.join(src_dmri_dir, subj_name + "_iso_mask.nii.gz")
    seed =  os.path.join(src_dmri_dir, subj_name + "_seed.nii.gz")
    fa = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    md =  os.path.join(src_dmri_dir, subj_name + "_MD.nii.gz")
    rgb =  os.path.join(src_dmri_dir, subj_name + "_RGB.nii.gz")
    cfa =  os.path.join(src_dmri_dir, subj_name + "_CFA.nii.gz")
    
    cmd = "dipy_fit_dti --force --save_metrics fa md rgb --out_fa %s --out_md %s --out_rgb %s %s %s %s %s" % (fa, md, rgb, dwi, bval, bvec, mask)
    pipe(cmd, print_sto=True, print_ste=True)

    FA = nib.load(fa).get_data()
    MD = nib.load(md).get_data()
    affine = nib.load(fa).affine
    #WM = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
    WM = FA >= 0.15
    WM = WM.astype('uint16')
    img_seed = nib.Nifti1Image(WM, affine)
    nib.save(img_seed, seed)

    RGB = nib.load(rgb)
    fix_rgb = decfa(RGB)
    nib.save(fix_rgb, cfa) 


def compute_dti_tracking(src_dmri_dir, out_trk_dir, subj_name):

    dwi =  os.path.join(src_dmri_dir, subj_name + par_iso_tag + ".nii.gz")
    bval =  os.path.join(src_dmri_dir, subj_name + ".bval")
    bvec =  os.path.join(src_dmri_dir, subj_name + ".bvec")
    mask =  os.path.join(src_dmri_dir, subj_name + "_seed.nii.gz")
    fa = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    trk = os.path.join(out_trk_dir, subj_name + "_dti_det.trk")

    compute_dti_det_tracking(dwi, bval, bvec, mask, fa, trk)


def compute_csd(src_dmri_dir, out_dmri_dir, subj_name):

    dwi =  os.path.join(src_dmri_dir, subj_name + par_iso_tag + ".nii.gz")
    bval =  os.path.join(src_dmri_dir, subj_name + ".bval")
    bvec =  os.path.join(src_dmri_dir, subj_name + ".bvec")
    mask =  os.path.join(src_dmri_dir, subj_name + "_iso_mask.nii.gz")
    fa = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    pam =  os.path.join(src_dmri_dir, subj_name + "_csd.pam5")

    cmd = 'dipy_fit_csd --force --out_pam %s %s %s %s %s' % \
          (pam, dwi, bval, bvec, mask)
    pipe(cmd, print_sto=True, print_ste=True)



def compute_csd_det_tracking(src_dmri_dir, out_trk_dir, subj_name):

    seed =  os.path.join(src_dmri_dir, subj_name + "_seed.nii.gz")
    md = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    fa = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    pam =  os.path.join(src_dmri_dir, subj_name + "_csd.pam5")
    trk = os.path.join(out_trk_dir, subj_name + "_csd_det.trk")

    cmd = 'dipy_track_local --force %s %s %s --tracking_method det --seed_density 1 --out_tractogram %s' % \
          (pam, seed, seed, trk)
    pipe(cmd, print_sto=False, print_ste=False)

    img = nib.load(fa)
    header = nib.streamlines.trk.TrkFile.create_empty_header()
    header['voxel_to_rasmm'] = img.affine.copy()
    header['voxel_sizes'] = img.header.get_zooms()[:3]
    header['dimensions'] = img.shape[:3]
    header['voxel_order'] = "".join(aff2axcodes(img.affine))

    tract = nib.streamlines.load(trk)
    min_length = 20.
    max_length = 250.
    streamlines_clean = []
    for s_src in tract.streamlines:
        s_length = list(length([s_src]))[0]
        if (s_length > min_length) and (s_length < max_length):
            streamlines_clean.append(s_src)

    fix_streamlines = transform_streamlines(streamlines_clean, \
                                            np.linalg.inv(img.affine))
    tractogram = nib.streamlines.Tractogram(fix_streamlines, \
                                            affine_to_rasmm=img.affine)
    nib.streamlines.save(tractogram, trk, header=header)


def compute_csd_prob_tracking(src_dmri_dir, out_trk_dir, subj_name):

    seed =  os.path.join(src_dmri_dir, subj_name + "_seed.nii.gz")
    fa = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    pam =  os.path.join(src_dmri_dir, subj_name + "_csd.pam5")
    trk = os.path.join(out_trk_dir, subj_name + "_csd_prob.trk")

    cmd = 'dipy_track_local --force %s %s %s --tracking_method prob --seed_density 1 --out_tractogram %s' % \
          (pam, fa, seed, trk)
    pipe(cmd, print_sto=False, print_ste=False)

    img = nib.load(fa)
    header = nib.streamlines.trk.TrkFile.create_empty_header()
    header['voxel_to_rasmm'] = img.affine.copy()
    header['voxel_sizes'] = img.header.get_zooms()[:3]
    header['dimensions'] = img.shape[:3]
    header['voxel_order'] = "".join(aff2axcodes(img.affine))

    tract = nib.streamlines.load(trk)
    min_length = 20.
    max_length = 250.
    streamlines_clean = []
    for s_src in tract.streamlines:
        s_length = list(length([s_src]))[0]
        if (s_length > min_length) and (s_length < max_length):
            streamlines_clean.append(s_src)

    fix_streamlines = transform_streamlines(streamlines_clean, \
                                            np.linalg.inv(img.affine))
    tractogram = nib.streamlines.Tractogram(fix_streamlines, \
                                            affine_to_rasmm=img.affine)
    nib.streamlines.save(tractogram, trk, header=header)


def tractome_preprocessing(src_trk_dir, subj_name):

    trk1 = subj_name + "_dti_det.trk"
    trk2 = subj_name + "_csd_det.trk"
    trk3 = subj_name + "_csd_prob.trk"
    
    for trk in (trk1, trk2, trk3):
        src_trk_file = os.path.join(src_trk_dir, trk)
        if os.path.exists(src_trk_file):
            spa_basename = os.path.splitext(trk_basename)[0] + '.spa'
            src_trk_file = os.path.join(src_trk_dir, trk_basename)
            out_spa_dir = os.path.join(src_trk_dir, '.temp')
            if not os.path.exists(out_spa_dir):
                os.makedirs(out_spa_dir)
            out_spa_file = os.path.join(out_spa_dir, spa_basename)

            tract = nib.streamlines.load(src_trk_file)
            dissimilarity_matrix = compute_dissimilarity(tract.streamlines,  prototype_distance, par_prototype_policy, par_prototype_num)
            
            info = {'dismatrix':dissimilarity_matrix, \
                    'nprot':par_prototype_num}
            pickle.dump(info, open(out_spa_file,'w+'), \
                        protocol=pickle.HIGHEST_PROTOCOL)


def roi_registration(src_ref_dir, out_roi_dir, subj_name):
    
    # Compute the affine from atlas
    # (reference image: the isotropic dwi file)
    src_atlas = os.path.join(par_atlas_dir, par_roi_atlas)
    if not os.path.exists(src_atlas):
        print('Atlas not found: %s' % src_atlas)
        return

    src_ref = os.path.join(src_ref_dir, subj_name + par_iso_tag + ".nii.gz")
    src_aff = os.path.join(src_ref_dir, par_roi_aff)
    cmd = 'flirt -in %s -ref %s -omat %s %s' % (src_atlas, src_ref, src_aff, par_flirt_opt)
    pipe(cmd, print_sto=False, print_ste=False)

    # Apply the affine to all ROIs
    if not os.path.exists(par_roi_dir):
        print('Pathname for ROI not found: %s' % par_roi_dir)
        return

    all_roi = [f for f in os.listdir(par_roi_dir) if f.endswith('.nii')]
    for roi in all_roi:
        src_roi = os.path.join(par_roi_dir, roi)
        out_roi = os.path.join(out_roi_dir, roi)
        cmd = 'flirt -in %s -ref %s -applyxfm -init %s -out %s -interp nearestneighbour' % (src_roi, src_ref, src_aff, out_roi)
        pipe(cmd, print_sto=False, print_ste=False)
