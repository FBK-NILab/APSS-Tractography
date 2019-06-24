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
        print
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
        print "FAIL: bet - File: %s" % src_bet_file
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
    src_ecc_file = os.path.join(src_ecc_dir, subj_name + par_ecc_tag)
    tmp_iso_file = os.path.join(out_iso_dir, subj_name + tmp_iso_tag)

    try:
        if src_ecc_file is not None:
            src_img = nib.load(src_ecc_file)
            src_data = src_img.get_data()
            src_affine = src_img.get_affine()
        src_ecc_size = src_img.get_header().get_zooms()[:3]
        out_iso_size = par_iso_voxel_size
        data, affine = reslice(src_data, src_affine, src_ecc_size,out_iso_size)
        data_img = nib.Nifti1Image(data, affine)
        nib.save(data_img, tmp_iso_file)

    except:
        print "FAIL: isotropic rescaling of dmri - File: %s" % src_ecc_file
        exit

    tmp_mask_tag = "_iso_mask.nii.gz"
    bet_mask_tag = "_bet_mask.nii.gz"
    src_mask_dir = os.path.join(src_ecc_dir)
    src_mask_file = os.path.join(src_mask_dir, subj_name + bet_mask_tag)
    out_mask_file = os.path.join(src_mask_dir, subj_name + tmp_mask_tag)

    try:
        if src_mask_file is not None:
            src_img = nib.load(src_mask_file)
            src_data = src_img.get_data()
            src_affine = src_img.get_affine()
        src_mask_size = src_img.get_header().get_zooms()[:3]
        out_iso_size = par_iso_voxel_size
        data, affine = reslice(src_data, src_affine, src_mask_size, out_iso_size, mode="nearest")
        data_img = nib.Nifti1Image(data, affine)
        nib.save(data_img, out_mask_file)

    except:
        print("FAIL: isotropic rescaling of mask - File: %s" % src_mask_file)
        exit


    src_bet_dir = os.path.join(src_ecc_dir, "../../Structural/Preprocess")
    src_bet_file = os.path.join(src_bet_dir, subj_name + par_bet_tag + ".nii.gz")
    out_bet_file = os.path.join(src_bet_dir, subj_name + par_bet_tag + par_iso_tag + ".nii.gz")

    try:
        if src_bet_file is not None:
            src_img = nib.load(src_bet_file)
            src_data = src_img.get_data()
            src_affine = src_img.get_affine()
        src_bet_size = src_img.get_header().get_zooms()[:3]
        out_iso_size = par_iso_voxel_size
        data, affine = reslice(src_data, src_affine, src_bet_size, out_iso_size)
        data_img = nib.Nifti1Image(data, affine)
        nib.save(data_img, out_bet_file)

    except:
        print "FAIL: isotropic rescaling of structural - File: %s" % src_bet_file
        exit

    out_iso_file = os.path.join(out_iso_dir, subj_name + par_iso_tag + ".nii.gz")
    fslmaths_cmd = "fslmaths " + tmp_iso_file + " -add " + out_mask_file + " " + out_iso_file
    pipe(fslmaths_cmd)
    
        

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


def compute_reconstruction(src_dmri_dir, subj_name):

    src_dmri_file = os.path.join(src_dmri_dir, subj_name + par_iso_tag + ".nii.gz")
    src_bval_file = src_dmri_dir +  [each for each in os.listdir(src_dmri_dir) if each.endswith('.bval')][0]
    src_bvec_file = src_dmri_dir +  [each for each in os.listdir(src_dmri_dir) if each.endswith('.bvec')][0]

    img = nib.load(src_dmri_file)
    bvals = np.loadtxt(src_bval_file)
    bvecs = np.loadtxt(src_bvec_file).T
    data = img.get_data()
    affine = img.get_affine()

    gradients = gradient_table(bvals,bvecs)
    tensor_model = dti.TensorModel(gradients)  
    tensors = tensor_model.fit(data)
    FA = dti.fractional_anisotropy(tensors.evals)
    FA[np.isnan(FA)] = 0
    Color_FA = np.array(255*(dti.color_fa(FA, tensors.evecs)),'uint8')
    
    out_evecs_file = os.path.join(src_dmri_dir, subj_name + par_evecs_tag)
    evecs_img = nib.Nifti1Image(tensors.evecs.astype(np.float32), affine)
    nib.save(evecs_img, out_evecs_file)

    out_fa_file = os.path.join(src_dmri_dir, subj_name + par_fa_tag)
    fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
    nib.save(fa_img, out_fa_file)

    out_cfa_file = os.path.join(src_dmri_dir, subj_name + par_cfa_tome_tag)
    cfa_img = nib.Nifti1Image(Color_FA, affine)
    nib.save(cfa_img, out_cfa_file)

    dt = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    out_cfa_file = os.path.join(src_dmri_dir, subj_name + par_cfa_trkvis_tag)
    cfa_img = nib.Nifti1Image((Color_FA.view((dt)).reshape(Color_FA.shape[:3])), affine)
    nib.save(cfa_img, out_cfa_file)


def compute_tracking(src_dti_dir, out_trk_dir, subj_name):

    # Loading FA and evecs data
    src_fa_file = os.path.join(src_dti_dir, subj_name + par_fa_tag)
    fa_img = nib.load(src_fa_file)
    FA = fa_img.get_data()
    affine = fa_img.get_affine()

    src_evecs_file = os.path.join(src_dti_dir, subj_name + par_evecs_tag)
    evecs_img = nib.load(src_evecs_file)
    evecs = evecs_img.get_data()

    # Computation of streamlines
    sphere = get_sphere('symmetric724') 
    peak_indices = dti.quantize_evecs(evecs, sphere.vertices)
    streamlines = EuDX(FA.astype('f8'),
                       ind=peak_indices, 
                       seeds=par_eudx_seeds,
                       odf_vertices= sphere.vertices,
                       a_low=par_eudx_threshold)

    # Saving tractography
    voxel_size = fa_img.get_header().get_zooms()[:3]
    dims = FA.shape[:3]
    seed = '_' + par_eudx_tag

    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = voxel_size
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = dims
    hdr['vox_to_ras'] = affine
    strm = ((sl, None, None) for sl in streamlines 
            if length(sl) > par_trk_min and length(sl) < par_trk_max)
    out_trk_file = os.path.join(out_trk_dir, subj_name + seed + par_trk_tag)
    nib.trackvis.write(out_trk_file, strm, hdr, points_space='voxel')    

    #tracks = [track for track in streamlines]
    #out_dipy_file = os.path.join(out_trk_dir, subj_name + seed + par_dipy_tag)
    #dpw = Dpy(out_dipy_file, 'w')
    #dpw.write_tracks(tracks)
    #dpw.close()


def tracking_eudx4csd(dir_src, dir_out, subj_name, verbose=False):

    # Load data
    fbval = dir_src +  [each for each in os.listdir(dir_src) if each.endswith('.bval')][0]
    fbvec = dir_src +  [each for each in os.listdir(dir_src) if each.endswith('.bvec')][0]

    #fbval = os.path.join(dir_src, subj_name + '.bval')
    #fbvec = os.path.join(dir_src, subj_name + '.bvec')
    fdwi =  os.path.join(dir_src, subj_name + par_iso_tag + ".nii.gz")
    #fmask = pjoin(dir_src, 'nodif_brain_mask_' + par_dim_tag + '.nii.gz')
    #fmask = pjoin(dir_src, 'wm_mask_' + par_b_tag + '_' + par_dim_tag + '.nii.gz')

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fdwi)
    data = img.get_data()
    affine = img.get_affine()
    voxel_size =  img.get_header().get_zooms()[:3]
    dims = data.shape[:3]
    #data, affine = load_nifti(fdwi, verbose)
    #mask, _ = load_nifti(fmask, verbose)

    sphere = get_sphere('symmetric724') 

    response, ratio = auto_response(gtab, data, roi_radius=par_ar_radius, 
                                    fa_thr=par_ar_fa_th)
    # print('Response function', response)

    # Model fitting
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    csd_peaks = peaks_from_model(csd_model, 
                                 data, 
                                 sphere,
                                 relative_peak_threshold=par_csd_peak,
                                 min_separation_angle=par_csd_angle,
                                 parallel=False)

    # Computation of streamlines
    streamlines = EuDX(csd_peaks.peak_values,
                       csd_peaks.peak_indices, 
                       seeds=par_eudx_seeds,
                       odf_vertices= sphere.vertices,
                       a_low=par_eudx_threshold)

    # Saving tractography
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = voxel_size
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = dims
    hdr['vox_to_ras'] = affine
    strm = ((sl, None, None) for sl in streamlines)
    trk_name = subj_name + '_' + par_csd_tag + '_' + par_eudx_tag + '.trk'
    trk_out = os.path.join(dir_out, trk_name)
    nib.trackvis.write(trk_out, strm, hdr, points_space='voxel')    


def tractome_preprocessing(src_trk_dir, subj_name):

    seeds = par_eudx_seeds
    par2fun={par_prototype_distance:bundles_distances_mam}
    prototype_distance=par2fun[par_prototype_distance]
    trk_basename = "%s_%s%s" % (subj_name,
                                par_eudx_tag,
                                par_trk_tag)
    spa_basename = os.path.splitext(trk_basename)[0] + '.spa'
    src_trk_file = os.path.join(src_trk_dir, trk_basename)
    out_spa_dir = os.path.join(src_trk_dir, '.temp')
    if not os.path.exists(out_spa_dir):
        os.makedirs(out_spa_dir)
    out_spa_file = os.path.join(out_spa_dir, spa_basename)

    streams, hdr = nib.trackvis.read(src_trk_file, points_space='voxel')
    streamlines =  np.array([s[0] for s in streams], dtype=np.object)
    dissimilarity_matrix = compute_dissimilarity(streamlines, 
            prototype_distance, par_prototype_policy, par_prototype_num)

    info = {'dismatrix':dissimilarity_matrix,'nprot':par_prototype_num}
    pickle.dump(info, open(out_spa_file,'w+'), protocol=pickle.HIGHEST_PROTOCOL)


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
