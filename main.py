# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 07:23:36 2014

@author: dianaporro
"""
import numpy as np
import nibabel as nib
from reconstruction_tract import tractography_rec, save_trk, save_dpy
from preprocessing import preprocess
import os


##Preprocessing
print "Setting preprocessing variables"

##Directory of DICOM files
dir_DICOM_data = '/home/tractome/meeting_Sarubbo/DTI_Sarubbo/Subject2_MT/DTI_DICOM'

##Setting directories for Outputs and Inputs of the rest of the methods
##General directory here data resides
main_data_directory = '/home/tractome/meeting_Sarubbo/DTI_Sarubbo/Subject2_MT/'

##General directory to save all
dirname = main_data_directory + 'DTI_nilab/'
if not os.path.exists(dirname):
    os.makedirs(dirname)

##Creating directory to save Niftii generated data
niftii_dirname = dirname + 'Niftii/'
if not os.path.exists(niftii_dirname):
    os.makedirs(niftii_dirname)

##Creating directory and filepaths for preprocessed data
##Directory to save preprocessed data
prepro_dirname = dirname + 'Preprocessed/'
if not os.path.exists(prepro_dirname):
    os.makedirs(prepro_dirname)

##Variables for bet
#bet_file = None
bet_file_savepath = prepro_dirname + 'data_bet.nii.gz'
# bet_options = ' -R -F -f .2 -g 0' (default)  "Activate if you want to change"

##Variables for eddy current correction
eddy_file = bet_file_savepath
eddy_file_savepath = prepro_dirname + 'data_ecc.nii.gz'

##Variables for voxel resizing
#voxel_size = [2., 2., 2.] (default)  "Activate if you want to change"
file_to_resize = eddy_file_savepath
resized_file_savepath = prepro_dirname + 'data_isotropic_voxels.nii' 


##Calling preprocessing function. In this version we are doing all preprocessing steps. If you want to skip any, don't specify the needed parameters.
preprocess(dicom_directory=dir_DICOM_data, niftii_output_dir = niftii_dirname, output_file_bet = bet_file_savepath, 
           filename_eddy = eddy_file, output_file_eddy = eddy_file_savepath, file_resizing = file_to_resize, output_file_resize = resized_file_savepath)


##Tractography reconstruction with dipy 
print "Setting reconstruction variables"

nii_filename = resized_file_savepath
bval_filename = niftii_dirname +  [each for each in os.listdir(niftii_dirname) if each.endswith('.bval')][0]
bvec_filename = niftii_dirname +  [each for each in os.listdir(niftii_dirname) if each.endswith('.bvec')][0]

print "Loading data"
img = nib.load(nii_filename)
bvals = np.loadtxt(bval_filename)
bvecs = np.loadtxt(bvec_filename).T

print "Setting parameters"
seed = 3000000
threshold = .2

print "Calling method to reconstruct tractography"
streamlines, FA, CFA = tractography_rec(img, bvals, bvecs, seed, threshold)

print "Saving data"
voxel_size = img.get_header().get_zooms()[:3]
dims = FA.shape[:3]

dti_directory = dirname + 'DTI/'
if not os.path.exists(dti_directory):
    os.makedirs(dti_directory)

save_filename_trk = dti_directory + 'dti_%s.trk' % seed
save_trk(streamlines, voxel_size, dims, save_filename_trk) 

save_filename_dpy = dti_directory + 'dti_%s.dpy' % seed
save_dpy(streamlines, save_filename_dpy)

print "Save FA"
mapfile = dti_directory+'FA_map.nii.gz'
nib.save(nib.Nifti1Image(FA.astype(np.float32), img.get_affine()), mapfile)

print "Save Color FA"
cfa_file = dti_directory+'Color_FA_map.nii.gz'
nib.save(nib.Nifti1Image(np.array(255*CFA,'uint8'), img.get_affine()), cfa_file)
