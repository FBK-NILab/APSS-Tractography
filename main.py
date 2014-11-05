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
print "Setting preprocessing variables."
## Unless the user wants a different name for the resulting files, none of the values for the defined variables have to change, except for dir_DICOM_data 
## and main_data_directory, which are the main directories containing the data and where the results will be saved, respectively.

##Directory of DICOM files
dir_DICOM_diffusion = './Subject/DTI_DICOM'
dir_DICOM_T1 = './Subject/T1_DICOM'

##Setting directories for Outputs and Inputs of the rest of the methods
##General directory where data resides and where to save the results
main_data_directory = './Subject/'

##Creating directory to save Niftii generated data
niftii_dirname = os.path.join(main_data_directory, 'Niftii')  
if not os.path.exists(niftii_dirname):
    os.makedirs(niftii_dirname)
    
niftii_diffusion_dirname = os.path.join(niftii_dirname, 'Diffusion/')  
if not os.path.exists(niftii_diffusion_dirname):
    os.makedirs(niftii_diffusion_dirname)
    
niftii_t1_dirname = os.path.join(niftii_dirname, 'Structural/')
if not os.path.exists(niftii_t1_dirname):
    os.makedirs(niftii_t1_dirname)
    
##Creating directory and filepaths for preprocessed data
##Directory to save preprocessed data
prepro_t1_dirname = os.path.join(niftii_t1_dirname,'Preprocess/')
if not os.path.exists(prepro_t1_dirname):
    os.makedirs(prepro_t1_dirname)
    
##Variables for bet T1
#bet_file = None
bet_t1_file_savepath = os.path.join(prepro_t1_dirname,'T1_bet.nii.gz')
bet_t1_options = ' -R -m -f .5 -g 0'

## Calling preprocessing function for T1 image
preprocess(dicom_directory=dir_DICOM_T1, niftii_output_dir = niftii_t1_dirname, output_file_bet = bet_t1_file_savepath,  bet_options = bet_t1_options)

### Moving T1_bet to Structural main directory
os.rename(bet_t1_file_savepath, os.path.join(niftii_t1_dirname,'T1_bet.nii.gz'))

## From this point, the steps are only related to the diffusion data

##Variables for bet on diffusion data
prepro_dif_dirname = os.path.join(niftii_diffusion_dirname,'Preprocess/')
if not os.path.exists(prepro_dif_dirname):
    os.makedirs(prepro_dif_dirname)

#bet_file = None
bet_diffusion_file_savepath = os.path.join(prepro_dif_dirname,'data_bet.nii.gz')
# bet_options = ' -R -F -f .2 -g 0' (default)  "Activate if you want to change"

##Variables for eddy current correction
eddy_file = bet_diffusion_file_savepath
eddy_file_savepath = os.path.join(prepro_dif_dirname,'data_ecc.nii.gz')

##Variables for voxel resizing
#voxel_size = [2., 2., 2.] (default)  "Activate if you want to change"
file_to_resize = eddy_file_savepath
resized_file_savepath = os.path.join(niftii_diffusion_dirname,'data_isotropic_voxels.nii') 


##Calling preprocessing function. In this version we are doing all preprocessing steps. If you want to skip any, don't specify the needed parameters.
preprocess(dicom_directory=dir_DICOM_diffusion, niftii_output_dir = niftii_diffusion_dirname, output_file_bet = bet_diffusion_file_savepath, 
           filename_eddy = eddy_file, output_file_eddy = eddy_file_savepath, file_resizing = file_to_resize, output_file_resize = resized_file_savepath)


#Tractography reconstruction with dipy 
print "Setting reconstruction variables"

nii_filename = resized_file_savepath
bval_filename = niftii_diffusion_dirname +  [each for each in os.listdir(niftii_diffusion_dirname) if each.endswith('.bval')][0]
bvec_filename = niftii_diffusion_dirname +  [each for each in os.listdir(niftii_diffusion_dirname) if each.endswith('.bvec')][0]

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

dti_directory = os.path.join(main_data_directory,'TRK/Tractography/')
if not os.path.exists(dti_directory):
    os.makedirs(dti_directory)

save_filename_trk = os.path.join(dti_directory,'tome_%s.trk') % seed
save_trk(streamlines, voxel_size, dims, save_filename_trk) 

save_filename_dpy = os.path.join(dti_directory,'tome_%s.dpy') % seed
save_dpy(streamlines, save_filename_dpy)

print "Save FA"
mapfile = os.path.join(niftii_diffusion_dirname,'FA_map.nii.gz')
nib.save(nib.Nifti1Image(FA.astype(np.float32), img.get_affine()), mapfile)

print "Save Color FA"
cfa_file = os.path.join(niftii_diffusion_dirname,'Color_FA_map.nii.gz')
nib.save(nib.Nifti1Image(np.array(255*CFA,'uint8'), img.get_affine()), cfa_file)
