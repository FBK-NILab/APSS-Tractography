# -*- coding: utf-8 -*-

"""
Module implementing the main sequence of pipeline analysis to process
diffusion MRI designed according to the requirements from APSS

Copyright (c) 2014, Fondazione Bruno Kessler
Distributed under the BSD 3-clause license. See COPYING.txt.
"""


## Setting directories for Outputs and Inputs of the rest of the methods
## General directory where data resides and where to save the results
main_data_directory = ''

## Suffix for output filename of structural nifti
par_mri_tag =  'mri'

## Suffix for output filename of diffusion nifti
par_dmri_tag =  'dmri'

## Optional parameters of 'dcm2nii' command
par_dcm2nii_options = '-f y -e n -p n -a n -d n -g n -i n -o'

## Optional parameters of 'bet' command
par_bet_options = ' -R -F -f .35 -g 0'
par_bet4dwi_options = ' -R -F'

## Suffix for output filename of 'bet' command
par_bet_tag =  '_bet'

## Suffix for output filename of 'eddy_correct' command
par_ecc_tag = '_ecc.nii.gz'

## Parameter 'ref' of 'eddy_correct' command
par_ecc_ref = 0

## Suffix for output filename of rescaling isotropic voxels
par_iso_tag = '_iso'

## Voxel size of isotropic rescaling 
par_iso_voxel_size = (2.,2.,2.)

## Dof parameter of flirt registration
par_flirt_dof = 6

## Flirt optional parameters
par_flirt_opt = '-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear'

## Suffix for output filename of flirt registration
par_flirt_tag = '_flirt.nii.gz'

## Suffix for filename of affine transformation
par_aff_tag = '_flirt.mat'

## Auto response radius
par_ar_radius = 10

## Auto response FA threshold
par_ar_fa_th = 0.7

## Relative peak threshold for constraint spherical deconvolution
par_csd_peak = 0.5

## Minimum separation angle for constraint spherical deconvolution
par_csd_angle = 25

## Number of seeds for tracking
par_eudx_seeds = 1000000

## FA threshold for tracking 
par_eudx_threshold = .2

## Number of seeds for tracking
par_eudx_tag = '1M'

## Suffix for output filename of eigen vectors
par_evecs_tag = '_evecs.nii.gz'

## Suffix for output filename of FA
par_fa_tag = '_FA.nii.gz'

## Suffix for output filename of color FA with trackvis data structure
par_cfa_trkvis_tag = '_Color_FA_trk.nii.gz'

## Suffix for output filename of color FA with dipy data structure
par_cfa_tome_tag = '_Color_FA_dpy.nii.gz'

## Suffix for Constraint Spherical Deconvolution reconstruction
par_csd_tag = 'csd'

## Pathname of atlas for registration
par_atlas_dir = '/usr/local/fsl/data/atlases/JHU/'

## Name of atlas for registration of FA 
par_atlas_file = 'JHU-ICBM-FA-2mm.nii.gz'

## Name of affine for roi registration
par_roi_aff = 'affine4roi'

## Pathname of rois for registration
par_roi_dir = '/Users/silviosarubbo/CLINT/Datasets/JHU/'

## Name of atlas for registration of ROI 
par_roi_atlas = 'JHU-ICBM-DWI-1mm.nii.gz'

## Suffix for output filename of registered atlas
par_atlas_tag = '_atlas.nii'

## Dof parameter of flirt registration
par_atlas_dof = 12

## Minimum length of streamlines
par_trk_min = 5

## Maximum length of streamlines
par_trk_max = 1500

## Suffix for output filename of dipy tractography
par_trk_tag = '_apss.trk'

## Suffix for output filename of trackvis tractography
par_dipy_tag = '_apss.dpy'

## Suffix for output filename of dipy tractography
par_csa_tag = '_csa.trk'

## Number of prototypes for dissimilarity computation
par_prototype_num = 40

## Policy of prototype selection for dissimilarity computation
par_prototype_policy = 'sff'

## Distance measure for dissimilarity computation
par_prototype_distance = bundles_distances_mam

