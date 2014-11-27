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
par_bet_options = ' -R -F -f .2 -g 0'

## Suffix for output filename of 'bet' command
par_bet_suffix =  '_bet.nii.gz'

## Suffix for output filename of 'eddy_correct' command
par_ecc_suffix = '_ecc.nii.gz'

## Parameter 'ref' of 'eddy_correct' command
par_ecc_ref = 0

## Suffix for output filename of rescaling isotropic voxels
par_ecc_suffix = '_iso.nii.gz'

## Voxel size of isotropic rescaling 
par_iso_voxel_size = (2.,2.,2.)

## Suffix for output filename of flirt registration
par_iso_suffix = '_iso.nii.gz'

## Dof parameter of flirt registration
par_flirt_dof = 6

## Suffix for output filename of flirt registration
par_flirt_suffix = '_flirt.nii.gz'

## Suffix for filename of affine transformation
par_aff_suffix = '_flirt.mat'

## Number of seeds for tracking
par_eudx_seeds = 3000000

## FA threshold for tracking 
par_eudx_threshold = .2

## Suffix for output filename of eigen vectors
par_evecs_suffix = '_evecs.nii.gz'

## Suffix for output filename of FA
par_fa_suffix = '_FA.nii.gz'

## Suffix for output filename of color FA with trackvis data structure
par_cfa_trkvis_suffix = '_Color_FA_trk.nii.gz'

## Suffix for output filename of color FA with dipy data structure
par_cfa_tome_suffix = '_Color_FA_dpy.nii.gz'

## Suffix for output filename of dipy tractography
par_atlas_file = 'HarvardOxford-cort-maxprob-thr50-2mm.nii.gz'

## Suffix for output filename of registered atlas
par_atlas_suffix = '_atlas.nii'

## Dof parameter of flirt registration
par_atlas_dof = 12

## Minimum length of streamlines
par_trk_min = 0

## Maximum length of streamlines
par_trk_max = 1000

## Suffix for output filename of dipy tractography
par_trk_suffix = '_apss.trk'

## Suffix for output filename of trackvis tractography
par_dipy_suffix = '_apss.dpy'

## Suffix for output filename of dipy tractography
par_csa_suffix = '_csa.trk'

## Number of prototypes for dissimilarity computation
par_prototype_num = 40

## Policy of prototype selection for dissimilarity computation
par_prototype_policy = 'sff'

## Distance measure for dissimilarity computation
par_prototype_distance = 'bundles_distances_mam'
