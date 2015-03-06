#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module implementing the main sequence of pipeline analysis to process
diffusion MRI designed according to the requirements from APSS

Copyright (c) 2014, Fondazione Bruno Kessler
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

"""
The available steps of pipeline analysis:
1. Structural Dicom to nifti
2. Structural brain extraction
3. Diffusion DICOM to nifti
4. Diffusion brain extraction
5. Eddy current correction
6. Rescaling isotropic voxel
7. Registration of structural data
8. Registration of atlas
9. Reconstruction of tensor model
10. Tracking of streamlines
11. Tractome preprocessing
12. Registration of ROI from atlas
"""

import os
import sys
import platform
import numpy as np
import nibabel as nib
from parameters import *
from pipenode import dicom_to_nifti, brain_extraction, eddy_current_correction, rescaling_isotropic_voxel, flirt_registration, atlas_registration, compute_reconstruction, compute_tracking, tractome_preprocessing, roi_registration

do_step = [1] * 13


def run_pipeline():

    print "*** BEGINNING OF PIPELINE COMPUTATION ***"

    if not os.path.isdir(main_data_directory):
        print "FAIL: setting parameters - FILE: data directory not found!"
        sys.exit()

    subj = os.path.basename(main_data_directory)
    step = 0

    print "Subject: ", subj
    print "Step %i: Setting parameters....." % step

    ## Directory of DICOM files

    dir_dicom_dmri = os.path.join(main_data_directory, 'DICOM/Diffusion')
    dir_dicom_mri = os.path.join(main_data_directory, 'DICOM/Structural')

    if platform.system() == 'Darwin':
        if os.path.isfile(os.path.join(dir_dicom_dmri, '.DS_Store')):
            os.remove(os.path.join(dir_dicom_dmri, '.DS_Store'))
        if os.path.isfile(os.path.join(dir_dicom_mri, '.DS_Store')):
            os.remove(os.path.join(dir_dicom_mri, '.DS_Store'))
                
    ## Creating directory to save Niftii generated data

    dir_nii = os.path.join(main_data_directory, 'Niftii')  
    if not os.path.exists(dir_nii):
        os.makedirs(dir_nii)
    
    dir_nii_mri = os.path.join(dir_nii, 'Structural/')
    if not os.path.exists(dir_nii_mri):
        os.makedirs(dir_nii_mri)
    
    dir_nii_dmri = os.path.join(dir_nii, 'Diffusion/')  
    if not os.path.exists(dir_nii_dmri):
        os.makedirs(dir_nii_dmri)
    
    ## Creating directory and filepaths for preprocessed data

    dir_mri_pre = os.path.join(dir_nii_mri,'Preprocess/')
    if not os.path.exists(dir_mri_pre):
        os.makedirs(dir_mri_pre)
    dir_dmri_pre = os.path.join(dir_nii_dmri,'Preprocess/')
    if not os.path.exists(dir_dmri_pre):
        os.makedirs(dir_dmri_pre)

    ## Creating directory for tractography

    dir_trk_tractography = os.path.join(main_data_directory,'TRK/Tractography/')
    if not os.path.exists(dir_trk_tractography):
        os.makedirs(dir_trk_tractography)

    dir_trk_dissection = os.path.join(main_data_directory,'TRK/Dissection/')
    if not os.path.exists(dir_trk_dissection):
        os.makedirs(dir_trk_dissection)

    dir_trk_roi = os.path.join(main_data_directory,'TRK/ROI/')
    if not os.path.exists(dir_trk_roi):
        os.makedirs(dir_trk_roi)

    print "DONE!"
    step += 1
    
    ## Preprocessing of Structural MRI data

    print "Step %i: Converting Structural DICOM files to Nifti..." % step
    if do_step[step]:
        dicom_to_nifti(dir_dicom_mri, dir_nii_mri, subj, par_mri_tag)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Brain extraction of structural data..." % step
    if do_step[step]:
        brain_extraction(dir_nii_mri, dir_mri_pre, subj, par_mri_tag)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    ## Preprocessing of Diffusion MRI data

    print "Step %i: Converting Diffusion DICOM files to Nifti..." % step
    if do_step[step]:
        dicom_to_nifti(dir_dicom_dmri, dir_nii_dmri, subj, par_dmri_tag)
        print 'DONE!'
    else:
        print "Skipped."
    step += 1

    print "Step %i: Brain extraction of diffusion data..." % step
    if do_step[step]:
        brain_extraction(dir_nii_dmri, dir_dmri_pre, subj, par_dmri_tag)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Eddy current correction of diffusion data..." % step
    if do_step[step]:
        eddy_current_correction(dir_dmri_pre, dir_dmri_pre, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Rescaling isotropic voxels..." % step
    if do_step[step]:
        rescaling_isotropic_voxel(dir_dmri_pre, dir_nii_dmri, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Registration of structural data..." % step
    if do_step[step]:
        flirt_registration(dir_mri_pre, dir_nii_dmri, dir_nii_mri, dir_mri_pre,subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Registration of atlas..." % step
    if do_step[step]:
        atlas_registration(dir_nii_dmri, dir_nii_dmri, dir_dmri_pre, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Reconstruction of tensor model..." % step
    if do_step[step]:
        compute_reconstruction(dir_nii_dmri, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Tracking of streamlines..." % step
    if do_step[step]:
        compute_tracking(dir_nii_dmri, dir_trk_tractography, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: Tractome preprocessing..." % step
    if do_step[step]:
        tractome_preprocessing(dir_trk_tractography, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "Step %i: ROI Registration..." % step
    if do_step[step]:
        roi_registration(dir_nii_dmri, dir_trk_roi, subj)
        print "DONE!"
    else:
        print "Skipped."
    step += 1

    print "*** END OF PIPELINE ***"


if __name__ == '__main__':
    
    for arg in sys.argv[1:]:
        if arg == '-h':
            print "Usage:"
            print "   pipeline.py <args>* "
            print "Arguments:"
            print "   path: <pathname>"
            print "         Data file directory"
            print "   step: 'number number ...'"
            print "         1. Structural Dicom to nifti"
            print "         2. Structural brain extraction"
            print "         3. Diffusion DICOM to nifti"
            print "         4. Diffusion brain extraction"
            print "         5. Eddy current correction"
            print "         6. Rescaling isotropic voxel"
            print "         7. Registration of structural data"
            print "         8. Registration of atlas"
            print "         9. Reconstruction of tensor model"
            print "         10. Tracking of streamlines"
            print "         11. Tractome preprocessing"
            print "         12. Registration of ROI from atlas"
            print "   help: -h"
            print "         this help"
            print "Examples:"
            print "   pipeline.py /path/to/my/data"
            print "   pipeline.py /path/to/my/data '1 2 7'"
            print "   pipeline.py '1 2 7'"
            sys.exit()

        if not os.path.isdir(arg):
            do_step =   [0] * 13
            arg_step = map(int, arg.split())
            for s in arg_step: do_step[s]=1

        if os.path.isdir(arg):
             main_data_directory = os.path.abspath(sys.argv[1])

    run_pipeline()


