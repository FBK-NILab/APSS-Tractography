APSS-Tractography
=================

Pipeline for processing diffusion MRI and recostruncting deterministic tractography according to the requirement of the Neurosurgery Department of APSS Trentino. At this moment the pipeline includes the following steps:

1. Creating .nii file from .dcm files for T1.
2. Creating .nii,.bval and .bvec files from .dcm files for the diffusion data.
2. Preprocessing:
 2.1. Brain Extraction (both for T1 and diffusion images)
 2.2. Eddy current correction
3. Deterministic tractography reconstruction


Dependencies
------------

* NiBabel : http://nipy.org/nibabel , provides read and write access to common medical and neuroimaging file formats.
* Dipy: http://www.dipy.org , provides tools for dMRI data analysis.
* mricron: http://www.nitrc.org/projects/mricron, Toolbox for magnetic resonance image conversion, viewing and       analysis.
* FSL: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/, provides tools for analysis of FMRI, MRI and DTI brain imaging data.


To Run the Pipeline
---------------
```
ipython 
run main.py
```

The current main.py file contains the definition of all input and output filepaths for the different steps. It also contains some of the default values for the options of the different processing methods. If the user want to change any of these, she just has to change them on the main. The only two variables that need to be changed before running the main.py on your PC are:

dir_DICOM_data: Local directory where dicom files are.

main_data_directory: Local directory where all the results should be saved. Moreover, this is the folder where the .nii, .bval and .bvec files are localized, in case the .dcm to .nii conversion has already been done.

