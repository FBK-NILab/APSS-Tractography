APSS-Tractography
=================

Pipeline for processing diffusion MRI and recostruncting deterministic tractography according to the requirement of the Neurosurgery Department of APSS Trentino. At this moment the pipeline includes the following steps:

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


Dependencies
------------

* NiBabel : http://nipy.org/nibabel , provides read and write access to common medical and neuroimaging file formats.
* Dipy: http://www.dipy.org , provides tools for dMRI data analysis.
* mricron: http://www.nitrc.org/projects/mricron, Toolbox for magnetic resonance image conversion, viewing and analysis.
* FSL: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/, provides tools for analysis of FMRI, MRI and DTI brain imaging data.


To Run the Pipeline
---------------
```
pipeline.py {<path-to-data>|"number number number"}
```

For a batch session of analysis all the parameters can be configured in the file parameter.py.

The expected organization of data folder is as follows
- <Subject_Folder>
-- DICOM
--- Structural
--- Diffusion
