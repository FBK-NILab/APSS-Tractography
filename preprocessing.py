import nibabel as nib
from dipy.external.fsl import bet, eddy_correct, dcm2nii
from dipy.align.aniso2iso import resample
import os


def preprocess (dicom_directory=None, niftii_output_dir = None, filename_bet = None, output_file_bet = None, bet_options = None,
                   filename_eddy = None, output_file_eddy = None, new_voxel_size = None, file_resizing = None, output_file_resize = None):


    #Dicom to niftii step
    print "Converting from dicom to niftii"
    if dicom_directory is not None and niftii_output_dir is not None:
        dcm2nii(dicom_directory, niftii_output_dir,filt='*.dcm', options = '-f y -e n -p n -a n -d n -g n -i n -o')
    
    
    #Brain extraction stepn

    print "Performing brain extraction"
    if output_file_bet is not None:
        if filename_bet is None: 
           #Get niftii filepath
            filename_bet = niftii_output_dir +  [each for each in os.listdir(niftii_output_dir) if each.endswith('.nii')][0]   
        if bet_options is not None:
            options = bet_options
        else:
            options = ' -R -F -f .2 -g 0'
        
        bet(filename_bet, output_file_bet,options)

    #Eddy Current correction
    print "Performing Eddy current correction"
    if filename_eddy is not None and output_file_eddy is not None:
        eddy_correct(filename_eddy, output_file_eddy, ref=0)


    #Resampling the size of the voxel for dipy reconstruction
    print "Resizing to isotropic voxel size"
    if file_resizing is not None and output_file_resize is not None:
        img = nib.load(file_resizing)
        old_data = img.get_data()
        old_affine = img.get_affine()
    
        zooms=img.get_header().get_zooms()[:3]
        print 'old zooms:', zooms
        if new_voxel_size is not None:
            new_zooms = new_voxel_size
        else:
            new_zooms=(2.,2.,2.)
        
        data, affine=resample(old_data,old_affine,zooms,new_zooms)
    
    #Save new data
    print "Saving Data after resapling:", output_file_resize
    data_img = nib.Nifti1Image(data=data, affine=affine)
    nib.save(data_img, output_file_resize)
