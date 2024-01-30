import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools
import numpy as np
from skimage.measure import label   
import pandas as pd
import gc  # Import the garbage collection module

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def boundingbox_from_mask(mask, return_type='slice'):
    """Return tuple of slices or lists that describe the bounding box of the given mask.

    """
    if not np.any(mask):
        raise ValueError('Given mask is empty. Cannot compute a bounding box!')

    out = []
    try:
        for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
            nonzero = np.any(mask, axis=ax)
            extent = np.where(nonzero)[0][[0, -1]]
           # extent[1] += 1  # since slices exclude the last index
            if return_type == 'slice':
                out.append(slice(*extent))
            elif return_type == 'list':
                out.append(extent.tolist())
    except IndexError:
        raise ValueError('Mask is empty. Cannot compute a bounding box!')

    return tuple(reversed(out))


def load_ct_image(ct_image_path, input_size=(250, 250, 250)):
    # Use glob to find all TIFF files in the specified directory
    ct_scan_files = sorted(glob.glob(os.path.join(ct_image_path, '*.tif?')))
    
    # Determine the start and end slice indices to load the middle input_size[2] slices
    total_slices = len(ct_scan_files)
    middle_index = total_slices // 2
    half_slices_to_load = input_size[2] // 2
    start_slice = max(middle_index - half_slices_to_load, 0)
    end_slice = min(middle_index + half_slices_to_load, total_slices)
    
    # Limit the files to the middle slices
    ct_scan_files = ct_scan_files[start_slice:end_slice]
    
    # Create a progress bar for loading CT scan slices
    pbar = tqdm(total=len(ct_scan_files), desc='Loading CT Slices', position=0, leave=True)
    
    ct_slices = []
    for file in ct_scan_files:
        # Read the image slice
        slice_img = sitk.ReadImage(file)
        
        slice_img = sitk.Flip(slice_img, flipAxes=[False, True, False])

        # Get the size of the slice
        size = slice_img.GetSize()
        
        # Calculate the start index for cropping to center the new size
        start_x = max((size[0] - input_size[0]) // 2, 0)
        start_y = max((size[1] - input_size[1]) // 2, 0)
        
        # Crop the slice to the specified dimensions, centered
        cropped_slice = sitk.RegionOfInterest(slice_img, size=(input_size[0], input_size[1]), index=(start_x, start_y))
        
        ct_slices.append(cropped_slice)
        pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    
    # Create a 3D CT scan image from the cropped slices
    ct_3d = sitk.JoinSeries(ct_slices)
    
    return ct_3d

def load_mask(mask_path, reference_image):
    # Load the MHA mask
    mask_image = sitk.ReadImage(mask_path)

    bb = boundingbox_from_mask(getLargestCC(sitk.GetArrayFromImage(mask_image)))
    start_indices = np.array([bbox.start for bbox in bb])
    
    # Calculate the new origin
    new_origin =  [(start_indices[2])* 0.034,(start_indices[1])* 0.034,(start_indices[0])* 0.034]  

    print(mask_image.GetSize())
    # Get the size of the CT image
    ct_size = reference_image.GetSize()
    mask_size = mask_image.GetSize()
    
    # Calculate the padding sizes to make the mask the same size as the CT image
    pad_x = (ct_size[0] - mask_size[0]) // 2
    pad_y = (ct_size[1] - mask_size[1]) // 2
    pad_z = (ct_size[2] - mask_size[2]) // 2
    
    # Get the mask data as a NumPy array
    mask_array = sitk.GetArrayFromImage(mask_image)
    
    # Pad the mask array in the center using NumPy
    padded_mask_array = np.pad(mask_array, ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    
    # Create a new SimpleITK image from the padded NumPy array
    padded_mask_image = sitk.GetImageFromArray(padded_mask_array)
    
    # Ensure mask and reference image have the same datatype
    if padded_mask_image.GetPixelID() != reference_image.GetPixelID():
        padded_mask_image = sitk.Cast(padded_mask_image, reference_image.GetPixelID())
    
    return padded_mask_image, new_origin


def register_itk_mask_and_ct(mask_3d, ct_3d):

    def command_iteration(method):
        # This function will be called at each iteration
        # Get the current parameters from the optimizer
        current_parameters = method.GetOptimizerPosition()
        # Print the iteration number, metric value, and current transformation parameters
        print(f"Iteration {method.GetOptimizerIteration()}: Metric value = {method.GetMetricValue()}, Parameters: {current_parameters}")
    

    mask_3d.SetOrigin((0.,0.,0.))
    mask_3d.SetSpacing((1.,1.,1.))
    mask_3d.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

    ct_3d.SetOrigin((0.,0.,0.))
    ct_3d.SetSpacing((1.,1.,1.))
    ct_3d.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    
    mask_3d = sitk.Cast(mask_3d, sitk.sitkFloat32)
    ct_3d = sitk.Cast(ct_3d, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)


    R.SetOptimizerAsPowell(numberOfIterations=100,
                        maximumLineIterations=500,)

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    transform = sitk.TranslationTransform(3)
    R.SetInitialTransform(transform, inPlace=False)

    outTx = R.Execute(fixed=ct_3d, moving=mask_3d)
    outTx.SetParameters(np.round(outTx.GetParameters()))

    # Resample the mask using the obtained transformation
    resampled_mask = sitk.Resample(mask_3d, ct_3d, outTx, sitk.sitkLinear, 0.0, mask_3d.GetPixelID())

    # Convert SimpleITK images to NumPy arrays
    resampled_mask_np = sitk.GetArrayFromImage(resampled_mask)
    ct_3d_np = sitk.GetArrayFromImage(ct_3d)

    # Find the bounding box of the mask where the value is 255
    bb=boundingbox_from_mask(getLargestCC(resampled_mask_np>0))

    # Crop the CT and mask arrays based on the bounding box
    cropped_ct_np = ct_3d_np[bb]
    cropped_mask_np = resampled_mask_np[bb]

    # Convert cropped NumPy arrays back to SimpleITK images
    cropped_ct = sitk.GetImageFromArray(cropped_ct_np)
    cropped_mask = sitk.GetImageFromArray(cropped_mask_np)

    print(cropped_ct_np.shape)
    print(cropped_mask_np.shape)
    
    return cropped_ct, cropped_mask

def plot_overlay(im1, im2, alpha=0.5):
    """
    Plots an overlay of two SimpleITK images with different origins.
    
    Parameters:
        im1 (SimpleITK.Image): The reference image.
        im2 (SimpleITK.Image): The image to overlay on the reference image.
        alpha (float): Opacity of the overlay image. Range [0, 1].
    """
    # Resample im2 to match im1

    im1 = sitk.Cast(im1, sitk.sitkFloat32)
    im2 = sitk.Cast(im2, sitk.sitkFloat32)


    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im1)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.AffineTransform(im1.GetDimension()))
    resampler.SetOutputSpacing(im1.GetSpacing())
    resampler.SetOutputDirection(im1.GetDirection())
    resampler.SetOutputOrigin(im1.GetOrigin())
    resampler.SetSize(im1.GetSize())
    
    resampled_im2 = resampler.Execute(im2)
    
    # Convert SimpleITK images to NumPy arrays for plotting
    im1_array = sitk.GetArrayFromImage(im1)
    im2_array = sitk.GetArrayFromImage(resampled_im2)
    
    # Ensure the images are 2D for display
    if im1_array.ndim > 2:
        im1_array = im1_array[:,im1_array.shape[1] // 2,:]  # Middle slice for 3D images


    if im2_array.ndim > 2:
        im2_array = im2_array[:,im2_array.shape[1] // 2,:]  # Middle slice for 3D images


    # Plot an overlay for visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(im1_array, cmap='gray')
    plt.title('CT Scan (Cropped)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(im1_array, cmap='gray')
    plt.imshow(im2_array, cmap='jet', alpha=0.2)
    plt.title('CT Scan with Mask Overlay')
    plt.axis('off')


def transfer_images(ct_image_path,mask_path,outpath=None):

    if outpath is None:
        outpath = './'

    output_dir = './'

    ct_3d = load_ct_image(ct_image_path)
    mask_3d, new_origin = load_mask(mask_path, ct_3d)
    cropped_ct, cropped_mask = register_itk_mask_and_ct(mask_3d,ct_3d)



    # Specify the original and target intensity ranges
    np_ct = sitk.GetArrayFromImage(cropped_ct)
    original_min = np.min(np_ct).astype(float)
    original_max = np.max(np_ct).astype(float)

    # Apply intensity windowing to rescale to the 8-bit range
    ct_image_8bit = sitk.IntensityWindowing(cropped_ct,
        windowMinimum=original_min, windowMaximum=original_max,
        outputMinimum=0, outputMaximum=original_max-original_min)

    # Cast the rescaled image to 8-bit unsigned integer
    ct_image_16bit = sitk.PermuteAxes(sitk.Cast(ct_image_8bit, sitk.sitkInt16), [0, 1, 2]) # dont permute :D 
    seg_image_8bit = sitk.PermuteAxes(sitk.Cast(cropped_mask, sitk.sitkUInt8), [0, 1, 2])

    ct_image_16bit.SetSpacing((0.034,0.034,0.034))
    seg_image_8bit.SetSpacing((0.034,0.034,0.034))
    ct_image_16bit.SetOrigin(new_origin)
    seg_image_8bit.SetOrigin(new_origin)

    plot_overlay(ct_image_16bit, seg_image_8bit)

    ct_path = outpath.replace('.mha','_IM.mha')
    seg_path = outpath.replace('.mha','_SEG.mha')
    print(f'Writing {ct_path}')
    sitk.WriteImage(ct_image_16bit, ct_path)
    print(f'Writing {seg_path}')
    sitk.WriteImage(seg_image_8bit,seg_path)



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Interactive script")
    parser.add_argument("--filelist", default='./*.csv', type=str, help="Path to the files to be transferred")
    parser.add_argument("--inpath", default='./', type=str, help="Path to the files to be transferred from")    
    parser.add_argument("--outpath", default='./', type=str, help="Path to the files to be transferred to")
    args = parser.parse_args()
    
    name_table = pd.read_excel(glob.glob(args.filelist)[0])
    output_path = args.outpath
    
    print(name_table)


    for k in range(0,len(name_table)):
        mha_paths = f"{args.inpath}/{name_table['Sample'][k]}/**/*.mha"
        try:
            print(mha_paths)
            mha_files = sorted(glob.glob(mha_paths,recursive=True))
            if len(mha_files)==0:
                mha_paths = f"{args.inpath}/{name_table['Sample'][k]}/**/*.tif"
                mha_files = sorted(glob.glob(mha_paths,recursive=True))

            mha_files = [mha_files[-1]] + mha_files[:-1]

            tiff_path = f"{args.inpath}/{name_table['Sample'][k]}/**/*.tiff"
            tiff_files = sorted(glob.glob(tiff_path,recursive=True))

            for i in range(0,12):
                try:
                    newname = '_'.join(name_table.iloc[k,1:]) + f'_{i}_PERCENT.mha'

                    if i == 0:
                        # For i == 0, exclude paths containing "percent"
                        print('Baseline image')
                        tiff_path = os.path.dirname([path for path in tiff_files if "percent" not in path][0])
                        mha_path = [path for path in mha_files if "percent" not in path][0]

                    else:
                        tag = f'_{i}percent'
                        paths_with_1percent = [path for path in tiff_files if tag in path]
                        tiff_path = os.path.dirname(sorted(paths_with_1percent)[0])
                        
                        mha_with_1percent = [path for path in mha_files if tag in path]
                        mha_path = sorted(mha_with_1percent)[0]


                    print(tiff_path)
                    print(mha_path)
                    print(newname)

                    transfer_images(tiff_path,mha_path,outpath=os.path.join(output_path,newname))

                    plt.savefig(os.path.join(output_path,newname).replace('.mha','.png'))
                    #plt.show()
                except Exception as e:
                    print(e)
                    print('--end of folder--')
        
                gc.collect()
        except Exception as e:
            print(e)
        
        gc.collect()
