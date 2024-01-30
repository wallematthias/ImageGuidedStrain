import SimpleITK as sitk
import sys
import os
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from glob import glob
import argparse
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from types import SimpleNamespace
import pandas as pd


def display_slice(img, component_index=0, slice_index=None):
    """
    Display a slice of a specific component (x, y, or z) from a 3D displacement field.
    
    Parameters:
    - img: SimpleITK.Image object representing a displacement field.
    - component_index: Index of the component to display (0 for x, 1 for y, 2 for z).
    - slice_index: Index of the slice to display. If None, displays the middle slice.
    """
    img_array = sitk.GetArrayFromImage(img)
    
    # If no slice index is provided, show the middle slice in the Z dimension
    if slice_index is None:
        slice_index = img_array.shape[0] // 2
    
    plt.figure(figsize=(10, 10))
    # Select the component to display
    plt.imshow(img_array[slice_index, :, :, component_index], cmap='gray')
    plt.axis('off')  # Hide the axes
    plt.show()
    
    
def get_bounding_box(binary_mask):
    """
    Get the bounding box slices of a 3D binary mask.
    
    Parameters:
        binary_mask (numpy.ndarray): 3D binary mask
    
    Returns:
        slices (tuple of slices): Bounding box slices for each dimension
    """
    # Find non-zero indices along each axis
    non_zero_indices = np.where(binary_mask)
    
    # Determine the bounding box slices
    slices = tuple(slice(np.min(idx), np.max(idx) + 1) for idx in non_zero_indices)
    
    return slices

def dict_to_vtkFile(data_dict, output_filename, spacing=None, origin=None, direction=None, array_type=vtk.VTK_FLOAT):
    '''Convert numpy arrays in a dictionary to vtkImageData

    Default spacing is 1 and default origin is 0.

    Args:
        data_dict (dict):       A dictionary with keys as names and numpy arrays as values
        spacing (np.ndarray):   Image spacing
        origin (np.ndarray):    Image origin
        array_type (int):       Datatype from vtk

    Returns:
        vtkImageReader:         The corresponding vtkImageReader or None
                                if one cannot be found.
    '''
    # Set default values
    if spacing is None:
        spacing = np.ones_like(next(iter(data_dict.values())).shape)
    if origin is None:
        origin = np.zeros_like(next(iter(data_dict.values())).shape)

    # Convert data_dict to vtkImageData
    image = vtk.vtkImageData()
    for name, array in data_dict.items():
        array = np.transpose(array, (2, 1, 0)) #Here we have to transpose for vtk format
        temp = np.ascontiguousarray(np.atleast_3d(array))
        vtkArray = numpy_to_vtk(
            temp.ravel(order='F'),
            deep=True, array_type=array_type
        )
        image.SetDimensions([d+1 for d in array.shape])
        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        vtkArray.SetName(name)  # Set the name of the data array
        image.GetCellData().AddArray(vtkArray)  # Add the data array to vtkImageData

    # Write the dataset to .vti file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(image)
    writer.Write()
    
def vtkFile_to_dict(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    
    # Create an empty dictionary to store the NumPy arrays with their names
    image_data_dict = {}
    
    # Get the number of point data arrays (fields) in the VTK file
    num_arrays = reader.GetNumberOfPointArrays()
    
    for i in range(num_arrays):
        array_name = reader.GetPointArrayName(i)
        print(f"Reading dataset with array name: {array_name}")
    
        # Get the i-th dataset (point data array)
        array = reader.GetOutput().GetPointData().GetArray(i)
    
        if array is not None:
            # Convert the VTK data array to a NumPy array
            numpy_array = vtk_to_numpy(array)
    
            # Get the original dimensions of the dataset
            extent = reader.GetOutput().GetExtent()
            x_min, x_max, y_min, y_max, z_min, z_max = extent
            x_dim = x_max - x_min + 1
            y_dim = y_max - y_min + 1
            z_dim = z_max - z_min + 1
    
            # Reshape the NumPy array to its original shape
            original_array = numpy_array.reshape((z_dim, y_dim, x_dim))
            rotated_array = np.transpose(original_array, (2, 1, 0))
    
            # Add the NumPy array to the dictionary with the image name as the key
            image_data_dict[array_name] = rotated_array
        else:
            print("Error: Unable to read the dataset.")

    return image_data_dict

def pad_images(images, extra_padding=0):
    """
    Pad a list of 3D images with zeros and additional padding to avoid image boundary effects.

    This function takes a list of 3D NumPy arrays representing images and pads each image
    with zeros along the x, y, and z axes to match the dimensions of the largest image in
    the list. It also adds an extra padding on all sides to avoid image boundary effects.

    Parameters:
    - images (list of ndarrays): List of 3D NumPy arrays representing images.
    - extra_padding (int): Extra padding to be added on all sides. Default is 20.

    Returns:
    - list of ndarrays: List of padded 3D images with dimensions matching the largest image.
    """

    # Find the maximum dimensions among the input images
    max_shape = np.max([image.shape for image in images], axis=0)
    
    # Add extra padding to the maximum dimensions
    max_shape += 2 * extra_padding
    
    # Pad each image to match the maximum dimensions with extra padding
    padded_images = []
    for image in images:
        current_shape = np.array(image.shape)
        pad_values = (max_shape - current_shape) // 2
        remainder = (max_shape - current_shape) % 2

        # Calculate padding for each dimension
        pad_x = (extra_padding + pad_values[0], extra_padding + pad_values[0] + remainder[0])
        pad_y = (extra_padding + pad_values[1], extra_padding + pad_values[1] + remainder[1])
        pad_z = (extra_padding + pad_values[2], extra_padding + pad_values[2] + remainder[2])

        # Pad the image with zeros and extra padding
        padded_image = np.pad(image, (pad_x, pad_y, pad_z), mode='constant', constant_values=0)
        padded_images.append(padded_image)
    
    return padded_images

def read_file(file_path):
    """
    Reads a Medical Head Image (MHA) file and returns the corresponding NumPy array.

    Parameters:
    - file_path (str): The path to the MHA file to be read.

    Returns:
    - numpy.ndarray or None: The NumPy array representing the image if successful, 
      otherwise returns None in case of an error.

    Raises:
    - Exception: If an error occurs during the file reading process.

    Example:
    >>> file_path = "path/to/your/file.mha"
    >>> image_array = read_file(file_path)
    >>> if image_array is not None:
    >>>     print("File successfully loaded.")
    >>> else:
    >>>     print("Error loading file.")
    """
    try:
        # Read the MHA file
        image = sitk.ReadImage(file_path)  

        # Convert SimpleITK image to NumPy array
        image_array = sitk.GetArrayFromImage(image)  
        
        return image_array, image.GetSpacing(), image.GetOrigin(), image.GetDirection()
    
    except Exception as e:
        print("Error loading file:", str(e))
        return None

def command_iteration(filter):
    """
    Prints the current iteration number and metric value during the execution of a registration filter.

    Parameters:
    - filter: The registration filter object being used.

    Returns:
    - None
    """
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    None


def disp_to_strain(np_displacement):
    """
    Calculate the strain tensor components from a 3D displacement array with displacement components
    stacked along the fourth axis, and return the strain tensor as a dictionary.

    Parameters:
    - np_displacement: A numpy array of shape (nx, ny, nz, 3) where nx, ny, nz are the dimensions
      in the x, y, and z directions respectively, and the last dimension stores the displacement
      components (u, v, w).
      
    Returns:
    - A dictionary with keys 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', each mapping to a numpy array
      representing the corresponding strain tensor component.
    """
    gradients = np.zeros(np_displacement.shape + (3,))  # Shape will be (nx, ny, nz, 3, 3)
    
    for i in range(3):  # Loop over the displacement components
        for j, axis in enumerate(['x', 'y', 'z']):  # Loop over the spatial dimensions order of the displacement field is weird
            gradients[..., i, j] = np.gradient(np_displacement[..., i], axis=j)
    
    # Extract gradients for easier access
    du_dx, du_dy, du_dz = gradients[..., 0, 0], gradients[..., 0, 1], gradients[..., 0, 2]
    dv_dx, dv_dy, dv_dz = gradients[..., 1, 0], gradients[..., 1, 1], gradients[..., 1, 2]
    dw_dx, dw_dy, dw_dz = gradients[..., 2, 0], gradients[..., 2, 1], gradients[..., 2, 2]
    
    # Compute the strain components
    strain_xx = du_dx
    strain_yy = dv_dy
    strain_zz = dw_dz
    strain_xy = 0.5 * (du_dy + dv_dx)
    strain_xz = 0.5 * (du_dz + dw_dx)
    strain_yz = 0.5 * (dv_dz + dw_dy)
    
    # Construct the strain tensor dictionary
    strain_tensor = {
        'xx': strain_xx,
        'yy': strain_yy,
        'zz': strain_zz,
        'xy': strain_xy,
        'xz': strain_xz,
        'yz': strain_yz
    }
    
    return strain_tensor

def strain_to_mises(strain_tensor):
    """
    Calculates the von Mises strain from a given 3D strain tensor. 

    Parameters:
    - strain_tensor (dict): A dictionary containing the 3D strain tensor components:
        - 'xx' (numpy.ndarray): εxx component
        - 'yy' (numpy.ndarray): εyy component
        - 'zz' (numpy.ndarray): εzz component
        - 'xy' (numpy.ndarray): εxy component
        - 'xz' (numpy.ndarray): εxz component
        - 'yz' (numpy.ndarray): εyz component

    Returns:
    - numpy.ndarray: The von Mises strain calculated from the input strain tensor.

    Example:
    >>> strain_tensor = {'xx': np.random.rand(10, 10, 10), 'yy': np.random.rand(10, 10, 10),
    >>>                  'zz': np.random.rand(10, 10, 10), 'xy': np.random.rand(10, 10, 10),
    >>>                  'xz': np.random.rand(10, 10, 10), 'yz': np.random.rand(10, 10, 10)}
    >>> von_mises_strain = strain_to_mises(strain_tensor)
    >>> print(von_mises_strain.shape)  # Shape of the von Mises strain array
    >>> print(von_mises_strain[0, 0, 0])  # Value of the von Mises strain at a specific point
    """
    # Extract individual strain components from the input dictionary
    strain_xx = strain_tensor['xx']
    strain_yy = strain_tensor['yy']
    strain_zz = strain_tensor['zz']
    strain_xy = strain_tensor['xy']
    strain_xz = strain_tensor['xz']
    strain_yz = strain_tensor['yz']
    
    # Calculate von Mises strain
    von_mises_strain = np.sqrt(0.5 * ((strain_xx - strain_yy)**2 + (strain_yy - strain_zz)**2 +
                                      (strain_zz - strain_xx)**2 + 6 * (strain_xy**2 + strain_yz**2 + strain_xz**2)))
    
    return von_mises_strain


def getDVC_field(map_path, dicom_image):
    
    with h5py.File(map_path, 'r') as file:
        # Directly load the arrays
        x = file['map/x'][:]
        y = file['map/y'][:]
        z = file['map/z'][:]

    # Calculate zoom factors for each dimension
    zoom_factors = [target / current for target, current in zip(dicom_image.GetSize(), x.shape)]

    # Resize (upscale) the array, no idea why the sitk one is not working properly
    x = zoom(x, zoom_factors, order=3)  # order=3 for cubic interpolation
    y = zoom(y, zoom_factors, order=3)  # order=3 for cubic interpolation
    z = zoom(z, zoom_factors, order=3)  # order=3 for cubic interpolation
   
    # Stack the arrays to form a (n, n, n, 3) shaped array
    np_displacement = np.stack((x, y, z), axis=-1)
    
    # Convert the numpy array into a SimpleITK Image, treating it as a vector
    displacement_image = sitk.GetImageFromArray(np.transpose(np_displacement, (2, 1, 0, 3)), isVector=True)
    
    # Initialize the resampler with explicit settings for size and spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(dicom_image.GetSize())
    resampler.SetOutputSpacing(dicom_image.GetSpacing())
    resampler.SetOutputOrigin(dicom_image.GetOrigin())
    resampler.SetOutputDirection(dicom_image.GetDirection())
    resampler.SetTransform(sitk.Transform(dicom_image.GetDimension(), sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)

    # Execute the resampling
    resampled_displacement_image = resampler.Execute(displacement_image)
    print(f'Field Shape: {x.shape} Reference Shape: {dicom_image.GetSize()} Resampled Shape: {resampled_displacement_image.GetSize()}')

    #display_slice(displacement_image, component_index=1, slice_index=None)
    
    return resampled_displacement_image

def demons_registration(np_fixed, np_moving, name=None, iterations=30, scaling_factors=[8, 4, 2], sigmas=[1, 10], map_path=None,spacing=None, origin=None, direction=None, additional_data=None):
    """
    Performs multi-resolution Demons registration on 3D images and computes the von Mises strain.
    According to Zwahlen et al., 2015. https://doi.org/10.1115/1.4028991 
    with adapted sigma to 5 instead of 10. 
    
    Parameters:
    - np_fixed (numpy.ndarray): The fixed 3D image data.
    - np_moving (numpy.ndarray): The moving 3D image data to be registered.
    - name (str): The name of the output VTK file. If not provided, the default name is 'deformation.vti'.
    - iterations (int): The number of iterations for each resolution level. Default is 30.
    - scaling_factors (list): The scaling factors for downscaling the images at each resolution level. Default is [8, 4, 2].
    - sigmas (list): Standard deviations for smoothing the displacement and update fields. Default is [1, 10].

    Returns:
    - dict: A dictionary containing various results, including fixed image, von Mises strain, and displacement fields.

    Example:
    >>> fixed_image = np.random.rand(10, 10, 10)
    >>> moving_image = np.random.rand(10, 10, 10)
    >>> registration_result = demons_registration(fixed_image, moving_image)
    >>> print(registration_result['fixed'].shape)  # Shape of the fixed image
    >>> print(registration_result['effective_strain'][0, 0, 0])  # Von Mises strain at a specific point
    >>> print(registration_result['Number Of Iterations'])  # Number of iterations performed in the registration
    """
    
    # Convert numpy arrays to SimpleITK images
    fixed = sitk.GetImageFromArray(np_fixed)
    moving = sitk.GetImageFromArray(np_moving)
    
    
    for j, scale in enumerate(scaling_factors):
        print(f'Downscaling image by a factor of {scale}...')
        
        # Downscale the image
        shrink_filter = sitk.ShrinkImageFilter()
        shrink_filter.SetShrinkFactors([scale, ] * 3)
        fixed_down = shrink_filter.Execute(fixed)
        moving_down = shrink_filter.Execute(moving)
        
        # Basic Demons Registration Filter
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(iterations)
        demons.SmoothDisplacementFieldOn()
        demons.SmoothUpdateFieldOn()
        demons.SetUpdateFieldStandardDeviations(sigmas[1])
        demons.SetStandardDeviations(sigmas[0])
        demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
            
        
        if (j == 0): # initial field
            if map_path is None:
                displacementField = demons.Execute(fixed_down,moving_down)
            else: # Case where you use the DVC image as initial displacement field
                dvc_displacement = getDVC_field(map_path, fixed)
                dvc_displacement_down = shrink_filter.Execute(dvc_displacement)
                displacementField = demons.Execute(fixed_down, moving_down, dvc_displacement_down)
        else:
            displacementField = shrink_filter.Execute(displacementField)
            displacementField = demons.Execute(fixed_down, moving_down, displacementField)

        # Upscale the displacement field
        expand_filter = sitk.ResampleImageFilter()
        expand_filter.SetSize(fixed.GetSize())
        displacementField = expand_filter.Execute(displacementField)

        
    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")
    
    # Convert data to numpy
    np_fixed = sitk.GetArrayFromImage(fixed)
    np_displacement = sitk.GetArrayFromImage(displacementField)
    
    np_strain_tensor = disp_to_strain(np_displacement)

    # Crop von Mises strain to the foreground
    von_mises_strain = strain_to_mises(np_strain_tensor)
    von_mises_strain[np_fixed == 0] = 0

    # Normalize von Mises strain to 99 percentile
    von_mises_strain_99 = np.percentile(von_mises_strain, 99)
    von_mises_strain[von_mises_strain > von_mises_strain_99] = von_mises_strain_99

    # Split displacements into their x,y,z components
    # Order of the xyz is a bit unclear 
    x, y, z = np.split(np_displacement, 3, axis=-1)
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)
 
    # This is necessary as our samples were oddly cropped and it is not clear where to pad them
    # in order to maintin the coordinate system. Therefore we need to change to origin of the coordinate     # system
    correction_factor = 0 - np.max(z)
    z+=correction_factor
    
    x*= np.asarray(spacing)[0]
    y*= np.asarray(spacing)[0]
    z*= np.asarray(spacing)[0]
    
    bb = get_bounding_box(np_fixed>0)
    
    # Merge dictionaries and rename keys
    data = {'baseline_seg': np_fixed[bb], 'demons_eff': von_mises_strain[bb],
            'demons_disp_x': x[bb], 'demons_disp_y': y[bb], 'demons_disp_z': z[bb],
            **{f'demons_strain{key}': np.squeeze(value)[bb] for key, value in np_strain_tensor.items()}}

    if additional_data is not None:
        if isinstance(additional_data, list):
            for path in additional_data:
                im = sitk.GetArrayFromImage(sitk.ReadImage(path))
                bb = get_bounding_box(im>0)
                data[os.path.basename(path).split('.')[0]] = im[bb]
        else: 
            im = sitk.GetArrayFromImage(sitk.ReadImage(additional_data))
            bb = get_bounding_box(im>0)
            data[os.path.basename(additional_data).split('.')[0]] = im[bb]
        
    
    if name is not None:
        # Writing the file: Note we just assume x=first dimension, y=second, z=third, if it's differen just interpret that way. 
        if name.split('.')[-1] == 'vti':
            print(f"Writing: {name}")
            dict_to_vtkFile(data, name, spacing=spacing, origin=origin, direction=direction)
        else: 
            
            for  key, image in data.items():
                im = sitk.GetImageFromArray(image)
                im.SetOrigin(origin)
                im.SetSpacing(spacing)
                im.SetDirection(direction)                
                new_path = '.'.join(name.split('.')[:-1]) +'_'+key+'.'+name.split('.')[-1]
                print(f"Writing: {new_path}")
                sitk.WriteImage(image, new_path)
    return data


def main(args):
    """
    Interactive script for performing demons registration with user-provided inputs.

    This script prompts the user to enter the paths for the moving and fixed image files,
    as well as additional parameters for the demons registration process. The user can press
    Enter to accept default values for optional parameters.

    Parameters:
    - fixed_path (str): Path to the fixed image file.
    - moving_path (str): Path to the moving image file.
    - name (str): Output name for the registration result. Default is based on the fixed image name.
    - iterations (int): Number of iterations for the demons registration. Default is 30.
    - scaling_factors (list): Scaling factors for downscaling at each resolution level. Default is [8, 4, 2].
    - update_field_sigma (float): Standard deviation for smoothing the update field. Default is 1.0.
    - deformation_field_sigma (float): Standard deviation for smoothing the deformation field. Default is 10.0.

    Returns:
    - None
    """
    
    # Read the moving and fixed images
    try:
        moving_image, _, _, _ = read_file(glob(args.moving_path)[0])
    except:
        print(f'Moving image not found {args.moving_path}')
    try:
        fixed_image, spacing, origin, direction = read_file(glob(args.fixed_path)[0])
    except:
        print(f'Fixed image not found {args.fixed_path}')
        
    fixed_image, moving_image = pad_images([fixed_image, moving_image])

    # Set default name if not provided
    default_path = os.path.dirname(args.fixed_path)
    default_name = f'{os.path.basename(args.moving_path).split(".")[0]}_demons_strain.vti'
    name = args.name or os.path.join(default_path, default_name)
    
    # Perform demons registration
    data = demons_registration(
        fixed_image, 
        moving_image, 
        name=name, 
        iterations=args.iterations,
        scaling_factors=args.scaling_factors,
        sigmas=[args.update_field_sigma, args.deformation_field_sigma],
        map_path=args.map, 
        spacing=spacing, 
        origin=origin, 
        direction=direction,
        additional_data=args.additional_data)

    return data 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive script for performing demons registration with user-provided inputs.")
    
    parser.add_argument("fixed_path", type=str, help="Path to the fixed image file.")
    parser.add_argument("moving_path", type=str, help="Path to the moving image file.")
    parser.add_argument("--map",default=None,type=str,help='Path to DVC map')
    parser.add_argument("--dicom",default=None,type=str,help='Path to dicom image')
    parser.add_argument("--name", type=str, default="", help="Output name for the registration result. Default is based on the fixed image name.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations for the demons registration. Default is 30.")
    parser.add_argument("--scaling_factors", type=int, nargs="+", default=[8, 4, 2], help="Scaling factors for downscaling at each resolution level. Default is [8, 4, 2].")
    parser.add_argument("--update_field_sigma", type=float, default=1.0, help="Standard deviation for smoothing the update field. Default is 1.0.")
    parser.add_argument("--deformation_field_sigma", type=float, default=10.0, help="Standard deviation for smoothing the deformation field. Default is 5.0.")
    parser.add_argument("--additional_data",default=None,type=str,help='Data added to the output vti file')
    args = parser.parse_args()
    
    
    main(args)