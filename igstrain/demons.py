import SimpleITK as sitk
import sys
import os
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from glob import glob
import argparse

def dict_to_vtkFile(data_dict, output_filename, spacing=None, origin=None, array_type=vtk.VTK_FLOAT):
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
        temp = np.ascontiguousarray(np.atleast_3d(array))
        vtkArray = numpy_to_vtk(
            temp.ravel(order='F'),
            deep=True, array_type=array_type
        )
        image.SetDimensions(array.shape)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        vtkArray.SetName(name)  # Set the name of the data array
        image.GetPointData().AddArray(vtkArray)  # Add the data array to vtkImageData

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

def pad_images(images):
    """
    Pad a list of 3D images with zeros to match the dimensions of the largest image.

    This function takes a list of 3D NumPy arrays representing images and pads each image
    with zeros along the x, y, and z axes to match the dimensions of the largest image in
    the list.

    Parameters:
    - images (list of ndarrays): List of 3D NumPy arrays representing images.

    Returns:
    - list of ndarrays: List of padded 3D images with dimensions matching the largest image.
    """

    # Find the maximum dimensions among the input images
    max_shape = np.max([image.shape for image in images], axis=0)
    
    # Pad each image to match the maximum dimensions
    padded_images = []
    for image in images:
        pad_x = max_shape[0] - image.shape[0]
        pad_y = max_shape[1] - image.shape[1]
        pad_z = max_shape[2] - image.shape[2]
        
        # Pad the image with zeros
        padded_image = np.pad(image, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant', constant_values=0)
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
        
        return image_array
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

def disp_to_strain(displacement, dx=1.0, dy=1.0, dz=1.0):
    """
    Computes the 3D strain tensor components from a given displacement field.

    Parameters:
    - displacement (numpy.ndarray): The 3D displacement field, where each component represents the displacement
      in the x, y, and z directions.
    - dx (float): Voxel size along the x-axis. Default is 1.0.
    - dy (float): Voxel size along the y-axis. Default is 1.0.
    - dz (float): Voxel size along the z-axis. Default is 1.0.

    Returns:
    - dict: A dictionary containing the 3D strain tensor components:
        - 'xx' (numpy.ndarray): εxx component
        - 'yy' (numpy.ndarray): εyy component
        - 'zz' (numpy.ndarray): εzz component
        - 'xy' (numpy.ndarray): εxy component
        - 'xz' (numpy.ndarray): εxz component
        - 'yz' (numpy.ndarray): εyz component

    Example:
    >>> displacement_field = np.random.rand(10, 10, 10, 3)  # Example 3D displacement field
    >>> strain_tensor = disp_to_strain(displacement_field, dx=0.5, dy=0.5, dz=0.5)
    >>> print(strain_tensor['xx'].shape)  # Shape of the εxx component
    >>> print(strain_tensor['yy'][0, 0, 0])  # Value of the εyy component at a specific point
    """
    # Compute gradients using the gradient function
    du_dx, du_dy, du_dz = np.gradient(displacement, dx, dy, dz, axis=(0, 1, 2))
    
    # Compute strain tensor components
    strain_xx = 0.5 * (du_dx[:, :, :, 0] + du_dx[:, :, :, 0])  # εxx component
    strain_yy = 0.5 * (du_dy[:, :, :, 1] + du_dy[:, :, :, 1])  # εyy component
    strain_zz = 0.5 * (du_dz[:, :, :, 2] + du_dz[:, :, :, 2])  # εzz component
    strain_xy = 0.5 * (du_dx[:, :, :, 1] + du_dy[:, :, :, 0])  # εxy component
    strain_xz = 0.5 * (du_dx[:, :, :, 2] + du_dz[:, :, :, 0])  # εxz component
    strain_yz = 0.5 * (du_dy[:, :, :, 2] + du_dz[:, :, :, 1])  # εyz component
    
    # Return the 3D strain tensor components as a dictionary
    strain_tensor = {'xx': strain_xx, 'yy': strain_yy, 'zz': strain_zz,
                     'xy': strain_xy, 'xz': strain_xz, 'yz': strain_yz}
    return strain_tensor

import numpy as np

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

import SimpleITK as sitk
import numpy as np

def demons_registration(np_fixed, np_moving, name=None, iterations=30, scaling_factors=[8, 4, 2], sigmas=[1, 10]):
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

    for j, scale in enumerate(scaling_factors + [1]):
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

        if j == 0:
            displacementField = demons.Execute(fixed_down, moving_down)
        else:
            displacementField = shrink_filter.Execute(displacementField)
            displacementField = demons.Execute(fixed_down, moving_down, displacementField)

        # Upscale the displacement field
        expand_filter = sitk.ResampleImageFilter()
        expand_filter.SetSize(fixed.GetSize())
        displacementField = expand_filter.Execute(displacementField)

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
    x, y, z = np.split(np_displacement, 3, axis=-1)
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)
    
    # Merge dictionaries and rename keys
    data = {'fixed': np_fixed, 'effective_strain': von_mises_strain,
            'displacement_x': x, 'displacement_y': y, 'displacement_z': z,
            **{f'strain_{key}': np.squeeze(value) for key, value in np_strain_tensor.items()}}


    if name is not None:
        
        if name.split('.')[-1] == 'vti':
            dict_to_vtkFile(data, name)
        else: 
            image = sitk.GetImageFromArray(von_mises_strain)
            sitk.WriteImage(image, name)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

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
        moving_image = read_file(glob(args.moving_path)[0])
    except:
        print(f'Moving image not found {args.moving_path}')
    try:
        fixed_image = read_file(glob(args.fixed_path)[0])
    except:
        print(f'Fixed image not found {args.fixed_path}')
        

    fixed_image, moving_image = pad_images([fixed_image, moving_image])

    
    # Set default name if not provided
    default_path = os.path.dirname(args.fixed_path)
    default_name = f'{os.path.basename(args.moving_path).split(".")[0]}_demons_strain.mha'

    # Check if the file already exists, and if it does, add a suffix
    suffix = 1
    while os.path.exists(default_name):
        default_name = os.path.join(default_path, f'{default_name.split(".")[0]}_{suffix}.mha')
        suffix += 1

    name = args.name or os.path.join(default_path, default_name)

    # Perform demons registration
    data = demons_registration(fixed_image, moving_image, name=name, iterations=args.iterations,
                        scaling_factors=args.scaling_factors,
                        sigmas=[args.update_field_sigma, args.deformation_field_sigma])

    return data 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive script for performing demons registration with user-provided inputs.")
    
    parser.add_argument("fixed_path", type=str, help="Path to the fixed image file.")
    parser.add_argument("moving_path", type=str, help="Path to the moving image file.")
    parser.add_argument("--name", type=str, default="", help="Output name for the registration result. Default is based on the fixed image name.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations for the demons registration. Default is 30.")
    parser.add_argument("--scaling_factors", type=int, nargs="+", default=[8, 4, 2], help="Scaling factors for downscaling at each resolution level. Default is [8, 4, 2].")
    parser.add_argument("--update_field_sigma", type=float, default=1.0, help="Standard deviation for smoothing the update field. Default is 1.0.")
    parser.add_argument("--deformation_field_sigma", type=float, default=10.0, help="Standard deviation for smoothing the deformation field. Default is 5.0.")
    
    args = parser.parse_args()
    
    
    main(args)