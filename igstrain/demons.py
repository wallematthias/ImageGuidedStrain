import SimpleITK as sitk
import sys
import os
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from glob import glob

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
    
    
def pad_images(images):
    max_shape = np.max([image.shape for image in images], axis=0)
    padded_images = []
    
    for image in images:
        pad_x = max_shape[0] - image.shape[0]
        pad_y = max_shape[1] - image.shape[1]
        pad_z = max_shape[2] - image.shape[2]
        
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
    x, y, z = np.split(np.squeeze(np_displacement), 3, axis=-1)

    # Merge dictionaries and rename keys
    data = {'fixed': np_fixed, 'effective_strain': von_mises_strain,
            'displacement_x': x, 'displacement_y': y, 'displacement_z': z,
            **{f'strain_{key}': value for key, value in np_strain_tensor.items()}}


    if name is not None:
        dict_to_vtkFile(data, name)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

    return data


def main():
    
    """
    Interactive script for performing demons registration with user-provided inputs.

    This script prompts the user to enter the paths for the moving and fixed image files,
    as well as additional parameters for the demons registration process. The user can press
    Enter to accept default values for optional parameters.

    Parameters:
    - moving_path (str): Path to the moving image file.
    - fixed_path (str): Path to the fixed image file.
    - name (str): Output name for the registration result. Default is None.
    - iterations (int): Number of iterations for the demons registration. Default is 30.
    - scaling_factors (list): Scaling factors for downscaling at each resolution level. Default is [8, 4, 2].
    - update_field_sigma (float): Standard deviation for smoothing the update field. Default is 1.
    - deformation_field_sigma (float): Standard deviation for smoothing the deformation field. Default is 10.

    Returns:
    - None
    """
    
    # Ask the user for the moving and fixed image paths
    fixed_path = input("Enter the path to the fixed image file: ")
    moving_path = input("Enter the path to the moving image file: ")

    # Read the moving and fixed images
    moving_image = read_file(moving_path)
    fixed_image = read_file(fixed_path)

    # Ask the user for additional parameters with defaults
    name = input("Enter the output name (press Enter for default, default is based on input name): ") or f'{os.path.basename(fixed_path).split('.')[0]}.vti'
    iterations = int(input("Enter the number of iterations (press Enter for default, default is 30): ") or 30)
    scaling_factors = list(map(int, input("Enter scaling factors (press Enter for default, default is 8, 4, 2): ") or [8, 4, 2]))
    update_field_sigma = float(input("Enter update field sigma (press Enter for default, default is 1): ") or 1)
    deformation_field_sigma = float(input("Enter deformation field sigma (press Enter for default, default is 10): ") or 10)

    # Perform demons registration
    demons_registration(fixed_image, moving_image, name=name, iterations=iterations,
                        scaling_factors=scaling_factors,
                        sigmas=[update_field_sigma, deformation_field_sigma])

if __name__ == "__main__":
    main()