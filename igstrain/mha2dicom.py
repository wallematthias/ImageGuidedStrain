import glob
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import os

def pad_image_to_size(image_array, target_size):
    """
    Pads the image_array to target_size, centering the original image in the padded one.
    
    :param image_array: 3D numpy array of the original image
    :param target_size: Tuple (z, y, x) representing the desired size
    :return: Padded 3D numpy array
    """
    assert len(target_size) == 3, "Target size must be a tuple of 3 elements (z, y, x)"
    
    padding = [(t - s) // 2 for s, t in zip(image_array.shape, target_size)]
    padding_diff = [(t - s) % 2 for s, t in zip(image_array.shape, target_size)]
    pad_width = [(p, p + diff) for p, diff in zip(padding, padding_diff)]
    
    padded_image = np.pad(image_array, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_image

def mha_to_dicom(mha_path, target_size):
    sitk_image = sitk.ReadImage(mha_path)
    image_array = sitk.GetArrayFromImage(sitk_image)
    
    if target_size:
        image_array = pad_image_to_size(image_array, target_size)

    if image_array.max() > 255 or image_array.min() < 0:
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientName = "Doe^John"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = 'CT'
    ds.Manufacturer = ''

    pixel_spacing_x, pixel_spacing_y = sitk_image.GetSpacing()[1], sitk_image.GetSpacing()[2]
    slice_thickness = sitk_image.GetSpacing()[0]

    ds.Rows, ds.Columns = image_array.shape[1], image_array.shape[2]
    ds.NumberOfFrames = image_array.shape[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [f"{pixel_spacing_x:.13f}", f"{pixel_spacing_y:.13f}"]
    ds.SliceThickness = f"{slice_thickness:.13f}"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.ImagesInAcquisition = "1"
    ds.PixelData = image_array.tobytes()

    dicom_path = os.path.splitext(mha_path)[0] + '.dcm'
    ds.save_as(dicom_path)
    print(f"Saved DICOM file: {dicom_path}")

def find_max_dimensions(mha_files):
    max_dims = (0, 0, 0)  # Initialize with zero dimensions
    for file in mha_files:
        sitk_image = sitk.ReadImage(file)
        dims = sitk.GetArrayFromImage(sitk_image).shape
        max_dims = tuple(max(current, new) for current, new in zip(max_dims, dims))
    return max_dims

if __name__ == "__main__":
    mha_files = glob.glob('*IM.mha')
    if not mha_files:
        print("No .mha files found in the current directory.")
    else:
        max_dimensions = find_max_dimensions(mha_files)
        print(f"Maximum dimensions found: {max_dimensions}")
        for mha_file in mha_files:
            mha_to_dicom(mha_file, max_dimensions)