from ormir_xct.util.hildebrand_thickness import compute_local_thickness_from_mask
from glob import glob
import SimpleITK as sitk
import numpy as np

def main(path):
    
    file_list = glob(path)

    for file in file_list:
        print(f"Processing: {file}")

        image = sitk.ReadImage(file)
        np_array = sitk.GetArrayFromImage(image)

        thickness = compute_local_thickness_from_mask(
            np_array,
            voxel_width = np.asarray(image.GetSpacing())[0],
            oversample = False,
            skeletonize = True)


        thickness_image = sitk.GetImageFromArray(thickness)
        thickness_image.SetOrigin(image.GetOrigin())
        thickness_image.SetSpacing(image.GetSpacing())
        thickness_image.SetDirection(image.GetDirection())

        print(f"Writing: {file.replace('.mha','_TBTH.mha')}")
        sitk.WriteImage(thickness_image,file.replace('.mha','_TBTH.mha'))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Interactive script")
    parser.add_argument("--input", default='./*.mha', type=str, help="Path to the files to be transferred")

    #'/home/matthias.walle/data/horse/*/*SEG.mha'
    args = parser.parse_args()
    
    main(args.input)