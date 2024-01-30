from igstrain.demons import main
import SimpleITK as sitk
from glob import glob
import os
from types import SimpleNamespace
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse


def process_folder(folder):

    path = os.path.join(folder,'*SEG.mha')
    files = sorted(glob(path))
    
    for k, moving_image in enumerate(files[1:]): #1
        try:
            name = moving_image.replace('.mha', '_demons_strain.vti')

            # Perform demons registration
            args = {
                "fixed_path":files[0],
                "moving_path":moving_image,
                "map": moving_image.replace('_SEG.mha','_IM.xdmf.h5'),
                #"map":None,
                "dicom":files[0].replace('_SEG.mha','_IM.dcm'),
                "name":name,
                "iterations":100, #100
                "scaling_factors":[8,4,2,1], #
                "update_field_sigma":3,
                "deformation_field_sigma":5,
                "additional_data":files[0].replace('.mha','_TBTH.mha')
            }

            data = main(SimpleNamespace(**args))

            if True:
                fig, axes= plt.subplots(1,4,figsize=(8,2.5))
                axes[0].imshow(data['demons_eff'][:,:,100])
                axes[0].set_title('Effective Strain')
                axes[1].imshow(data['demons_strainxx'][:,:,100])
                axes[1].set_title('Strain XX')
                axes[2].imshow(data['demons_strainyy'][:,:,100])
                axes[2].set_title('Strain YY')
                axes[3].imshow(data['demons_strainzz'][:,:,100])
                axes[3].set_title('Strain ZZ')
                fig.suptitle(f'{k+1} percent')
                for ax in axes:
                    ax.axis('off')
                    cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.6, label='')

                plt.tight_layout()
                name = os.path.join(folder,f'{os.path.basename(folder)}_{k+1}_percent_combined.png')
                plt.savefig(name)  # You can change the extension to other formats like 'jpg', 'pdf', etc.
                plt.show()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive script for performing demons registration with user-provided inputs.")
    parser.add_argument("--folder", default='./', type=str, help="Path to the working dir")
    args = parser.parse_args()
    process_folder(args.folder)