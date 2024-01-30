#!/bin/bash

submit_jobs() {
    
    python /home/matthias.walle/software/github/ImageGuidedStrain/igstrain/mha2dicom.py
    
    # Find all *FAIM.dcm files, sort them, and store them in an array
    mapfile -t dcm_files < <(find . -name "*.dcm" | sort)

    # The first file is the baseline, remove the './' from the path
    baseline_file="${dcm_files[0]}"
    baseline_file="${baseline_file#./}"  # Remove './'
    baseline_path="${baseline_file%.dcm}"  # Remove '.dcm'

    echo baseline_file
    
    # Loop through all the follow-up files
    for i in "${!dcm_files[@]}"; do
        # Skip the first file since it's the baseline
        if [ "$i" -eq 0 ]; then
            continue
        fi

        # Current follow-up file, remove the './' from the path
        followup_file="${dcm_files[$i]}"
        followup_file="${followup_file#./}"  # Remove './'
        followup_path="${followup_file%.dcm}"  # Remove '.dcm'

        # Generate pFIRE config file
        /home/matthias.walle/software/github/ImageGuidedStrain/igstrain/generate_config.sh "${baseline_path}" "${followup_path}"

        # Create a job script for the follow-up
        job_script=$(mktemp job_script.XXXXXX)

        # Write the job script
        cat <<EOF > "${job_script}"
#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

dcm_dir=\$(pwd)

# Run the Apptainer container with the generated configuration
apptainer run --bind \${dcm_dir}:/data --pwd /data /home/matthias.walle/software/github/ImageGuidedStrain/igstrain/dvc_recompiled.sif mpirun -np 1 pfire ${followup_path}.cfg

python /home/matthias.walle/software/github/ImageGuidedStrain/igstrain/calculate_thickness.py "./*SEG.mha"
python /home/matthias.walle/software/github/ImageGuidedStrain/igstrain/demons_folder.py dcm_dir

EOF

        # Submit the job script
        sbatch "${job_script}"

        # Remove the temporary job script
        rm -f "${job_script}"
    done
}

# Call the function to submit jobs
submit_jobs