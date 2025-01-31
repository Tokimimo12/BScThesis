#!/bin/bash
#SBATCH --time=20:30:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

# Activate the virtual environment
source $HOME/BScThesis/BscThesisvenv/bin/activate

# Install the requirements from requirements.txt
pip install --no-cache-dir -r $HOME/BScThesis/Thesis/requirements.txt

# Run the preprocessing script
# python modelAudio.py
# python modelText.py
# python modelMDREFinal.py
# python original.py
# python SingleEncoderModelText.py

python main2.py
