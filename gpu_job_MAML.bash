#!/bin/bash

#SBATCH --job-name=you_job_name
#SBATCH --output=your_job_name.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH -p jonfan
#SBATCH --gres gpu:1

ml python/3.6.1
ml py-pytorch/1.4.0_py36
ml py-matplotlib/3.2.1_py36
ml py-pandas/1.0.3_py36
ml py-scikit-learn/0.19.1_py36 
ml py-numpy/1.18.1_py36
ml py-tensorflow/2.1.0_py36
ml py-keras/2.3.1_py36
python3 MAML.py
#python3 custom_loop.py
#python3 test_conv.py
