#!/bin/bash
#SBATCH --job-name=spm
#SBATCH --time=02:00:00
#SBATCH --output=out_%j.log
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "[SLURM] Job starting on $(hostname)"
date

# --- Setup environment ---
# bash ~/Scrivania/my_project/setup_env.sh

cd ~/my_project
source .venv/bin/activate

echo "[ENV] Python version:"
python --version
echo "[ENV] Torch and NumPy:"
python -c "import torch, numpy; print('Torch', torch.__version__, '| CUDA', torch.version.cuda, '| NumPy', numpy.__version__)"
echo "[ENV] CUDA available:", $(python -c "import torch; print(torch.cuda.is_available())")

# --- Run experiment ---
echo "[RUN] Starting run_spm.py..."
python examples/run_spm.py

echo "[DONE] Completed run:"
date
