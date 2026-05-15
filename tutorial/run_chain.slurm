#!/bin/bash
#SBATCH --job-name=hmfast_chain
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hmfast_chain_%j.out
#SBATCH --error=logs/hmfast_chain_%j.err

set -e

echo "=== hmfast Cobaya chain ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

# Verify environment
python -c "import jax; print('JAX devices:', jax.devices())"

# Run from tutorial directory
cd /scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/tutorial

python run_cobaya_chain.py --max-samples 100000 \
    2>&1 | tee ../logs/hmfast_chain_output_${SLURM_JOB_ID}.log

echo "=== Done at $(date) ==="
