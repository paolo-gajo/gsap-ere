#!/usr/bin/env bash
#SBATCH --job-name=gsap-lora-full
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -euo pipefail

export TRAINING_REGIME=full-pass
exec bash "${SLURM_SUBMIT_DIR:?submit this job from the repository root}/lora_pilot.sbatch"
