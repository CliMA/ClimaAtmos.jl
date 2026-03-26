#!/bin/bash
#SBATCH --job-name=scm_sensitivity
#SBATCH --output=sensitivity_%j.out
#SBATCH --error=sensitivity_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G

# Parameter Sensitivity Analysis Submission Script
# =================================================
# This script launches the main sensitivity analysis driver.
# Workers are spawned dynamically via ClusterManagers/SlurmManager.
#
# Usage:
#   sbatch submit_sensitivity.sbatch [--config CONFIG_FILE]
#
# The driver process runs on this allocation and manages worker jobs.

set -e

# Parse arguments
CONFIG_FILE="${1:-sensitivity_config.toml}"

echo "========================================"
echo "SCM Parameter Sensitivity Analysis"
echo "========================================"
echo "Start time: $(date)"
echo "Config file: $CONFIG_FILE"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""

# Activate Julia environment
cd /groups/esm/cchristo/clima/copies4/ClimaAtmos.jl/calibration/experiments/gcm_driven_scm

module purge
export MODULEPATH="/resnick/groups/esm/modules:$MODULEPATH"
module load climacommon/2025_07_30

# Run sensitivity analysis
julia --project=. param_sensitivity/run_sensitivity.jl --config "param_sensitivity/$CONFIG_FILE"

echo ""
echo "========================================"
echo "Sensitivity analysis complete"
echo "End time: $(date)"
echo "========================================"
