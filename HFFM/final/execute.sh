
SCRIPT_DIR="/projects/academic/alipour/payamabd/HFFM/scripts"

for script in "$SCRIPT_DIR"/*.slurm.script.sh; do
    echo "Submitting: $script"
    sbatch "$script"
done