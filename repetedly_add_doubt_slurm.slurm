#!/bin/bash


#SBATCH --job-name=ront_train_resnet
#SBATCH --output=/vol/scratch/rontsibulsky/slurm_outputs/slurm_%j.out # redirect stdout
#SBATCH --error=/vol/scratch/rontsibulsky/slurm_outputs/slurm_%j.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
use_wandb=""
give_feedback=""
num_iterations=""
model_id=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --use-wandb) use_wandb="$2"; shift 2;;
        --give-feedback) give_feedback="$2"; shift 2;;
        --num-iterations) num_iterations="$2"; shift 2;;
        --model-id) model_id="$2"; shift 2;;

        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

cmd="python repetedly_add_doubt.py"
if [[ -n "$use_wandb" ]]; then cmd+=" --use_wandb $use_wandb"; fi
if [[ -n "$give_feedback" ]]; then cmd+=" --give_feedback $give_feedback"; fi
if [[ -n "$num_iterations" ]]; then cmd+=" --num_iterations $num_iterations"; fi
if [[ -n "$model_id" ]]; then cmd+=" --model_id $model_id"; fi


# Run the command
echo "Running command: $cmd"
eval "$cmd"