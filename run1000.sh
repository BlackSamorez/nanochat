#!/bin/bash
#SBATCH --account=a140
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a140/containers/megatron.toml
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/blacksamorez/nanochat/logs/debug_%A.out
#SBATCH --error=/iopsstor/scratch/cscs/blacksamorez/nanochat/logs/debug_%A.err

set -euo pipefail

# export QAT_METHOD="bf16"
export QAT_METHOD="quartet_v2"
# export QAT_METHOD="nvidia"
# export QAT_METHOD="46"

export WANDB_RUN="1000-${QAT_METHOD}"

WORKDIR="/iopsstor/scratch/cscs/blacksamorez/nanochat"

# Match your allocation: 4 GPUs per node => 4 processes per node
NPROC_PER_NODE=4

cd "$WORKDIR"

# Master Address Logic
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29500
echo "Master: $MASTER_ADDR:$MASTER_PORT | Nodes: $SLURM_NNODES | GPUs/Node: $NPROC_PER_NODE"

# ---- srun + torchrun wrapper (one torchrun per node; torchrun spawns GPU workers) ----
# SLURM_PROCID is 0..(ntasks-1); with ntasks-per-node=1, it acts as the node_rank. :contentReference[oaicite:1]{index=1}
# Using srun with --ntasks=$SLURM_NNODES and --ntasks-per-node=1 ensures 1 task (launcher) per node. :contentReference[oaicite:2]{index=2}
torchrun_srun () {
    local user_args="$*"

    CMD="cd $WORKDIR && \
        pip uninstall -y torch &&
        pip install uv && \
        uv venv /tmp/venv --allow-existing && \
        UV_PROJECT_ENVIRONMENT=/tmp/venv uv sync --extra gpu && \
        source /tmp/venv/bin/activate && \
        torchrun \
          --nnodes=$SLURM_NNODES \
          --nproc_per_node=$NPROC_PER_NODE \
          --rdzv_id=$SLURM_JOB_ID \
          --rdzv_backend=c10d \
          --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
          --node_rank=\${SLURM_PROCID} \
          ${user_args}
    "

    srun \
      --kill-on-bad-exit=1 \
      --export=ALL \
      --output="/iopsstor/scratch/cscs/blacksamorez/nanochat/logs/%x_%j_%s_t%t.out" \
      --error="/iopsstor/scratch/cscs/blacksamorez/nanochat/logs/%x_%j_%s_t%t.err" \
      --label \
      bash -lc "$CMD"
}

# -------------------------------------------------------------------
# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${SCRATCH}/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# ---- Fix curl CA bundle path (common in containers) ----
# If your environment points curl at a non-existent bundle, override it.
for c in \
  /etc/ssl/certs/ca-certificates.crt \
  /etc/pki/tls/certs/ca-bundle.crt \
  /etc/ssl/ca-bundle.crt \
  /etc/ssl/cert.pem
do
  if [ -r "$c" ]; then
    export SSL_CERT_FILE="$c"
    export CURL_CA_BUNDLE="$c"
    export REQUESTS_CA_BUNDLE="$c"
    echo "Using CA bundle: $c"
    break
  fi
done

pip uninstall -y torch
pip install uv
[ -d "/tmp/venv" ] || uv venv /tmp/venv
UV_PROJECT_ENVIRONMENT="/tmp/venv" uv sync --extra gpu
source /tmp/venv/bin/activate

# System info
echo "NVIDIA-smi"
nvidia-smi

# IMPORTANT for srun-launched shells: export WANDB_RUN so it exists on all nodes/steps
export WANDB_RUN="${WANDB_RUN:-dummy}"

echo "Resetting the report"
python -m nanochat.report reset
wget --no-check-certificate -O "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# train tokenizer on ~4B characters and kick off download of the rest for pretraining
echo "Downloading the data"
# python -m nanochat.dataset -n 16
# python -m nanochat.dataset -n 1200 &
# python -m scripts.tok_train --max-chars=4000000000 --vocab-size=65536
# python -m scripts.tok_eval

# -------------------------------------------------------------------
# distributed runs (ALL torchrun calls now go through srun)
echo "Pre-training"
torchrun_srun -m scripts.base_train -- --depth=32 --target-param-data-ratio=20 --device-batch-size=4 --save-every=10000 --resume-from-step=-1 --run="$WANDB_RUN" --model-tag="$QAT_METHOD"
torchrun_srun -m scripts.base_loss -- --model-tag="$QAT_METHOD"
torchrun_srun -m scripts.base_eval --  --model-tag="$QAT_METHOD"

# midtrain
torchrun_srun -m scripts.mid_train -- --device-batch-size=4 --run="$WANDB_RUN" --model-tag="$QAT_METHOD"
torchrun_srun -m scripts.chat_eval -- -i mid --model-tag="$QAT_METHOD"

# sft
torchrun_srun -m scripts.chat_sft -- --run="$WANDB_RUN" --model-tag="$QAT_METHOD"
torchrun_srun -m scripts.chat_eval -- -i sft --model-tag="$QAT_METHOD"

# generate final report
python -m nanochat.report generate

# talk to it
# python -m scripts.chat_web
