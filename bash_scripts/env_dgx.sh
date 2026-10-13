# Usage:  source env_dgx.sh        → pick a free GPU automatically
#         source env_dgx.sh 3      → force GPU 3
source .venv/bin/activate
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "Using GPU $1 (chosen by hand)"
else
    read -r GPU USED <<< "$(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | tr ',' ' ')"
    if [ "$USED" -lt 1000 ]; then
        export CUDA_VISIBLE_DEVICES=$GPU
        echo "Using GPU $GPU ($USED MiB in use)"
    else
        export CUDA_VISIBLE_DEVICES=""
        echo "No free GPU (least used: GPU $GPU, $USED MiB). Running on CPU only."
    fi
fi