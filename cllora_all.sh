DEVICE=0

DATASETS="cddb domainnet officehome core50"
# DATASETS="domainnet"

for DS in $DATASETS; do
    for ORDER in {1..5}; do
        echo "=== CL-LoRA | ${DS} | order ${ORDER} ==="
        python main.py ./exps/cllora/${DS}.json -order $ORDER -device $DEVICE
    done
done
