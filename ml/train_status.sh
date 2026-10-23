#!/bin/bash
MODELS=("yolov8n_canonical_e100" "yolo12n_canonical_e100" "yolo26n_canonical_e100")

# The logs show Ultralytics created a nested "runs/detect" inside "runs/detect"
# We'll check both the nested and the flat path.
BASE_DIRS=("runs/detect/runs/detect/timpla_full" "runs/detect/timpla_full")

echo "--------------------------------------------------------------------------------"
printf "%-25s | %-10s | %-10s | %-10s\n" "Model" "Epoch" "mAP50" "mAP50-95"
echo "--------------------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    FOUND=false
    for BASE in "${BASE_DIRS[@]}"; do
        CSV="$BASE/$MODEL/results.csv"
        if [ -f "$CSV" ]; then
            LAST_LINE=$(tail -n 1 "$CSV")
            EPOCH=$(echo $LAST_LINE | cut -d',' -f1)
            MAP50=$(echo $LAST_LINE | cut -d',' -f8)
            MAP50_95=$(echo $LAST_LINE | cut -d',' -f9)
            printf "%-25s | %-10s | %-10s | %-10s\n" "$MODEL" "$EPOCH" "$MAP50" "$MAP50_95"
            FOUND=true
            break
        fi
    done
    if [ "$FOUND" = false ]; then
        printf "%-25s | %-10s | %-10s | %-10s\n" "$MODEL" "Pending" "N/A" "N/A"
    fi
done
echo "--------------------------------------------------------------------------------"
