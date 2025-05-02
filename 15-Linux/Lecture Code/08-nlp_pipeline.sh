# ----------------------------------------------------------------------------------------------------------------------
set -euo pipefail
# ----------------------------------------------------------------------------------------------------------------------
# 0. Defaults (override with flags if desired)
RAW_CORPUS="small_corpus.txt"
MODEL="bert-base-uncased"
EPOCHS=3
# ----------------------------------------------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL=$2;   shift 2;;
    --epochs)       EPOCHS=$2;  shift 2;;
    --corpus)       RAW_CORPUS=$2; shift 2;;
    *) echo "Unknown flag: $1"; exit 1;;
  esac
done
# ----------------------------------------------------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_DIR="nlp_pipeline"
DATA_DIR="$ROOT_DIR/data"
RUN_DIR="$ROOT_DIR/models/run_$TIMESTAMP"
REPORT_DIR="$ROOT_DIR/reports"
mkdir -p "$DATA_DIR" "$RUN_DIR" "$REPORT_DIR"

echo -e "\n=== NLP Pipeline started at $TIMESTAMP ==="
echo "Corpus : $RAW_CORPUS"
echo "Model  : $MODEL"
echo "Epochs : $EPOCHS"
echo "--------------------------------------------------"
# ----------------------------------------------------------------------------------------------------------------------
# 1. Pre-processing
echo -e "\n[1/3] Pre-processing ..."
cp "$RAW_CORPUS" "$DATA_DIR/raw.txt"

python preprocess.py \
        --input  "$DATA_DIR/raw.txt" \
        --train  "$DATA_DIR/train.jsonl" \
        --test   "$DATA_DIR/test.jsonl" \
        --test-split 0.1
# ----------------------------------------------------------------------------------------------------------------------
# 2. Training / fine-tuning
echo -e "\n[2/3] Training ..."
python train.py \
        --train_file  "$DATA_DIR/train.jsonl" \
        --model_name  "$MODEL" \
        --output_dir  "$RUN_DIR" \
        --epochs      "$EPOCHS"
# ----------------------------------------------------------------------------------------------------------------------
# 3. Evaluation
echo -e "\n[3/3] Evaluation ..."
python evaluate.py \
        --model_dir   "$RUN_DIR" \
        --test_file   "$DATA_DIR/test.jsonl" \
        --metrics_out "$REPORT_DIR/metrics_${TIMESTAMP}.json"

echo -e "\n✅  Pipeline finished.  Trained model in  $RUN_DIR"
echo "   Evaluation metrics saved to $REPORT_DIR/metrics_${TIMESTAMP}.json"
