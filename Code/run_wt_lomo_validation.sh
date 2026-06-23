#!/bin/bash
set -e

SOMNOTATE="/Users/margaridaseabra/somnotate"
BASE="/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data"
FOLD_DIR="$BASE/manifests/wt_lomo_validation"
MODEL_DIR="$BASE/models/wt_lomo_validation"

mkdir -p "$MODEL_DIR"

for train_csv in "$FOLD_DIR"/fold_mouse*_train.csv; do
    base_name=$(basename "$train_csv" "_train.csv")
    test_csv="$FOLD_DIR/${base_name}_test.csv"
    model_path="$MODEL_DIR/${base_name}_model.pickle"

    echo ""
    echo "=========================================="
    echo "Running $base_name"
    echo "Train: $train_csv"
    echo "Test:  $test_csv"
    echo "Model: $model_path"
    echo "=========================================="

    python "$SOMNOTATE/example_pipeline/03_train_state_annotation.py" \
      "$train_csv" \
      "$model_path"

    python "$SOMNOTATE/example_pipeline/04_run_state_annotation.py" \
      "$test_csv" \
      "$model_path"

    python "$SOMNOTATE/example_pipeline/07_compute_state_probabilities.py" \
      "$test_csv" \
      "$model_path"
done

echo ""
echo "All leave-one-mouse-out folds completed."
