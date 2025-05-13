import os, sys
import numpy as np
import tensorflow as tf
from ml_peptide_self_assembly.SA_ML_predictive.code.utils import load_data, merge_data, reshape_for_model, convert_list, MAX_LEN, MODELS_PATH, ALL_MODELS
from ml_peptide_self_assembly.SA_ML_predictive.code.automate_training import BATCH_SIZE

# Determine model.
model = "AP_SP"
if len(sys.argv) > 4 and sys.argv[4] in ALL_MODELS:
    model = sys.argv[4]
else:
    if len(sys.argv) <= 4:
        print("No model selected, using", model, "model")
    elif sys.argv[4] not in ALL_MODELS:
        print("Model", sys.argv[4], "does not exist, using", model, "model")

# Read input sequences from file or single sequence.
input_arg = sys.argv[2]
if os.path.isfile(input_arg):
    with open(input_arg, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]
else:
    sequences = [input_arg]

# Filter out sequences that are too long.
valid_sequences = []
for seq in sequences:
    if len(seq) <= MAX_LEN:
        valid_sequences.append(seq)
    else:
        print("Sequence", seq, "is too long, maximum is", MAX_LEN)
if not valid_sequences:
    sys.exit("No valid sequences to process.")

# Load the best model once.
best_model = tf.keras.models.load_model(MODELS_PATH + model + ".h5")
offset = 1
properties = np.ones(95); properties[0] = 0
mask_value = 2

results = []
for seq in valid_sequences:
    try:
        # Append dummy sequence as in original script.
        pep_list = [seq, "A" * MAX_LEN]
        pep_labels = ["1", "1"]
        SA, NSA = load_data(model, [pep_list, pep_labels], offset, properties, mask_value)
        all_data, all_labels = merge_data(SA, NSA)
        test_data, test_labels = reshape_for_model(model, all_data, all_labels)
        pred = best_model.predict(test_data, batch_size=BATCH_SIZE)
        results.append((seq, convert_list(pred)[0]))
    except Exception as e:
        results.append((seq, f"Error: {str(e)}"))

# Save predictions to file.
output_file = "predictions.txt" if len(sys.argv) <= 5 else sys.argv[5]
with open(output_file, "w") as fout:
    for seq, prediction in results:
        fout.write(f"{seq}\t{prediction}\n")
print("Predictions saved to", output_file)