import os
from tokenizers import ByteLevelBPETokenizer

# Print the current working directory
print("Current working directory (os.getcwd()):", os.getcwd())

# The directory of the running script itself
script_dir = os.path.dirname(os.path.realpath(__file__))
print("Script directory (os.path.realpath(__file__)):", script_dir)

# Construct the full path to corpus.txt inside the script directory
file_path = os.path.join(script_dir, "corpus.txt")
print("Looking for corpus.txt at:", file_path)
print("File exists?", os.path.isfile(file_path))

# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on corpus.txt
tokenizer.train(
    files=file_path,
    vocab_size=2000,
    min_frequency=2
)

# Output directory for tokenizer files
output_dir = os.path.join(script_dir, "my_tokenizer")

# Save the trained tokenizer
print("Saving tokenizer to:", output_dir)
tokenizer.save_model(output_dir)
