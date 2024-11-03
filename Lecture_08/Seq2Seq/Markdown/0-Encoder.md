"""
# Simple RNN for Sequence Output
In this step, we demonstrate a simple Recurrent Neural Network (RNN) that processes an input sequence and outputs a sequence of the same length.
This foundational example helps illustrate how RNN layers handle sequential data before we progress into more complex Seq2Seq structures.

## Model Architecture
The model consists of:
1. **RNN Layer**: Takes the input sequence and computes hidden states at each time step.
2. **Fully Connected Layer**: Maps the RNN hidden states to the desired output size at each time step.

## Hyperparameters
- `input_size`: Dimension of each input vector in the sequence (e.g., embedding size).
- `hidden_size`: Number of hidden units in the RNN.
- `output_size`: Dimension of the output vector at each time step.
- `seq_len`: Length of the input sequence (e.g., number of words in a sentence).
- `batch_size`: Number of sequences processed in parallel.

## Expected Output
The model takes an input sequence of shape `(batch_size, seq_len, input_size)` and produces an output sequence of shape `(batch_size, seq_len, output_size)`.
"""