## Step 3: Sequence-to-Sequence Model

This model encapsulates the encoder and decoder into a single `Seq2Seq` model. The encoder processes an input sequence and passes its final hidden state to the decoder, which uses it to generate a target sequence. Here's a breakdown:

- **Encoder**: Encodes the input sequence and provides a final hidden state.
- **Decoder**: Initializes with the encoder's hidden state and generates a target sequence, with each output feeding the next input.

**Code Execution**:
Run the `Seq2Seq` model on a sample input to observe the generated sequence shape.

**Expected Output**:
- `Output sequence shape:` Should match `(batch_size, target_len, output_size)`, confirming that the target sequence is generated correctly.

This sets up the base for adding attention in the next step.
