# Seq2Seq Model with Encoder-Decoder Architecture - Step 2 in Seq2Seq Lecture Series

"""
# Seq2Seq Model with Encoder-Decoder
This step introduces the Seq2Seq (Sequence-to-Sequence) model with an Encoder-Decoder architecture.
The encoder-decoder model is the backbone of many translation, summarization, and conversational models.

## Model Architecture
1. **Encoder**: An RNN processes the input sequence and summarizes it into a final hidden state, which serves as context for the decoder.
2. **Decoder**: A separate RNN uses the encoder's context (final hidden state) to generate the output sequence step by step.

## Key Components
- **Encoder RNN**: Processes the input sequence and returns its hidden state.
- **Decoder RNN**: Takes the encoder's final hidden state as its initial hidden state and generates an output sequence.
- **Fully Connected Layer**: Maps the decoder’s hidden states to the output vocabulary size (for sequence generation).

## Expected Output
The model takes an input sequence and produces a target output sequence. This step provides the structure needed for tasks such as translation and summarization.
"""