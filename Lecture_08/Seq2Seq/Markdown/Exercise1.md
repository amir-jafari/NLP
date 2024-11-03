# Neural Machine Translation Exercise
## Overview
In this exercise, you will implement a basic Neural Machine Translation (NMT) system that translates English sentences to Spanish. You'll work with sequence-to-sequence (Seq2Seq) architecture and implement an attention mechanism.

## Learning Objectives
- Understand the components of a neural machine translation system
- Implement attention mechanism in a Seq2Seq model
- Learn about vocabulary building and data preprocessing
- Practice working with PyTorch's nn.Module
- Gain hands-on experience with teacher forcing and model training

## Prerequisites
- Basic understanding of Python
- Familiarity with PyTorch
- Understanding of basic neural network concepts
- Knowledge of RNNs (Recurrent Neural Networks)

## Required Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
```

## Project Structure
The exercise is divided into several key components:
1. Data Preprocessing
2. Vocabulary Building
3. Dataset Creation
4. Model Architecture
   - Encoder
   - Attention
   - Decoder
   - Seq2Seq
5. Training Loop
6. Translation Function

## Tasks Overview

### Task 1: Vocabulary Implementation
Complete the Vocabulary class to handle word-to-index and index-to-word conversions.
- Implement vocabulary building from sentences
- Create methods for converting sentences to indices and back

### Task 2: Dataset Creation
Implement the TranslationDataset class that will prepare data for the model.
- Create proper data loading mechanisms
- Handle padding and sequence length

### Task 3: Attention Mechanism
Implement the attention mechanism that helps the model focus on relevant parts of the input sequence.
- Calculate attention scores
- Implement attention weights computation
- Create context vector generation

### Task 4: Encoder Implementation
Build the encoder part of the Seq2Seq model.
- Implement embedding layer
- Create bidirectional GRU
- Handle hidden state processing

### Task 5: Decoder Implementation
Create the decoder that will generate the translated sequence.
- Implement the decoder's forward pass
- Integrate attention mechanism
- Handle teacher forcing

### Task 6: Training Loop
Implement the training procedure.
- Create training step function
- Implement evaluation function
- Add checkpoint saving

## Grading Criteria
- Code functionality (40%)
- Implementation completeness (20%)
- Code organization and style (15%)
- Documentation (15%)
- Performance optimization (10%)

## Bonus Challenges
1. Implement beam search for better translations
2. Add support for batched translations
3. Implement bidirectional encoder
4. Add dropout for regularization
5. Implement learning rate scheduling

## Testing Your Implementation
We provide test cases to verify your implementation:
```python
test_sentences = [
    "hello",
    "thank you",
    "how are you",
    "good morning",
    "i love programming"
]
```

Expected outputs should be coherent Spanish translations.

## Resources
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Attention paper: "Neural Machine Translation by Jointly Learning to Align and Translate"
- Seq2Seq tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

## Submission Requirements
1. Completed Python file with all TODOs implemented
2. Brief report (max 2 pages) explaining:
   - Your implementation choices
   - Challenges faced
   - Performance analysis
   - Possible improvements
3. Training logs showing loss curves
4. 5 example translations with your model


