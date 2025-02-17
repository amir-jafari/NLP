import torch
from captum.attr import KernelShap
from transformers import BertTokenizer, BertForSequenceClassification

# Load a small, fine-tuned spam model (already trained on SMS spam data).
model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

text_input = "Professor Amir's NLP lecture - Explainability code example!"

# Convert text to token IDs
input_ids = tokenizer.encode(text_input, add_special_tokens=True, return_tensors="pt")

seq_len = input_ids.size(1)
feature_mask = torch.arange(seq_len).unsqueeze(0)  # shape: [1, seq_len]

def forward_func(ids):
    # Create an attention mask for non-PAD tokens
    attention_mask = (ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=attention_mask)
    # Return probabilities (for spam vs. ham)
    return torch.softmax(outputs.logits, dim=-1)

##############################################################################
#         KERNEL SHAP (Captum) EXPLANATION
##############################################################################
# KernelShap is a model-agnostic approach that samples perturbed inputs.
# 'baselines' sets a default reference (e.g., all PAD tokens).
##############################################################################

shap = KernelShap(forward_func)

baselines = torch.full_like(input_ids, tokenizer.pad_token_id)

# Generate SHAP attributions
# - 'n_samples' controls how many perturbed samples to use
shap_values = shap.attribute(
    inputs=input_ids,
    baselines=baselines,
    feature_mask=feature_mask,
    target=1,       # spam label
    n_samples=300
)

# Convert tokens and retrieve attribution scores
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
attr_scores = shap_values[0].tolist()

# Show each token’s SHAP contribution
for tok, score in zip(tokens, attr_scores):
    print(f"{tok} => {score:.4f}")
