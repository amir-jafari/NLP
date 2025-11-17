#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
import inspect
import torch

"""
Compatibility patch:
Some versions of `transformers` call `Accelerator.unwrap_model(..., keep_torch_compile=False)`
which is not present in older `accelerate` versions. To avoid a runtime TypeError,
we detect whether `keep_torch_compile` exists in the signature and, if not, we
monkey‑patch `unwrap_model` to drop that kwarg.
Long-term fix: align package versions (upgrade `accelerate` to a version compatible
with your `transformers`).
"""
try:
    _sig = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" not in _sig.parameters:
        _orig_unwrap = Accelerator.unwrap_model

        def _unwrap_model_compat(self, model, *args, **kwargs):
            kwargs.pop("keep_torch_compile", None)
            return _orig_unwrap(self, model, *args, **kwargs)

        Accelerator.unwrap_model = _unwrap_model_compat  # type: ignore[attr-defined]
except Exception:
    # If anything goes wrong, proceed without the patch; the original error will still surface.
    pass

#
# Additional compatibility patch:
# Some versions of `accelerate` define `AcceleratedOptimizer.train/eval` methods that
# incorrectly forward to the underlying torch optimizer's `.train()`/`.eval()` methods,
# which do not exist (only nn.Module implements train/eval). This leads to:
# AttributeError: 'AdamW' object has no attribute 'train'
# We patch these methods to be safe no-ops.
try:
    if hasattr(AcceleratedOptimizer, "train"):
        def _noop_train(self):
            return self
        AcceleratedOptimizer.train = _noop_train  # type: ignore[attr-defined]
    if hasattr(AcceleratedOptimizer, "eval"):
        def _noop_eval(self):
            return self
        AcceleratedOptimizer.eval = _noop_eval  # type: ignore[attr-defined]
except Exception:
    # If patching fails, continue; training may still work on compatible versions.
    pass

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

#%% --------------------------------------------------------------------------------------------------------------------
data = load_dataset("imdb")
train_data = data["train"].shuffle(seed=42).select(range(2000))
eval_data = data["test"].shuffle(seed=42).select(range(800))

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_data = train_data.map(tokenize, batched=True)
eval_data = eval_data.map(tokenize, batched=True)
# Rename label column to "labels" so Trainer can find it even with PEFT wrappers
train_data = train_data.rename_column("label", "labels")
eval_data = eval_data.rename_column("label", "labels")
# Ensure PyTorch tensors are returned and select the columns Trainer/model expect
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) 
eval_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) 

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, config)

#%% --------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="peft_out",
    num_train_epochs=1,
    per_device_train_batch_size=4,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)
trainer.train()
print(trainer.evaluate())

#%% --------------------------------------------------------------------------------------------------------------------
test_text = "I absolutely loved this movie."
encoded_input = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=16)
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
with torch.no_grad():
    output = model(**encoded_input)
    prediction = torch.argmax(output.logits, dim=-1).item()

print("Text:", test_text)
print("Predicted label:", prediction)