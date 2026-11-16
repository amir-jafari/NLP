# QLoRA: fine-tune a 4-bit quantized LLM on Dolly-format data
QLoRA = **Q**uantized base model (4-bit, bitsandbytes) + **LoRA** adapters (peft).
The big model is frozen and compressed; only a few million adapter weights are trained.

`pip install bitsandbytes peft accelerate`

- **create_dataset.py** (optional)
  - downloads the full *dolly-15k* dataset to `../Data/dolly-15k_train.csv`. The course uses the 200-row sample `../Data/dolly-200.csv`.
- **train_qlora.py**
  - loads `Qwen/Qwen2.5-0.5B-Instruct` in 4-bit, adds LoRA adapters, trains them with Trainer, saves the adapters.
- **inference_qlora.py**
  - loads the 4-bit base model, then the adapters on top, and compares base vs fine-tuned answers.
- **utils.py**
  - 4-bit loading (`BitsAndBytesConfig`), LoRA setup (`LoraConfig`), Dolly -> chat-template tokenization (prompt masked with -100), and generation.
- Settings (model, LoRA rank, learning rate, ...) are in the `[casualLM_QLoRA]` section of `../config/config.ini`.
