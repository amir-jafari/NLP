import pandas as pd
from peft import PeftModel
from utils import cfg, load_tokenizer, load_4bit_model, generate

# Run train_qlora.py first: it saves the LoRA adapters to model_path
if __name__ == "__main__":
    tokenizer = load_tokenizer(cfg['model_name'])
    base_model = load_4bit_model(cfg['model_name'])

    df = pd.read_csv(cfg['file_path']).sample(3, random_state=1)
    questions = [("What is a polygon?", None)] + list(zip(df['instruction'], df['context']))

    print("=" * 30, "4-bit base model (no fine-tuning)", "=" * 30)
    base_answers = [generate(base_model, tokenizer, q, c) for q, c in questions]

    # Put the trained LoRA adapters on top of the same 4-bit base model
    model = PeftModel.from_pretrained(base_model, cfg['model_path'])
    for (q, c), base in zip(questions, base_answers):
        print(f"\nQ: {q}")
        print(f"  base      : {base[:300]}")
        print(f"  fine-tuned: {generate(model, tokenizer, q, c)[:300]}")
