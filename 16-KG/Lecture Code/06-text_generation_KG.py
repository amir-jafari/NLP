#%% --------------------------------------------------------------------------------------------------------------------
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
#%% --------------------------------------------------------------------------------------------------------------------
class KnowledgeAwareNLPModel:
    def __init__(self, model_name: str = "gpt2", lr: float = 1e-5, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        return re.sub(r"[^\w\s]", "", text).strip()

    def prepare_inputs(self, text: str, kg_facts=None) -> dict:
        clean = self.preprocess_text(text)
        if kg_facts:
            facts = " ".join(self.preprocess_text(f) for f in (kg_facts if isinstance(kg_facts, list) else [kg_facts]))
            clean += f"\nRelevant Facts: {facts}\n"
        encoded = self.tokenizer(clean, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(self.device) for k, v in encoded.items()}

    def generate_text(self, prompt: str, kg_facts=None, max_length: int = 50) -> str:
        inputs = self.prepare_inputs(prompt, kg_facts)
        output = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=self.model.config.pad_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def fine_tune_step(self, text: str, kg_facts=None) -> float:
        self.model.train()
        clean = self.preprocess_text(text)
        if kg_facts:
            facts = " ".join(self.preprocess_text(f) for f in (kg_facts if isinstance(kg_facts, list) else [kg_facts]))
            clean += f"\nFacts: {facts}\n"
        data = self.tokenizer(clean, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in data.items()}
        loss = self.model(**inputs, labels=inputs['input_ids']).loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    gen = KnowledgeAwareNLPModel()
    prompt = "In modern NLP pipelines"
    print("\n=== Without KG Facts ===")
    print(gen.generate_text(prompt, max_length=60))
    print("\n=== With KG Facts ===")
    facts = [
        "Transformer models excel at contextual understanding.",
        "Fine-tuning adapts models to specific tasks."
    ]
    print(gen.generate_text(prompt, kg_facts=facts, max_length=60))
    loss = gen.fine_tune_step(
        "NLP stands for Natural Language Processing.",
        kg_facts=[
            "It deals with text and speech.",
            "Applications include translation and sentiment analysis."
        ]
    )
    print(f"\n=== Fine-tune Loss: {loss:.4f} ===")
