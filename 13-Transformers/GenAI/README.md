Sample Code for Different Head from Transformers
============================
> Folder structure
> 
### Top-level directory layout
> ./Transformers/GenAI

├── LMHeadModel/ (transformer with LMHeadModel head)
│   ├── train_LMHead.py (training with huggingface trainer)
│   ├── train_classic_LMHead.py (classic training process using gradient calculation)
│   ├── inference_LMHead.py (inference)
│   └── utils.py
├── CasualLM/ (transformer with AutoModelForCausalLM head)
│   ├── train_causalLM.py
│   ├── inference_causalLM.py
│   └── utils.py
├── ConditionaGeneration/ (transformer with ConditionaGeneration head)
│   ├── train_condi_gen.py
│   ├── train_classic_condi.py
│   ├── inference_condi.py
│   └── utils.py
├── seq2seq/ (transformer with AutoModelForSeq2SeqLM head)
│   ├── train_seq2seq.py
│   ├── train_classic_seq2seq.py
│   ├── inference_seq2seq.py
│   └── utils.py
├── CasualLM-Dolly/ (transformer with AutoModelForSeq2SeqLM head, for the Dolly format data)
│   ├── train_dolly.py
│   ├── inference_dolly.py
│   └── utils.py
├── CasualLM-QLoRA/ (4-bit quantized Qwen2.5 + LoRA adapters (bitsandbytes + peft), for the Dolly format data)
│   ├── train_qlora.py
│   ├── inference_qlora.py
│   └── utils.py
├── LLM_API/ (AWS Bedrock: Claude Sonnet 4.5, Llama 3, Titan embeddings, multi-agent demo; keys in config/config.ini)
├── RAG/ (PDF RAG with Titan embeddings + FAISS + Claude Sonnet 4.5; Streamlit UI_Demo.py)
├── config (configuration folder)
└── README.md (The main readme)

> ./Data/

* Amazon_Review_200.csv
  * use for LMHeadModel and CasualLM.
  * data needs to contain a ``colunm_name`` that could be defined in **config**  with context that use for training.
* Amazon_Review_200_KW.csv 
  * use for ConditionalGeneration and Seq2seq.
  * data needs to contain ``input_feature``(for keywords, separate by , ) and ``label``(for context) that could be defined in **config**.
* dolly-200.csv 
  * use for CasualLM-Dolly and CasualLM-QLoRA.
  * data needs to be dolly format, which contains columns ``instruction``(for questions), ``context`` (for the context that question based) and ``response``(for answers). 

> How to run
* Every script reads `config/config.ini` (the `.env` file sets `CONFIG_FILE = config.ini`); paths are relative, so run each script **from its own folder**, e.g. `cd LMHeadModel && python train_LMHead.py`.
* Run `train_*.py` before `inference_*.py` (inference loads the fine-tuned model from `NLG_results/`).
