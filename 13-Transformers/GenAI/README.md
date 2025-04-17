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
├── CasualLM-GTPQ/ (transformer with AutoModelForSeq2SeqLM head, using AutoGTPQ)
│   ├── train_autogtpq.py
│   ├── inference_autogtpq.py
│   └── utils.py
├── config (configuration folder)
└── README.md (The main readme)

> ../../Data/

* Amazon_Review_200.csv
  * use for LMHeadModel and CasualLM.
  * data needs to contain a ``colunm_name`` that could be defined in **config**  with context that use for training.
* Amazon_Review_200_KW.csv 
  * use for ConditionalGeneration and Seq2seq.
  * data needs to contain ``input_feature``(for keywords, separate by , ) and ``label``(for context) that could be defined in **config**.
* dolly-200.csv 
  * use for CasualLM-Dolly and CasualLM-GTPQ.
  * data needs to be dolly format, which contains columns ``instruction``(for questions), ``context`` (for the context that question based) and ``response``(for answers). 