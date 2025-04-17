# Q&A for Dolly format data
- run the **create_dataset** file first
  - the dataset is *dolly-15k* data that saved in the dataset.load_dataset, run **create_dataset** could saved it into csv format
- train_dolly.py
  - contains the training process with Trainer.
- inference_dolly.py
  - contains the inference for the text generation.
- utils.py
  - contains the train features preparation, tokenizer and model. 
  - changes needed if you want change the model.