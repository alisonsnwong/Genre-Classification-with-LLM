# Genre-Classfication-with-LLM

For this project, I performed genre classification of book blurbs with a fine-tuned Google BERT (Bidirectional Encoder Representations from Transformers) Model. The model achieved an overall classification accuracy of 94% with balanced F1 scores across all genres. 

* **Tools & Libraries**: Python, Pandas, Hugging Face Transformers, PyTorch, Google BERT, Datasets, Scikit-learn, Seaborn, Matplotlib
* **Data Source**: U. Hamburg Language Technology Group’s Blurb Genre Collection
* **Model**: google-bert/bert-base-uncased (available on [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased))

Performed rigorous model training (warmup, early stopping) using Hugging Face Transformers and PyTorch

Summary of steps:
1. **Data Preprocessing**: Loaded and processed a dataset of book blurbs, converting genre strings into unique numerical labels for model training. Created a Hugging Face Dataset object for efficient data handling.

2. **Model Fine-Tuning**:: Loaded the pre-trained BERT model. Set up label mappings and prepared the dataset for training by tokenizing text data. Split the dataset into training and testing sets. Defined and applied a data collator for dynamic padding.

3. **Model Training Setup**:: Defined training arguments including batch size, learning rate, number of epochs, warmup steps, and weight decay. Configured metrics (accuracy and F1) to monitor model performance. Initialized the Hugging Face Trainer with the model, tokenizer, data collator, training arguments, datasets, and early stopping callback.

4. **Model Training**:: Trained the BERT model for 5 epochs with early stopping if the model did not improve for 3 consecutive epochs. Achieved increasing accuracy and F1 scores with each epoch. Saved the final fine-tuned model.

5. **Model Evaluation**:: Reloaded the fine-tuned model and tokenizer. Used the Hugging Face pipeline to classify sample texts and evaluate the model’s performance. Generated a classification report and confusion matrix to visualize and assess model accuracy.
