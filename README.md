# Genre Classification with Google BERT Model

For this project, I performed genre classification of book blurbs with a fine-tuned Google BERT (Bidirectional Encoder Representations from Transformers) Model. The model achieved an overall classification accuracy of 94% with balanced F1 scores across all genres. :100:

:robot: **Tools & Libraries**: Python, Pandas, Hugging Face Transformers, PyTorch, Google BERT, Datasets, Scikit-learn, Seaborn, Matplotlib

:robot: **Data Source**: U. Hamburg Language Technology Group’s Blurb Genre Collection

:robot: **Model**: google-bert/bert-base-uncased (available on [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased))

## Summary of steps:
1. **Data Preprocessing**: Loaded and processed a dataset of book blurbs, converting genre strings into unique numerical labels for model training. Created a Hugging Face Dataset object for efficient data handling.

2. **Model Fine-Tuning**: Loaded the pre-trained BERT model. Set up label mappings and prepared the dataset for training by tokenizing text data. Split the dataset into training and testing sets. Defined and applied a data collator for dynamic padding.

* Padding: to make sure each sequence in a batch is the same length - add 0's to sequence till max length, attention mask of padded tokens are 0's
* Truncation: to remove tokens from sequences longer than the context window size

3. **Model Training Setup**:: Defined training arguments including batch size, learning rate, number of epochs, warmup steps, and weight decay. Configured metrics (accuracy and F1) to monitor model performance. Initialized the Hugging Face Trainer with the model, tokenizer, data collator, training arguments, datasets, and early stopping callback.

* Batch size: no. training examples/testing examples used in one iteration 
* Learning rate: step size at each iteration while moving towards a min of the loss function in gradient descent
* No. epochs: no. times training dataset will be passed through the model during training
* Warmup steps: help stabilize a model’s final parameters by gradually increasing the learning rate over a set number of steps
* Weight decay: helps prevent overfitted models by keeping model weights from growing too large

> [!NOTE]
> Due to limitations in compute units while training the BERT model, I had to adjust several hyperparameters from the typical range.
> Decreased batch size, no. epochs to shorten memory usage and training time. Increased learning rate to help model converge faster.

4. **Model Training**: Trained the BERT model for 5 epochs with early stopping if the model did not improve for 3 consecutive epochs. Achieved increasing accuracy and F1 scores with each epoch. Saved the final fine-tuned model.

5. **Model Evaluation**: Reloaded the fine-tuned model and tokenizer. Used the Hugging Face pipeline to classify sample texts and evaluate the model’s performance. Generated a classification report and confusion matrix to visualize and assess model accuracy.
