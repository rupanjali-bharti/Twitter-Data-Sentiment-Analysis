# Twitter-Data-Sentiment-Analysis
To classify the sentiment of social media text into three categories (Positive, Negative, Neutral) using the BERT (Bidirectional Encoder Representations from Transformers) architecture.

# Methodology: Sentiment Analysis using BERT

## 1. Dataset Selection and Preparation
The project utilizes the "Twitter Entity Sentiment Analysis" dataset from Kaggle.
*   **Data Acquisition:** The dataset contains over 74,000 tweets categorized by sentiment (Positive, Negative, Neutral, and Irrelevant).
*   **Cleaning:** Rows containing 'Irrelevant' labels were removed to focus on sentiment polarity. Missing values (NaNs) were handled by converting all inputs to strings and filling empty cells with blank strings.
*   **Label Encoding:** Categorical labels were mapped to numerical values:
    - Negative → 0
    - Neutral → 1
    - Positive → 2

## 2. Advanced Model Architecture: BERT
The core engine of this project is BERT (Bidirectional Encoder Representations from Transformers), specifically the 'bert-base-uncased' variant.
*   **Bidirectional Context:** Unlike traditional RNNs that process text sequentially, BERT uses Transformers to attend to words on both the left and right of a target word, capturing deeper context.
*   **Pre-trained Knowledge:** The model leverages transfer learning, having been pre-trained on the entire English Wikipedia and BookCorpus (3,300M words).

## 3. The Tokenization Pipeline
Before feeding text into the neural network, it passes through a specialized BERT Tokenizer:
*   **WordPiece Tokenization:** Breaks down unknown or complex words into sub-units (e.g., "embeddings" → "em", "##bed", "##dings") to handle out-of-vocabulary terms.
*   **Input Formatting:** Every input sequence is transformed into:
    - **Input IDs:** Numerical mappings of tokens.
    - **Attention Masks:** Binary masks (1s and 0s) to differentiate actual content from padding.
    - **Special Tokens:** Addition of `[CLS]` for classification tasks and `[SEP]` to mark sentence boundaries.



## 4. Fine-Tuning and Training
The model was fine-tuned specifically for the Twitter domain using the following hyperparameters and techniques:
*   **Optimizer:** AdamW (Adam with Weight Decay) was used to optimize the 110M parameters while preventing overfitting.
*   **Hardware:** Training was conducted on a GPU (CUDA) to handle the intensive matrix multiplications.
*   **Loss Function:** Categorical Cross-Entropy was utilized to measure the disparity between the predicted probability distribution and the actual label.
*   **Mini-Batching:** A batch size of 32 was used to balance training stability and memory efficiency.

## 5. Model Evaluation
The performance of the model was validated using a 20% hold-out test set:
*   **Logit Processing:** The raw outputs (logits) were processed through an Argmax function to determine the final predicted class.
*   **Metrics:** Accuracy, Precision, Recall, and F1-Score were calculated to evaluate the model's robustness across all three sentiment classes.

## 6. User Interface (Deployment)
To make the model accessible, a UI was developed using the Gradio framework:
*   **Input:** Users can input full paragraphs into a multi-line text area.
*   **Inference:** The text is tokenized in real-time and passed through the saved model weights.
*   **Output:** The interface displays the final sentiment category with a public-facing URL tunnel.
