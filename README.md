# Sentiment Analysis Using Machine Learning

A Python-based sentiment analysis system that classifies text into **Positive, Negative, and Neutral** categories using TF-IDF feature extraction and Logistic Regression.

## 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) technique used to identify the emotional tone expressed in text. This project implements a supervised machine learning model trained on the TweetEval sentiment dataset to classify text into three sentiment categories.

The model is developed using Google Colab and Scikit-learn. It supports sentiment prediction for new text inputs and provides evaluation metrics to assess classification performance.

## 🎯 Objectives

- Build a machine learning model for sentiment classification.
- Classify text into positive, negative, and neutral sentiments.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Visualize sentiment classification performance.
- Save the trained model for reuse and potential local deployment.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Model development and training |
| Pandas | Dataset processing |
| Scikit-learn | Machine learning and evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Sentiment classification |
| Matplotlib | Performance visualization |
| IPyWidgets | Interactive notebook interface |
| Joblib | Model serialization |
| Hugging Face Datasets | Dataset loading |

## 🧠 Methodology

The project follows these steps:

1. **Dataset loading:** Load the TweetEval sentiment dataset.
2. **Data preparation:** Map numerical sentiment labels to readable categories.
3. **Feature extraction:** Convert text into TF-IDF vectors using unigrams and bigrams.
4. **Model training:** Train a Logistic Regression classifier on labeled training data.
5. **Model evaluation:** Evaluate predictions on a held-out test set.
6. **Sentiment prediction:** Classify new text inputs.
7. **Visualization:** Generate performance charts and sentiment score breakdowns.
8. **Model export:** Save the fitted pipeline using Joblib.

## 🏗️ System Architecture

```text
          Input Text
              |
              v
      Text Preprocessing
              |
              v
       TF-IDF Vectorizer
              |
              v
   Logistic Regression Model
              |
              v
     Sentiment Classification
              |
       +------+------+ 
       |      |      |
       v      v      v
    Positive Neutral Negative
              |
              v
      Display Prediction
```

## 📊 Model Performance

The current model achieved **62.41% test accuracy** on 12,284 examples from the TweetEval test set.

### Classification Report

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.63 | 0.61 | 0.62 | 3,972 |
| Neutral | 0.66 | 0.62 | 0.64 | 5,937 |
| Positive | 0.56 | 0.66 | 0.60 | 2,375 |
| **Accuracy** | | | **0.62** | **12,284** |
| **Macro average** | **0.61** | **0.63** | **0.62** | **12,284** |
| **Weighted average** | **0.63** | **0.62** | **0.62** | **12,284** |

### Performance Analysis

- **Negative sentiment:** F1-score of 0.62, with recall of 0.61.
- **Neutral sentiment:** Highest class-specific F1-score at 0.64.
- **Positive sentiment:** Recall of 0.66, with precision of 0.56.
- **Macro F1-score:** 0.62, reflecting performance across all three classes equally.

These results indicate that the model can identify sentiment patterns, although there is room for improvement.

*Note: Results reflect the current trained model and may vary with changes in preprocessing, model configuration, or data. TweetEval is tweet-based, so performance on product reviews and other domains may differ.*

## 💻 Example Usage

```python
import joblib

model = joblib.load("sentiment_model.joblib")

text = "I absolutely love this product!"
prediction = model.predict([text])[0]

print("Predicted sentiment:", prediction)
```

Possible output:

```text
Predicted sentiment: positive
```

The actual prediction depends on the trained model and input text.

## 📁 Project Structure

```text
Sentiment-Analysis-ML/
│
├── Python_Sentiment_Analyzer.ipynb
├── sentiment_model.joblib
├── sentiment_confusion_matrix.png
├── requirements.txt
└── README.md
```

- `Python_Sentiment_Analyzer.ipynb` — Dataset loading, training, evaluation, and interactive prediction.
- `sentiment_model.joblib` — Saved TF-IDF and Logistic Regression pipeline.
- `sentiment_confusion_matrix.png` — Confusion matrix, if generated.
- `requirements.txt` — Python dependency list.
- `README.md` — Project documentation.

Only include files that are actually present in your repository.

## 🚀 How to Run

### Google Colab

1. Open Google Colab.
2. Upload `Python_Sentiment_Analyzer.ipynb`.
3. Run the installation and dataset-loading cells.
4. Execute the preprocessing and model-training cells.
5. Run the evaluation and prediction cells.
6. Enter custom text in the interactive analyzer.

An internet connection is required to download the dataset and any missing packages. Once the model and dependencies are available, prediction can run locally without cloud inference.

### Local Execution

1. Install a compatible Python version.
2. Install the dependencies listed in `requirements.txt`.
3. Download `sentiment_model.joblib`.
4. Place the model file beside your Python script.
5. Run the prediction script.

Example dependency installation:

```bash
python -m pip install scikit-learn joblib pandas matplotlib ipywidgets datasets
```

For local deployment, use compatible versions of Scikit-learn and Joblib. Saved model compatibility across Scikit-learn versions is not guaranteed.

## 🌟 Key Features

- Three-class sentiment classification.
- TF-IDF-based text feature extraction.
- Logistic Regression machine learning model.
- Evaluation using accuracy, precision, recall, and F1-score.
- Interactive sentiment analysis in Google Colab.
- Visual performance analysis.
- Model saving and reuse through Joblib.
- Lightweight inference without requiring a GPU.

## ⚠️ Limitations

- The current model has moderate classification accuracy.
- Sarcasm, irony, slang, and complex contextual meanings may be misclassified.
- Tweet-based training data may not generalize well to product reviews or formal text.
- Model probability scores are estimates, not guaranteed measures of correctness.
- The current model is designed for English-language sentiment classification.

## 🔮 Future Enhancements

- Improve performance through hyperparameter tuning and cross-validation.
- Train and evaluate on larger, domain-specific review datasets.
- Analyze misclassified examples using a confusion matrix.
- Develop a desktop GUI using Tkinter.
- Add batch sentiment analysis for CSV files.
- Explore Hindi, Marathi, and Hinglish sentiment classification.
- Compare TF-IDF with transformer-based models.

## 👥 Project Team

**Mini-Project:** Sentiment Analysis Using Machine Learning

**Submitted by:**
- Atharv Sachin Kate
- Devyani Prithviraj Patil
- Samruddhi Mahesh Yadav
- Prakarsh Jitendra Shende

**Class:** T.Y. (CSE), Division A

**Project Guide:** Mrs. Swati G. Ambi

## 📜 License

This project is intended for educational and academic purposes. Review the applicable dataset and software licenses before redistributing datasets or trained model artifacts.

---

**Developed as a mini-project on Sentiment Analysis Using Machine Learning.**
