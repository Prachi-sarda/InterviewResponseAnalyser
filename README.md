# Team26_InterviewResponseAnalyser
 NLP Final Project


## Overview

This repository contains two Python scripts used for text classification tasks, specifically predicting the quality of interview answers. The quality of each answer is classified as `good`, `average`, or `poor`, based on various features extracted from the answers. The two models employed for classification are:
- **Decision Tree** (`DT.py`)
- **Support Vector Machine (SVM)** (`SVM.py`)

Both models use natural language processing (NLP) techniques to preprocess the text, extract features, and predict the quality of interview answers.
---

## Dependencies

To run these scripts, ensure the following libraries are installed:

- **pandas**: For data manipulation and analysis.
- **nltk**: For natural language processing tasks like tokenization, stopword removal, and lemmatization.
- **textblob**: For sentiment analysis.
- **spacy**: For sentence structure analysis using part-of-speech tagging.
- **scikit-learn**: For machine learning algorithms, including decision trees, SVM, and model evaluation.
- **textstat**: For computing text complexity (Flesch-Kincaid Grade).
- **imblearn**: For handling imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique).

### Installation

You can install the required libraries by running the following command:

pip install pandas nltk textblob spacy scikit-learn textstat imbalanced-learn

Additionally, download the necessary `spacy` language model:


python -m spacy download en_core_web_sm


---

## Files
### 1. `DT.py`

This script performs text preprocessing and classification using a **Decision Tree** algorithm.

#### Key Features:
- **Text Preprocessing**: 
  - Removes non-ASCII characters.
  - Removes punctuation and converts text to lowercase.
  - Tokenizes and removes stopwords.
  - Lemmatizes words.
  
- **Feature Extraction**:
  - Sentiment polarity score.
  - Word count.
  - Sentence structure variability.
  - Text complexity using the Flesch-Kincaid Grade level.
  
- **Model**: 
  - A Decision Tree Classifier is used to predict the answer quality (`good`, `average`, `poor`).
  
#### How it works:
1. The script loads the dataset and preprocesses the answers.
2. Features like sentiment, word count, sentence structure, and text complexity are extracted.
3. A **Decision Tree** classifier is trained using the processed features and evaluated using the classification report.

#### Execution:
python DT.py


---

### 2. `SVM.py`

This script uses a **Support Vector Machine (SVM)** model for text classification. It includes additional steps for handling class imbalance with SMOTE and feature selection using Random Forest.

#### Key Features:
- **Text Preprocessing**: Similar to `DT.py`, including the removal of unwanted characters and stopwords, and lemmatization.
  
- **Feature Extraction**: 
  - Sentiment polarity score.
  - Word count.
  - Text complexity.
  
- **Class Imbalance Handling**: Uses **SMOTE** to balance the dataset before training the model.
  
- **Model**: 
  - A **Support Vector Machine (SVM)** with a linear kernel is used for classification.
  
#### How it works:
1. The dataset is preprocessed, and feature extraction is performed.
2. **Random Forest** is used to identify low-importance features, which are then removed.
3. **SMOTE** is applied to oversample the minority class in the training data.
4. The **SVM model** is trained and evaluated, with the accuracy and classification report displayed.

#### Execution:
python SVM.py
---

## Results

Both models output the following:
1. **Accuracy**: The percentage of correct classifications.
2. **Classification Report**: Precision, recall, and F1-score for each class (`good`, `average`, `poor`).

---

## Improvements and Future Work

- Experiment with different classifiers like Logistic Regression, XGBoost, or Neural Networks.
- Optimize hyperparameters using GridSearch or RandomSearch for better performance.
- Extend the preprocessing pipeline to handle other forms of text input, like contractions or abbreviations.
- Handle more complex datasets with a broader variety of answer types.