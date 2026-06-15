# Smart Review Classifier

An end-to-end Machine Learning pipeline that analyzes e-commerce product reviews and classifies their sentiment in real-time. Built using Scikit-learn, Pandas, and Streamlit, this project demonstrates a complete workflow from raw data processing to production-ready deployment.

---

## Features

- **Data Pipeline:** Processes and samples large-scale text data efficiently using Pandas.
- **Feature Engineering:** Implements TF-IDF vectorization with unigrams and bigrams, filtering out English stop words.
- **Ensemble Learning:** Utilizes a soft-voting classifier combining Logistic Regression and Naive Bayes for optimized text classification.
- **Interactive UI:** Provides a clean web interface built with Streamlit for real-time inference.
- **Reproducibility:** Locked random states ensure identical model parameters and evaluation metrics across runs.

---

## Project Structure

- `train.py`: Script to load data, train the ensemble model, evaluate metrics, and serialize artifacts.
- `app.py`: Streamlit application file handling the user interface and real-time predictions.
- `requirements.txt`: List of dependencies required to run the project.
- `vectorizer.pkl`: Serialized TF-IDF vectorizer (generated after training).
- `model.pkl`: Serialized voting classifier model (generated after training).

---

## Installation and Setup

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone [https://github.com/your-username/smart-review-classifier.git](https://github.com/your-username/smart-review-classifier.git)
cd smart-review-classifier

```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 3. Install Dependencies

Install all required libraries using pip:

```bash
pip install -r requirements.txt

```

### 4. Download the Dataset

1. Download the Amazon Fine Food Reviews dataset from Kaggle.
2. Extract the downloaded zip archive.
3. Place the `Reviews.csv` file directly into the root directory of this project.

---

## How to Run

### Step 1: Train the Model

Run the training script to process the dataset, train the ensemble classifier, and generate the necessary model artifacts.

```bash
python train.py

```

Upon completion, the script will output the validation metrics and save `vectorizer.pkl` and `model.pkl` to your project directory.

### Step 2: Launch the Web Application

Start the interactive Streamlit dashboard locally:

```bash
streamlit run app.py

```

This command will launch the server and open the application in your default web browser automatically.

---

## Evaluation and Reproducibility

To ensure consistent performance benchmarking, all pipeline components are initialized with a fixed seed (`random_state=42`). Out-of-sample validation on a 20% holdout set yields the following deterministic metrics:

- **Accuracy:** 91.2%
- **F1-Score:** 0.89

```

```
