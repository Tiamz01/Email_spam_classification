# Email/SMS Spam Classification Project

Welcome to the SMS Spam Classification project! In this project, we build a machine-learning model to classify SMS messages as either spam or not spam (ham). We've deployed this project using Streamlit, allowing you to interact with the model through a user-friendly web application.

## Table of Contents
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

Here's an overview of the project structure:

```
sms_spam_classification/
│
├── data/
│   └── spams.csv
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── app.py
│
├── requirements.txt
│
├── README.md
│
└── saved_models/
    ├── vectorizer.pkl
    └── model.pkl
```

- **data**: This directory contains the dataset file (`spams.csv`) used for training and testing the models.

- **notebooks**: Jupyter Notebook files for exploratory data analysis and model development can be found here.

- **src**: The source code for the project resides here:

  - `preprocessing.py`: Contains functions for data cleaning and preprocessing.
  - `model.py`: Includes code for building, training, and evaluating machine learning models.
  - `app.py`: The Streamlit web application for model deployment.

- **requirements.txt**: Lists project dependencies for easy replication using `pip install -r requirements.txt`.

- **README.md**: This README file provides an overview of the project, its components, and instructions for use.

- **saved_models**: Directory for storing saved models and related files (e.g., TF-IDF vectorizer).

## Data Preprocessing

Data preprocessing is a crucial step in preparing the dataset for model training. Here's an outline of the preprocessing steps:

- **Import Necessary Libraries**: We import essential Python libraries, including NumPy, Pandas, and NLTK, for data manipulation, analysis, and text processing.

- **Load the Dataset**: We load the SMS dataset (`spams.csv`) into a Pandas DataFrame for further processing.

- **Data Cleaning**: We perform data cleaning tasks such as removing unnecessary columns, renaming columns, and handling duplicate values.

- **Encoding Labels**: The target labels ("ham" and "spam") are encoded using Label Encoding, transforming them into numerical values.

- **Exploratory Data Analysis (EDA)**: EDA involves analyzing the dataset to gain insights, visualize data distributions, and understand characteristics of ham and spam messages. This helps in feature engineering and model selection.

- **Text Preprocessing**: We apply text preprocessing techniques such as lowercasing, tokenization, special character removal, and stemming to clean and normalize the text data.

## Exploratory Data Analysis (EDA)

EDA is a critical phase for understanding the data. Some key EDA steps include:

- Analyzing data distribution: We visualize the distribution of ham and spam messages in the dataset.

- Character, word, and sentence analysis: We compute statistics such as character count, word count, and sentence count for both ham and spam messages.

- Word cloud visualization: Word clouds are generated to visualize the most frequent words in both ham and spam messages.

## Machine Learning Models

We explore various machine learning models for SMS spam classification, including Naive Bayes, Support Vector Machine (SVM), Random Forest, and more. Model evaluation metrics such as accuracy, precision, and confusion matrices are calculated for each model.

## Model Deployment

We've deployed the SMS spam classification model using Streamlit. The web application allows users to input an SMS message and receive an instant classification result—whether it's "ham" or "spam."

## Usage

To run the Streamlit web application and interact with the SMS spam classification model, follow these steps:

1. Install the project dependencies using `pip install -r requirements.txt`.

2. Run the Streamlit application by executing `streamlit run src/app.py` in your terminal.

3. Access the web application in your browser and input an SMS message for classification.

