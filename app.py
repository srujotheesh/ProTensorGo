import pandas as pd
import numpy as np
import io
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text by removing stopwords and symbols
def preprocess_text(text):
    tok = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    symbols = set(string.punctuation)
    cleaned_tokens = [word for word in tok if word.lower() not in stop_words and word not in symbols]
    return cleaned_tokens

# Function to perform statistical operations
def get_statistical_answer(column_name, statistic):
    if column_name in df.columns:
        data_column = df[column_name]

        if statistic == 'mean':
            return data_column.mean()
        elif statistic == 'max':
            return data_column.max()
        elif statistic == 'min':
            return data_column.min()
        elif statistic == 'std':
            return data_column.std()
        elif statistic == 'histogram':
            return generate_histogram(df, column_name)
        elif statistic == 'line':
            return generate_line_plot(df, column_name)
        else:
            return "Statistic not recognized."
    else:
        return "Column not found."

# Functions to generate plots
def generate_histogram(df, column):
    plt.hist(df[column], bins=10, alpha=0.7, color='blue')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def generate_scatter_plot(df, col1, col2):
    plt.scatter(df[col1], df[col2], alpha=0.7)
    plt.title(f'Scatter plot of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

def generate_line_plot(df, column):
    plt.plot(df[column])
    plt.title(f'Line plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.show()

# Function to answer the question
def answer_question(question):
    if len(question) > 2:
        result = generate_scatter_plot(df, question[0], question[2])
    else:
        result = get_statistical_answer(question[1], question[0])
    return result

# Streamlit input
st.title("Statistical Analysis App")
path = st.text_input("Enter your dataset path.")
if path:
    df = pd.read_csv(path)
    st.write("Data Preview:")
    st.dataframe(df.head())

    question_input = st.text_input("Enter your question (e.g., 'min Age line'): ")
    if question_input:
        question = preprocess_text(question_input)
        answer = answer_question(question)
        st.write("Answer:", answer)

