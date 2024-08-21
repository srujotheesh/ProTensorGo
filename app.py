import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
def simple_tokenize(text):
    return text.split()

def preprocess_text(text):
    tok = simple_tokenize(text)
    stop_words = set(stopwords.words('english'))
    symbols = set(string.punctuation)
    cleaned_tokens = [word for word in tok if word.lower() not in stop_words and word not in symbols]
    return cleaned_tokens


# Cache the data loading
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Get statistical answer
def get_statistical_answer(df, column_name, statistic):
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
        elif statistic == 'box':
            return generate_box_plot(df, column_name)
        else:
            return "Statistic not recognized."
    else:
        return "Column not found."

# Plot functions
def generate_histogram(df, column):
    plt.hist(df[column], bins=10, alpha=0.7, color='blue')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)

def generate_box_plot(df, column):
    if column in df.columns:
        plt.boxplot(df[column].dropna())
        plt.title(f'Box plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Values')
        st.pyplot(plt)
    else:
        st.write(f"Column '{column}' not found in the DataFrame.")

def generate_line_plot(df, column):
    plt.plot(df[column])
    plt.title(f'Line plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    st.pyplot(plt)

# Answer the question
def answer_question(df, question):
    if len(question) > 2:
        generate_scatter_plot(df, question[0], question[2])
    else:
        result = get_statistical_answer(df, question[1], question[0])
        st.write(f"Answer: {result}")

# Streamlit app layout
st.title("CSV Statistical Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.dataframe(df)

    question_input = st.text_input("Enter your question (format: 'statistical_term column_name')")

    if st.button("Get Answer"):
        if question_input:
            question = preprocess_text(question_input)
            answer_question(df, question)
        else:
            st.write("Please enter a valid question.")
