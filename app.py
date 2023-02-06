import streamlit as st
import pandas as pd
import os
import time
import openai
import tenacity
import nltk
from keybert import KeyBERT
import matplotlib.pyplot as plt

 

with st.sidebar:
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    col2.image("https://www.digite.com/wp-content/uploads/2020/08/Chat-Bots.png", use_column_width=True)
    st.title("Automated feedback analysis")
    choice=st.radio("Please follow the instructions below",["1. Upload","2. Classification","3. Keyphrase identification","4. Download"])
    st.info("Categorize customers feedback and find out what are the keyphrases mentioned on these topics :sunglasses:")

if os.path.exists("sourcedata.csv"):
    df1 = pd.read_csv("sourcedata.csv", index_col=None)

if os.path.exists("Completions.csv"):
    df2 = pd.read_csv("Completions.csv", index_col=None)

if choice == "1. Upload":
    st.title("Automated Feedback Analayzer")
    st.header('Step1: Uploading feedback in CSV')
    st.caption(":warning: Before uploading, make sure you have prepared the following:")
    checkbox1 = st.checkbox(" Place your feedback under a column and name the column header as: 'Feedback'")
    checkbox2 = st.checkbox(" Create another column on the right and name the column header as: 'Category'")
    checkbox3 =st.checkbox(" Save your file as CSV-UTF8 (Comma Delimited) before uploading ")

    #testing
    # Store the initial value of widgets in session state
    if checkbox1 and checkbox2 and checkbox3:
        file = st.file_uploader("Upload Your Dataset Here",disabled=False)
    else:
        file = None
        st.file_uploader("Upload Your Dataset Here", disabled=True)

    if file is not None:
        st.success("File is uploaded!")

    
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

    pass
    

if choice == "2. Classification":
    st.header('Step2: Categorize your data')
    st.caption("Please wait while we classify your data....")
    # Create a progress bar
    total_rows = df1.shape[0]
    progress_bar = st.progress(0)
    
    # Create an empty list to store the completions
    completions = []
    df1.columns = ['Prompt','Completion'] 
    
    # Loop through each row in the DataFrame
    for index, row in df1.iterrows():
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )  # for exponential backoff
    
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
        def completion_with_backoff(**kwargs):
            return openai.Completion.create(**kwargs)
        
        prompt = row["Prompt"]
        # Send the prompt to the model
        openai.api_key = api_key
        response = completion_with_backoff (
            model= "ada:ft-singapore-polytechnic-2023-01-24-03-28-17",
            prompt= prompt + "\n\nIntent:\n\n",
            max_tokens=5,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[" END"]
        )
        response=response['choices'][0]['text']
        # Append the completion to the completions list
        completions.append(response)
        # Update the progress bar
        progress_bar.progress((index+1)/ total_rows)
    
    # Add the completions list to the DataFrame as a new column
    df1["Completion"] = completions
    st.dataframe(df1)
    # Save a new csv file
    st.success('Congratulations. Classification is complete')
    st.balloons()
    
    #testing
    import random
    value_counts = df1['Completion'].value_counts()

    # Define a function to generate random RGB colors
    def get_random_color():
        return (random.random(), random.random(), random.random())

    # Generate a list of random colors for each bar
    colors = [get_random_color() for i in range(len(value_counts))]

    # Plot the bar chart
    fig, ax = plt.subplots()
    value_counts.plot.bar(color=colors, ax=ax)

    # Show the plot in Streamlit
    st.pyplot(fig)
    
    df1.to_csv("Completions.csv", index=False)
    pass

if choice == "3. Keyphrase identification":
    st.title("Identify the keyphrases")

    #Group data based on values in column "Completion"
    grouped = df2.groupby('Completion')
    
    selected_identifier = st.selectbox('Choose your category:', list(grouped.groups.keys()))  
    
    # Use the selected identifier to select the group
    selected_group = grouped.get_group(selected_identifier)
    
    #testing of keybert
    doc = '. '.join(selected_group["Prompt"].astype(str).values)
    #Using keybert with a sentence transformer
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sentence_model)
    answer =kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 5), stop_words= 'english',use_mmr=True, diversity=0.3)


    #testing
    def display_mind_map(answer):
        # Create the plot
        fig, ax = plt.subplots()
        ax.axis('off')

        # Add the words to the plot
        ax.text(0.5, 0.5, selected_identifier, ha='center', va='center', style='italic',fontsize=20,
        bbox={'facecolor': 'red', 'alpha': 0.5})
        ax.text(0.5, 0.9, answer[0], ha='center', va='center', fontsize=15)
        ax.text(0, 0.6, answer[1], ha='center', va='center', fontsize=15)
        ax.text(0.95, 0.7, answer[2], ha='center', va='center', fontsize=15)
        ax.text(0.45, 0.3, answer[3], ha='center', va='center', fontsize=15)
        ax.text(0.8, 0.2, answer[4], ha='center', va='center', fontsize=15)
        
       
        st.pyplot(fig)

    display_mind_map(answer)

    # Display the selected DataFrame
    st.write(selected_group)
    

    pass

if choice == "4. Download":
    st.header("Download your categorized data below")
    @st.cache
    def convert_df(df2):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df2.to_csv().encode('utf-8')

    csv = convert_df(df2)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Completeclassifications.csv',
        mime='text/csv',
    )

    pass
