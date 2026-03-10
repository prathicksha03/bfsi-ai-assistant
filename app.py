import streamlit as st
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


with open("bfsi_alpaca_dataset_.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df["query"] = df["instruction"] + " " + df["input"]



embed_model = SentenceTransformer("all-MiniLM-L6-v2")

dataset_embeddings = embed_model.encode(df["query"].tolist())



def find_best_match(user_query):

    query_embedding = embed_model.encode([user_query])

    similarity = cosine_similarity(query_embedding, dataset_embeddings)

    best_index = np.argmax(similarity)
    best_score = similarity[0][best_index]

    return best_index, best_score



model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_llm_response(query):

    inputs = tokenizer(query, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=150
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def bfsi_assistant(query):

    index, score = find_best_match(query)

    if score > 0.75:
        return df.iloc[index]["output"]
    else:
        return generate_llm_response(query)

st.title("🏦 BFSI AI Assistant")

st.write("Banking Customer Support Chatbot")

query = st.text_input("Ask your banking question:")

if st.button("Submit"):

    if query:

        response = bfsi_assistant(query)

        st.success(response)
        