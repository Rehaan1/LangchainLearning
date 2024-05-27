import requests
import streamlit as st


# def get_openai_response(input_text):
#     url = "http://localhost:8000/essay/invoke"
#     data = {
#         "input": {
#             "topic": input_text
#         }
#     }
#     response = requests.post(url, json=data)
    
#     return response.json()["output"]["content"]


def get_ollama_response(input_text):
    url = "http://localhost:8000/poem/invoke"
    data = {
        "input": {
            "topic": input_text
        }
    }
    response = requests.post(url, json=data)
    
    return response.json()["output"]


st.title("Langchain Demo App")
# input_text = st.text_input("Write an essay on")
input_text_1 = st.text_input("Write a poem on")

# if input_text:
#     st.write(get_openai_response(input_text))

if input_text_1:
    st.write(get_ollama_response(input_text_1))