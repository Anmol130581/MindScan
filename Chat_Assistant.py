import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def chat_bot(user_messages):
    system_prompt = (
        "You are MindCare AI, a kind and supportive emotional wellness companion. "
        "You provide warm, non-judgmental advice and emotional support. "
        "You are not a therapist. Suggest coping strategies, self-care ideas, and kindly encourage professional help if needed."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += user_messages

    response = openai.ChatCompletion.create(
        model="gpt-4o",  
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    return response['choices'][0]['message']['content']
