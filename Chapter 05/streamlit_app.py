import streamlit as st
from langchain.llms import OpenAI
import boto3
import json

mixtral_model_id = 'mistral.mixtral-8x7b-instruct-v0:1'
claude_model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
bedrock_runtime_client = boto3.client('bedrock-runtime')

st.title('LLM Playground')

def call_mixtral(prompt):
    instruction = f"<s>[INST] {prompt} [/INST]"
    body = {
        "prompt": instruction,
        "max_tokens": 200,
        "temperature": 0.5,
    }

    response = bedrock_runtime_client.invoke_model(
        modelId=mixtral_model_id, body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())
    outputs = response_body.get("outputs")

    completions = [output["text"] for output in outputs]

    return completions

def call_claude_3(prompt):

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 0.5
    }

    response = bedrock_runtime_client.invoke_model(
        modelId=claude_model_id, body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())

    return response_body['content'][0]['text']

with st.form('my_form'):
    model = st.selectbox(
        "Select model",
        ("Mistral", "Claude")
    )
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if model == 'Mistral':
            t = call_mixtral(text)
        else:
            t = call_claude_3(text)
        st.write(t)