"""
# Chatbot for Sports goods retail website
"""

import streamlit as st
import boto3 
import json
import uuid
import os
import base64
import pathlib
from PIL import Image as PILImage
from botocore.exceptions import NoCredentialsError

#Constants
APP_TITLE = "Welcome to XYZ Sports Depot"

#Bedrock
# Assuming your AWS credentials are stored in the default location (~/.aws/credentials)
session = boto3.Session(profile_name='default')
# Create a Bedrock client
bedrock_client = boto3.client('bedrock')
# Create a Bedrock runtime
bedrock_runtime = boto3.client('bedrock-runtime')
#Model ids
#Check if you have the right models in your account
claude_sonnet_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
titan_text_model_id="amazon.titan-text-lite-v1"

#Initialize variables
prompt, image_path, guardrailId = "",  "", ""

#Resize the image
def resize_img(b64imgstr, size=(256, 256)):
    buffer = io.BytesIO()
    img = base64.b64decode(b64imgstr)
    img = PILImage.open(io.BytesIO(img))

    rimg = img.resize(size, PILImage.LANCZOS)
    rimg.save(buffer, format=img.format)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

#Convert the image to base64
def img2base64(image_path,resize=False):
    with open(image_path, "rb") as img_f:
        img_data = base64.b64encode(img_f.read())
    if resize:
        return resize_img(img_data.decode())
    else:
        return img_data.decode()

##Create Bedrock Guardrail with Denied Topics - Finance, Policy
##Content Filters - 
def create_guardrail():
    unique_id = str(uuid.uuid4())[:4]
    response = bedrock_client.create_guardrail(
        name="retail-chatbot-guardrail-{}".format(unique_id),
        description="Only respond to the retail product questions",
        topicPolicyConfig={
            'topicsConfig': [
                  {
                      'name': 'Finance',
                      'definition': "Statements or questions about finances, transactions or monetary advise.",
                      'examples': [
                          "What are the cheapest rates?",
                          "Where can I invest to get rich?",
                          "I want a refund!"
                      ],
                      'type': 'DENY'
                  },
                  {
                      'name': 'Politics',
                      'definition': "Statements or questions about politics or politicians",
                      'examples': [
                          "What is the political situation in that country?",
                          "Give me a list of destinations governed by the greens"
                      ],
                      'type': 'DENY'
                  },
             ]
        },
        contentPolicyConfig={
              'filtersConfig': [
                  {
                      "type": "SEXUAL",
                      "inputStrength": "HIGH",
                      "outputStrength": "HIGH"
                  },
                  {
                      "type": "VIOLENCE",
                      "inputStrength": "HIGH",
                      "outputStrength": "HIGH"
                  },
                  {
                      "type": "HATE",
                      "inputStrength": "HIGH",
                      "outputStrength": "HIGH"
                  },
                  {
                      "type": "INSULTS",
                      "inputStrength": "HIGH",
                      "outputStrength": "HIGH"
                  },
                  {
                      "type": "MISCONDUCT",
                      "inputStrength": "HIGH",
                      "outputStrength": "HIGH"
                  },
                  {
                      "type": "PROMPT_ATTACK",
                      "inputStrength": "HIGH",
                      "outputStrength": "NONE"
                  }
              ]
        },
        wordPolicyConfig={
            'wordsConfig': [
                {
                    'text': 'SeaScanner'
                },
                {
                    'text': 'Megatravel Deals'
                }
            ],
            'managedWordListsConfig': [
                {
                    'type': 'PROFANITY'
                }
            ]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {
                'type': 'AGE',
                'action': 'ANONYMIZE'
                },
            ]
        },
        blockedInputMessaging="Sorry, I cannot respond to this. I can recommend you sports related products and answer your questions about these.",
        blockedOutputsMessaging="Sorry, I cannot respond to this. I can recommend you sports related products and answer your questions about these.",
    
    )

    guardrailId = response["guardrailId"]
    #print("The guardrail id is",response["guardrailId"])
    return guardrailId

#Invoke the bedrock titan model
def call_bedrock_titan_model_with_guardrails(user_input):    
    input_body = {
        "inputText": user_input
    }
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(input_body),
        trace="ENABLED",
        guardrailIdentifier= guardrailId,
        guardrailVersion= "DRAFT"
    )
    output_body = json.loads(response["body"].read().decode())
 
    action = output_body["amazon-bedrock-guardrailAction"]
    if action == "INTERVENED":
        print("Guardrail Intervention: {}".format(json.dumps(output_body["amazon-bedrock-trace"]["guardrail"], indent=2)))
    return output_body["results"][0]["outputText"]

#Invoke the claude sonnet model with text and image prompts
def invoke_claude_sonnet_multi(prompts, image_paths):
    text_prompts = []
    image_prompts = []
    for p in prompts:
        text_prompts.append( {"type": "text", "text": p})
    for ip in image_paths:
        if ip:
            ext = pathlib.Path(ip).suffix[1:]
            if ext == 'jpg':
                ext = 'jpeg' #Validation
            base64_string = img2base64(ip)
            image_prompts.append({"type": "image", "source": {"type": "base64","media_type": f"image/{ext}","data": base64_string}})

    body = json.dumps({"anthropic_version": "bedrock-2023-05-31","max_tokens": 4096, "temperature": 1.0, "messages": [ {"role": "user", "content": text_prompts + image_prompts}]})
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=claude_sonnet_model_id, accept=accept, contentType=contentType
    )

    response_body = json.loads(response.get("body").read())
    #st.write("Response from the model")
    #st.write(response_body)
    return response_body.get("content")[0]["text"]

#Invoke the Cloude sonnet model
def invoke_claude_sonnet(prompt, image_path):
    prompts = [prompt]
    image_paths = [image_path]
    return invoke_claude_sonnet_multi(prompts,image_paths)

##Streamlit app

# Create a title for the app
st.title(APP_TITLE)

st.text("Ask me about our sports products. I can also make recommendations. Let's chat.")

# Add a horizontal line with custom styles
st.markdown("""
    <hr style="height:2px;border-width:0;color:gray;background-color:gray">
""", unsafe_allow_html=True)
        
#Prompt to combine with the image
prompt = "What would you like to know"
text_prompt = st.text_input(prompt)

#Create a file uploader
uploaded_image = st.file_uploader("Optionally upload an image along with your query", type=["jpg", "png", "jpeg"])

# Add a horizontal line with custom styles
st.markdown("""
    <hr style="height:2px;border-width:0;color:gray;background-color:gray">
""", unsafe_allow_html=True)

# Call the function to display the uploader
save_path=""
result=""
if text_prompt != "":
    if uploaded_image is not None:
        #print("Handle the Text and Image prompt")
        #Save the uploaded image to a file
        save_path = os.path.join("uploaded_images", uploaded_image.name)
        print('save_path ' + save_path)
        with open(save_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        #Display the uploaded image
        st.image(PILImage.open(uploaded_image), caption="Image to Analyze", use_column_width=True)
        result = invoke_claude_sonnet(text_prompt,save_path)
    else:
        #print("Handle Text prompt")
        print("guardrailId : " , guardrailId)
        if guardrailId == "":
            guardrailId = create_guardrail()
        result = call_bedrock_titan_model_with_guardrails(text_prompt)

#Display the results
if result != "":
    st.write(result)

