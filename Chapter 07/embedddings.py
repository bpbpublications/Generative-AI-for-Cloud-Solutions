"""
# Embedding Example
"""
import boto3
import json
 
#Create the connection to Bedrock
bedrock = boto3.client(service_name='bedrock')
 
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
 
# Define prompt and model parameters
prompt_data = "Give me a paragraph on cloud providers"
 
body = json.dumps({
    "inputText": prompt_data,
})
 
model_id = 'amazon.titan-embed-text-v1' #look for embeddings in the modelID
accept = 'application/json' 
content_type = 'application/json'
 
# Invoke model 
response = bedrock_runtime.invoke_model(
    body=body, 
    modelId=model_id, 
    accept=accept, 
    contentType=content_type
)
 
# Print response
response_body = json.loads(response['body'].read())
embedding = response_body.get('embedding')
 
#Print the Embedding
print("Embedding for the prompt: ", prompt_data)
print(embedding)
print("Length of the embedding vector: ", len(embedding))
