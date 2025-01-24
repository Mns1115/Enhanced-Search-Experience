import requests

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

def query_google_API(context, query ):
    llm=GooglePalm(google_api_key="AIzaSyAF9UND7RFqn5NLbOJAI9zxo4mTxhB65cU")
    llm.temperature=0.2

    template = [f"""
    <s>[INST] <<SYS>>
    Confine your answer within the given context and do not generate the next context.
    Answer truthful answers, don't try to make up an answer.
    <</SYS>>
    Context: YOUR_CONTEXT_HERE
    The YOUR_CONTEXT_DESCRIPTION_HERE.
    Question: YOUR_QUESTION_HERE
    Answer: YOUR_ANSWER_HERE
    ***
    Context: {context}
    Question: {query}
    Answer: """]
    llm_result= llm._generate(template)

    res=llm_result.generations
    print(res[0][0].text)
    return (res[0][0].text)

def query_llama2_EP(context, query, endpoint_url):
    # Define the template for the query
    template = f"""
    <s>[INST] <<SYS>>
    Confine your answer within the given context and do not generate the next context.
    Answer truthful answers, don't try to make up an answer.
    <</SYS>>
    Context: YOUR_CONTEXT_HERE
    The YOUR_CONTEXT_DESCRIPTION_HERE.
    Question: YOUR_QUESTION_HERE
    Answer: YOUR_ANSWER_HERE
    ***
    Context: {context}
    Question: {query}
    Answer: """
    
    print(template)
    
    # Define headers for the request
    headers = {
        'Authorization': 'YOUR_AUTHORIZATION_KEY_HERE',
        'Content-Type': 'application/json',
    }
    
    # Define configuration parameters for the request
    config = {
        "max_new_tokens": 200,
        "temperature": 0.01,
        "return_full_text": False,
        "early_stopping": False,
        "stop_sequence": "***",
        "do_sample": True,
        "top_p": 0.9,
        "num_return_sequences": 1
    }
    
    # Define the JSON data for the request
    json_data = {
        'inputs': template,
        'parameters': config
    }
    
    # Make the request to the provided endpoint
    response = requests.post(endpoint_url, headers=headers, json=json_data)
    
    # Extract the desired information from the response
    result = response.text.split("\":\"")[1].split("\"}]")[0]
    print(result)
    
    return result