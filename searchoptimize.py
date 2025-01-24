import requests

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
# AIzaSyAAtQB3-EszabbmDUT7V_ml9hHWvss0h8Q new one
# 


def searchModify(query,context ):
    llm=GooglePalm(google_api_key="AIzaSyAF9UND7RFqn5NLbOJAI9zxo4mTxhB65cU")
    llm.temperature=0.2
    print("Context:",context)
    template = [f"""
    <s>[INST] <<SYS>>
                You modify search queries and give relevant and personalised search query suggestions using context
                and strictly generate only from the query text provided below do not generate new query answers.
                Modify and give atleast 4 search modified answers. Apend the modified search answers with this string "https://www.google.com/search?q="
                Embed the modified string in markdown with a title
    <</SYS>>
    Context: {context}
    Query: {query}
    Answer: 
    1.
    2.
    3.
    4."""]
    llm_result= llm._generate(template)

    res=llm_result.generations
    # text= res[0][0].text
    # for char in res[0][0].text:
    #     split= text.split("**")
    print(res[0][0].text.split('\n'))
    return (res[0][0].text)
