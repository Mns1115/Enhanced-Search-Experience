import requests
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import DirectoryLoader
import chromadb
import gradio as gr
from llama import query_llama2_EP, query_google_API
from searchoptimize import searchModify

# Hugging Face api token
HUGGINGFACEHUB_API_TOKEN = "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"

def initialize_embeddings():
    model_identifier = "sentence-transformers/all-mpnet-base-v2"
    print(">>>Embeddings setup completed successfully<<<")
    return HuggingFaceEmbeddings(model_name=model_identifier)

def process_and_embed_docs(dir_path, hf_model):
    chroma_instance = chromadb.Client()
    doc_loader = DirectoryLoader(dir_path)
    loaded_docs = doc_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = splitter.split_documents(loaded_docs)
    database = Chroma.from_documents(documents=split_docs, embedding=hf_model)
    print(">>>Embedding and chunking process completed successfully<<<")
    return database

def concatenate_documents(document_list):
    combined_content = "".join([doc.page_content for doc in document_list])
    print(">>>Few-shot prompting process completed successfully<<<")
    print(">>>Prompt engineering process completed successfully<<<")
    return combined_content

hf = initialize_embeddings()

# Replace the path below with the path to your dataset
example_path = "docs"
db = process_and_embed_docs(example_path, hf)
db2=process_and_embed_docs("internet", hf)
endpoint = 'YOUR_ENDPOINT_URL_HERE'


def process_query(query):
    retrieved_docs = db.similarity_search(query)
    # print(retrieved_docs)
    sources= set()
    for docs in retrieved_docs:
      sources.add(docs.metadata['source'])
    print(sources)
    sourceout="Answers are found from the following source files:\n"
    for src in sources:
        sourceout+=src+"\n"
    combined_context = concatenate_documents(retrieved_docs)
    # print(combined_context)
    answer = query_google_API(combined_context, query)
    return answer.replace("\\n", "\n"), sourceout,combined_context

def process_internet(query):
    # retrieved_docs = db2.similarity_search(query)
    # # print(retrieved_docs)
    # sources= set()
    # for docs in retrieved_docs:
    #   sources.add(docs.metadata['source'])
    # print(sources)
    # sourceout="Answers are found from the following source files:\n"
    # for src in sources:
    #     sourceout+=src+"\n"
    # combined_context = concatenate_documents(retrieved_docs)
    # # print(combined_context)
    combined_context=""
    answer = searchModify(query,combined_context)
    return answer
# print(process_query("which college is divij, manasa are studying in?"))

# demo = gr.Interface(
#     fn=process_query,
#     inputs="textbox",
#     outputs=["textbox","textbox"],
#     title="Knowledge Search"
# )




with gr.Blocks() as demo:
    gr.Markdown('''
                # Flip from knowledge to internet search.
                ''')
    with gr.Tab("Knowledge Search"):
        knowledge_input = gr.Textbox(label="Input")
        knowledge_output0 = gr.Textbox(label="Output")
        knowledge_output1 = gr.Textbox(label="Sources")
        context= gr.Textbox(label="Context")
        knowledge_button = gr.Button("Search")
        knowledge_button.click(process_query, inputs=knowledge_input, outputs=[knowledge_output0,knowledge_output1,context])

        
    with gr.Tab("Internet Search"):
        with gr.Row():
            internet_input = gr.Textbox(label="Search Query")            
            # internet_output = gr.Textbox(label="Output")
            internet_output= gr.Markdown("""
                #OUTPUT

            """)
        internet_button = gr.Button("Search")

        internet_button.click(process_internet, inputs=internet_input, outputs=internet_output)
 
# demo.launch(share=True)


demo.launch()
