import os
import time
import streamlit as st

from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

# load env vars
load_dotenv()

# OpenAI api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_response(query):

    # check if storage already exists
    if not os.path.exists('storage'):
        # load documents from the 'data' directory
        documents = SimpleDirectoryReader('data').load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir='storage')     # simplest way to store your indexed data
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir='storage')
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    if response is None:
        st.error("Oops! No result found")
    else:
        st.success(response)


# Define a simple Streamlit app
st.title("DocExtractor - LlamaIndex - RAG")
query = st.text_input("What would you like to ask?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            get_response(query)
        except Exception as e:
            st.error(f"An error occurred: {e}")
