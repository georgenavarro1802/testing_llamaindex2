import base64
import os
import streamlit as st

from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

# load env vars
load_dotenv()

# OpenAI api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# Function to convert file content into a downloadable link
def get_download_link(filename):
    with open(filename, "rb") as f:
        # Read the file in binary mode
        file_content = f.read()
    # Encode the file content in base64
    b64 = base64.b64encode(file_content).decode()
    # Set the correct MIME type for PDF
    href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(filename)}">Download</a>'
    return href


def get_response(query):

    # check if storage already exists
    if not os.path.exists('storage'):
        loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
        documents = loader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir='storage')
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir='storage')
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    if response is None:
        st.error("Oops! No result found")
    else:
        # In our Streamlit app (print response and file ref)
        st.success(response)
        file_path = f"{response.source_nodes[0].metadata['file_path']}"
        download_url = get_download_link(file_path)

        # Using markdown to create a neat download link
        filename_to_display = os.path.basename(file_path)
        download_text = f"[{filename_to_display}]({download_url})"
        st.markdown(f":sunglasses: Reference: {download_text}", unsafe_allow_html=True)


# Define a simple Streamlit app
st.title("DocExtractor - LlamaIndex - RAG")
query = st.text_input("What would you like to ask?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            # apply rag
            get_response(query)

        except Exception as e:
            st.error(f"An error occurred: {e}")


# # # Testing block
# # check if storage already exists
# if not os.path.exists("storage"):
#     # load the documents and create the index
#     print('Loading ...')
#     documents = SimpleDirectoryReader("data").load_data()
#     print('Indexing ...')
#     index = VectorStoreIndex.from_documents(documents)
#     print('Storing ...')
#     index.storage_context.persist("storage")
# else:
#     print('Loading Index from Storage ...')
#     storage_context = StorageContext.from_defaults(persist_dir="storage")
#     index = load_index_from_storage(storage_context)
#
# # Query Engine (either way we can now query the index)
# print('Querying ...')
# query_engine = index.as_query_engine()
# question = "What is Principle of PIV Technique?"
# print(f"Question:\n {question}")
# response = query_engine.query(question)
# print(f"Response:\n {response}")
# print(f"Reference:\n {response.source_nodes[0].metadata['file_path']}")
