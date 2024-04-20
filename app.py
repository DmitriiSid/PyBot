import os
import time
from pathlib import Path

import numpy as np
import requests
from streamlit_lottie import st_lottie
from tqdm import tqdm
from stqdm import stqdm
import streamlit as st
from faiss import IndexFlatL2
import pickle
import faiss
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
#from sentence_transformers import SentenceTransformer
#from pypdf import PdfReader

st.set_page_config(layout = "wide")

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_file =load_lottie("https://lottie.host/96a838d6-a704-42f7-ad13-7380a543bd58/3FFOc0GGnx.json")


@st.cache_resource
def get_client():
    """Returns a cached instance of the MistralClient."""
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)


CLIENT: MistralClient = get_client()

PROMPT_RES = """
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""

PROMPT = """
Please review the following excerpt from the Python documentation:

---------------------
{context}
---------------------

Based on the provided documentation excerpt, answer the query below. If the excerpt contains all necessary information to respond accurately, provide the answer using only this information. If the excerpt does not contain sufficient details to provide a comprehensive and accurate answer, kindly decline to respond.

Do not use knowledge beyond the provided context unless the query specifically pertains to Python implementation practices that are not covered in the excerpt.

Query: {query}
Answer:
"""


# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to add a message to the chat
def add_message(msg, agent="ai", stream=True, store=True):
    """Adds a message to the chat interface, optionally streaming the output."""
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))


# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)


# Function to stream the response from the AI
def stream_response(response):
    """Yields responses from the AI, replacing placeholders as needed."""
    for r in response:
        content = r.choices[0].delta.content
        # prevent $ from rendering as LaTeX
        content = content.replace("$", "\$")
        yield content


#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Decorator to cache the embedding computation
@st.cache_data
def embed(text: str):
    """Returns the embedding for a given text, caching the result."""
    return CLIENT.embeddings("mistral-embed", text).data[0].embedding


# Function to build and cache the index from PDFs in a directory
# @st.cache_resource
# def build_and_cache_index():
#     """Builds and caches the index from PDF documents in the specified directory."""
#     pdf_files = Path("data").glob("*.pdf")
#     text = ""

#     for pdf_file in stqdm(pdf_files, frontend = True):
#         reader = PdfReader(pdf_file)
#         for page in reader.pages:
#             text += page.extract_text() + "\n\n"

#     chunk_size = 500
#     chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

#     embeddings = np.array([embed(chunk) for chunk in chunks])
#     dimension = embeddings.shape[1]
#     index = IndexFlatL2(dimension)
#     index.add(embeddings)

#     return index, chunks

# Function to build and cache the index from TXTs in a directory

def save_data(index, chunks, index_path, chunks_path):
    """Save the index and chunks to files."""
    faiss.write_index(index, index_path)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

def load_data(index_path, chunks_path):
    """Load the index and chunks from files."""
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        return index, chunks
    else:
        return None, None
    

@st.cache_resource
def build_and_cache_index():
    """Builds and caches the index from PDF documents in the specified directory."""
    index_path = "indexes/index.faiss"
    chunks_path = "indexes/chunks.pkl"
    # Check if data is already saved
    index, chunks = load_data(index_path, chunks_path)
    if index is not None and chunks is not None:
        return index, chunks
    
    txt_files = Path("data").glob("*.txt")
    text = ""
    for txt_file in stqdm(txt_files, frontend=True):
        with open(txt_file, 'r') as file:
            text += file.read()

    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = np.array([embed(chunk) for chunk in tqdm(chunks)])
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    # Save data for future use
    save_data(index, chunks, index_path, chunks_path)
    return index, chunks


# Function to reply to queries using the built index
def reply(query: str, index: IndexFlatL2, chunks):
    """Generates a reply to the user's query based on the indexed PDF content."""
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k=2)
    context = [chunks[i] for i in indexes.tolist()[0]]

    messages = [
        ChatMessage(role="user", content=PROMPT.format(context=context, query=query))
    ]
    response = CLIENT.chat_stream(model="mistral-medium", messages=messages)
    add_message(stream_response(response))


# Main application logic
def main():
    """Main function to run the application logic."""
    is_logged_in = False
    pswd = st.text_input('Enter a password')
    if pswd == os.environ["PASS"]:
        is_logged_in = True
    else: 
        is_logged_in = False 
    if is_logged_in:
        col1, col2,col3 = st.columns((0.2,1.3,1))
        with col2:
            st.header("🤖 PyBot")
            st.subheader("Ask your Python expert on questions regarding coding 🤓")
        with col3:
            st_lottie(lottie_file,width=200, height=150)
        
        
        if st.sidebar.button("🔴 Reset conversation"):
            st.session_state.messages = []

        index, chunks = build_and_cache_index()
        

        for message in st.session_state.messages:
            with st.chat_message(message["agent"]):
                st.write(message["content"])

        query = st.chat_input("Ask something about Python")

        if not st.session_state.messages:
            add_message("Ask me anything!")

        if query:
            add_message(query, agent="human", stream=False, store=True)
            reply(query, index, chunks)
    else:
        st.error("Please, log in using password")


if __name__ == "__main__":
    main()