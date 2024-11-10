import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss  # Import faiss for direct index saving/loading

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("News Research Tool 📈")
st.sidebar.title("Article URLs")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_path = "faiss_index.index"
docs_path = "docs.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)

    # Save docs to a pickle file
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

    # Create embeddings and build FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save the FAISS index to a file using faiss
    faiss.write_index(vectorstore_openai.index, index_path)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_path) and os.path.exists(docs_path):
        # Load the FAISS index from file
        index = faiss.read_index(index_path)

        # Load the saved documents
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        # Recreate the vector store from the index and documents, including the embedding function and InMemoryDocstore
        vectorstore = FAISS(
            index=index,
            docstore=InMemoryDocstore({i: doc for i, doc in enumerate(docs)}),  # Use InMemoryDocstore
            index_to_docstore_id={i: i for i in range(len(docs))},
            embedding_function=OpenAIEmbeddings()  # Add the embedding function
        )

        # Set up the chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        
        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
