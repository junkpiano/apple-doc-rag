from pathlib import Path
import contextlib
import sys
import os

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader, TextLoader
from llama_cpp import Llama
from bs4 import BeautifulSoup

# === STEP 1: Load raw text files (Apple Docs, WWDC transcripts, Swift Evolution) ===
loader = DirectoryLoader(
    path="./docs",
    glob="**/*",
    loader_cls=lambda p: (
        BSHTMLLoader(p) if Path(p).suffix == ".html" else TextLoader(p)
    )
)

all_docs = loader.load()

# === STEP 2: Split text into chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(all_docs)

# === STEP 3: Create vector embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# === STEP 4: Load local LLM (RakutenAI 7B, for example) ===
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def custom_llm(prompt):
    with suppress_stdout_stderr():
        llm = Llama.from_pretrained(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            filename="mistral-7b-instruct-v0.1.Q2_K.gguf",
            n_ctx=2048,
            n_threads=os.cpu_count(),
            n_batch=64,
        )
        output = llm(prompt, max_tokens=1024, stop=None)
        llm.close()
        return output["choices"][0]["text"].strip()

# === STEP 5: Retrieval-Augmented QA ===
retriever = vectorstore.as_retriever()

def ask_question(question):
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are an iOS expert. As a mentor for learners,
please answer the question from them based on the {context}.
"""
    return custom_llm(prompt)

# === Example usage ===
if __name__ == "__main__":
    query = input("Ask a question about SwiftUI: ")
    answer = ask_question(query)
    print("\nAnswer:\n", answer)
