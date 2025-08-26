# this file is our knowledge base
from langchain_community.document_loaders import PyPDFDirectoryLoader # to read all PDFs in a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter # to split large documents into chunks
from langchain_openai import OpenAIEmbeddings # to get embeddings from Euri
from langchain_chroma import Chroma # to store vectors locally we use Chroma
from uuid import uuid4 # to create unique IDs for each chunk so DB can reference them

from dotenv import load_dotenv # to load env vars from .env file
load_dotenv() 


# configuration where our data lives and where to store the vector DB
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Picked embedding model available on Euri (both are 1536-d)
EMBED_MODEL = "text-embedding-3-small"  # or: "gemini-embedding-001"

embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL)

# connect to chroma (it will create the directory if it doesn't exist)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load PDFs under data directory
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Split into chunks into ~300 tokens (with 100 token overlap)
# Overlap avoids losing context between chunks boundaries
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
# actually do the splitting get many small documents
chunks = text_splitter.split_documents(raw_documents)

# Unique IDs required by Chroma
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Add to Chroma (embeds via Euri under the hood)
# Critical step: embed the chunks and store vectors + text in Chroma
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"Ingested {len(chunks)} chunks using embedding model: {EMBED_MODEL}")
