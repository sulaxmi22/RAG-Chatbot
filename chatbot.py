# Import LLM + embedding interfaces, vector store, and the web UI (Gradio).
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Must match the model used during ingestion
EMBED_MODEL = "text-embedding-3-small"  # or "gemini-embedding-001"
embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL)

# LLM from Euri’s catalog
# Options you listed: "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", etc.
llm = ChatOpenAI(temperature=0.5, model="gpt-4.1-mini")

# connect to chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# retriever - Turn the vector store into a retriever: given a query, return top-5 similar chunks.
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# This function runs on every user message. It streams tokens back as they arrive.
def stream_response(message, history):
    # R STEP - Retrieve relevant chunks
    docs = retriever.invoke(message)

    # Merge all chunk texts into one big “knowledge” string for the prompt
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    if message is not None:
        partial_message = ""
        
        # A STEP - Answer using LLM with retrieved context (RAG)
        # Instructs the model to only use retrieved context.
        # Includes current question, chat history, and the retrieved “knowledge.”
        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}
        """

        print(rag_prompt)

        # Stream tokens from Euri model
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# G STEP - call the llm and stream tokens to the UI as they arrive.
# Create a simple Gradio chat UI that uses stream_response as the backend.
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

chatbot.launch()
