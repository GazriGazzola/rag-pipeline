import streamlit as st
import ollama
from qdrant_client import QdrantClient

# --- Configuration ---
QDRANT_HOST = "http://localhost:6333"
OLLAMA_HOST = "http://localhost:11434"
COLLECTION_NAME = "enterprise_knowledge"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

qdrant_client = QdrantClient(url=QDRANT_HOST, check_compatibility=False)
ollama_client = ollama.Client(host=OLLAMA_HOST)

# --- UI Setup ---
st.title("Enterprise RAG Assistant")
st.caption("Running locally on an RTX 3070.")

# Add a slider to the sidebar for real-time temperature control
temperature_setting = st.sidebar.slider("AI Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Add a reset button to the sidebar
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# --- Memory Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Core Chat Logic ---
if prompt := st.chat_input("Ask about the IT documentation..."):
    # Display the user's question
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Step 1: Embed the user's question
    query_vector = ollama_client.embed(model=EMBED_MODEL, input=prompt)['embeddings'][0]

    # Step 2: Search Qdrant for the closest matching document chunks
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2
    )
    
    retrieved_contexts = [point.payload["text"] for point in search_results.points]
    context_block = "\n---\n".join(retrieved_contexts)

    # Step 3: Manipulate the LLM with the context
    system_prompt = (
        "You are a helpful assistant. Use ONLY the provided Context below to answer the Question.\n"
        "If the context does not contain the answer, say 'I cannot find that in the local database.'\n\n"
        f"Context:\n{context_block}"
    )

    # Step 4: Stream the response back to the UI
    with st.chat_message("assistant"):
        # Inject the memory by combining the system prompt with the chat history
        messages_to_send = [{'role': 'system', 'content': system_prompt}] + st.session_state.messages
        
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=messages_to_send,
            options={'temperature': temperature_setting} # Controlled by the UI slider
        )
        bot_reply = response['message']['content']
        st.markdown(bot_reply)
        
    # Save the AI's response to memory
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})