from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader

from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import streamlit as st
from datetime import datetime

current_year = datetime.now().year


llm_options = {
    "Llama3": Ollama(model="llama3", base_url="http://127.0.0.1:11434"),
    "Mistral": Ollama(model="mistral", base_url="http://127.0.0.1:11434"), 
    "salmatrafi/acegpt:7b": Ollama(model="salmatrafi/acegpt:7b", base_url="http://127.0.0.1:11434"),
}

st.image('Bannière LinkedIn moderne et tech bleu et noir.png')
st.markdown("---")
st.write("Welcome to the RAG Bot – an advanced assistant powered by Retrieval-Augmented Generation (RAG), designed to provide precise, context-aware responses by retrieving and generating information to meet your needs! ")
st.write("For arabic documents , choose salmatrafi/acegpt:7b ")
st.markdown("---")
selected_llm_name = st.selectbox("Choose an LLM", options=list(llm_options.keys()))

llm = llm_options[selected_llm_name]

embed_model = OllamaEmbeddings(
    model =selected_llm_name.split()[0].lower(),
    base_url = "http://127.0.0.1:11434"
)

persist_directory = r"C:\Users\PC\anaconda3\envs\rag_env\Lib\site-packages\chromadb"

vector_store = Chroma(persist_directory=persist_directory, embedding_function = embed_model)

#llm = Ollama(model= "llama3", base_url="http://127.0.0.1:11434")

retriever = vector_store.as_retriever(search_kwargs={"k":5})


#retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

def custom_qa_prompt(context,input):
    if context:
        return f"Voici le contexte extrait du PDF : {context}. Maintenant, réponds à la question : {input}"
    else:
        return f"Je n'ai trouvé aucun contexte pertinent dans le PDF. donne une réponse courte et directesur la question suivante : {input}"

# Wrap the custom_qa_prompt in a PromptTemplate
qa_prompt_template = PromptTemplate(
    input_variables=["input", "context"],
    template=custom_qa_prompt("{input}", "{context}")
)
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt_template)
retrieval_chain = create_retrieval_chain(retriever,combine_docs_chain)


def load_and_process_pdf(pdf_path):
    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents_split = text_splitter.split_documents(documents)

    # Add the split documents to the vector store (Chroma)
    vector_store.add_documents(documents_split)



pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file is not None:
    # Process the uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())

    load_and_process_pdf("temp.pdf")

    st.success("PDF loaded and processed successfully!")

    
question = st.text_input("Type your question!")

if st.button("Get answer!"):
    if question:
        with st.spinner("Generating..."):
            response = retrieval_chain.invoke({"input": question})
        st.write('**Réponse : **')
        st.write(response['answer'])
    else:
        st.write("veuillez entrer une réponse adéquate")

st.markdown("---")
# Footer with dynamic year
st.markdown(
    """
    <style>
    .footer {
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f396d3;
        text-align: center;
        padding: 10px 0;
        font-size: small;
    }
    .footer a {
        color: #007BFF;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <small>
            © 2024 My Streamlit App. 
            <a href="https://github.com/ChaimaeBLM" target="_blank">GitHub</a>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)