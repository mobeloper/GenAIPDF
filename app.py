import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import faiss


im = Image.open("favicon.ico")

def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):

    # separator="\n\n",
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks



#Vector Store Database
#TODO:
#it runs locally, it gets destroyed after closing the app
#we need a persistant database to keep this in the cloud
#others:
# https://python.langchain.com/docs/integrations/vectorstores/pinecone

# Facebook AI Similarity Search (Faiss)
def create_faiss_vectorstore(text_chunks):
  
    # create embeddings
    #Reference: Massive Text Embedding Benchmark (MTEB) 
    #https://huggingface.co/spaces/mteb/leaderboard

    #Option 1:
    #using OpenAI (paid) - https://openai.com/pricing
    # Ada v2 $0.0004/1K tokens 
    #text-embedding-3-small	$0.00002 / 1K tokens
    # https://platform.openai.com/docs/models/embeddings
    # https://openai.com/blog/new-embedding-models-and-api-updates
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Option 2:
    #using instructor (free) slow when running in local cpu
    #https://huggingface.co/hkunlp/instructor-xl
    #https://instructor-embedding.github.io/


    #Create vector store
    #vector_store = FAISS.from_text(text_chunks, embeddings)
    vector_store = faiss.FAISS.from_texts(text_chunks, embeddings)

    return vector_store



def main():

    load_dotenv()   #load environment variables

    st.set_page_config(
        page_title="GenAIPDF",
        page_icon=im,
        layout="wide")

    st.header("Chat with your PDFs 😎")
    #st.write(st.__version__)


    name = ''
    if name=='':
        name = st.text_input("Your name?")
    else:
        st.write(f"Hello, {name or 'there!'}!")

    
    st.text_input("Ask question about your PDFs:")

    with st.sidebar:
        st.subheader("Your Documents")

        uploaded_files =  st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if uploaded_files:            
            # for uploaded_file in uploaded_files:            
            #     # st.write(f"{uploaded_file.name} ✔️")
            #     st.write(f"{uploaded_file.name} ✅")
            #     # bytes_data = uploaded_file.read()
            #     # st.write(bytes_data)
            
            #st.download_button(f"Download {uploaded_file.name}", 
            #                   data=uploaded_file, file_name=uploaded_file.name)

            if st.button("Process"):
                with st.spinner("Processing..."):
                    # get pdfs
                    raw_text = get_pdf_text(uploaded_files)
                    # st.write(raw_text)    #display the text extracted from the pdfs                    

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write(text_chunks)

                    # create vector store
                    vector_store = create_faiss_vectorstore(text_chunks)




if __name__ == '__main__':
    main()
