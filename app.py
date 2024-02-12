import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import faiss

#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#Custom Html files
from templates.HtmlTemplates import css,bot_template,user_template


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
  
    print("vectorspace and embeddings started...\n")
    # create embeddings
    #Reference: Massive Text Embedding Benchmark (MTEB) 
    #https://huggingface.co/spaces/mteb/leaderboard

    #Option 1:
    #using OpenAI (paid) - https://openai.com/pricing
    # Ada v2 $0.0004/1K tokens 
    # text-embedding-3-small	$0.00002 / 1K tokens
    # https://platform.openai.com/docs/models/embeddings
    # https://openai.com/blog/new-embedding-models-and-api-updates
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    # Option 2:
    #using instructor (free) slow when running in local cpu
    #https://huggingface.co/hkunlp/instructor-xl
    #https://instructor-embedding.github.io/
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    print("embeddings finished!!!")


    # #Create vector store
    # #vector_store = FAISS.from_text(text_chunks, embeddings)
    vector_store = faiss.FAISS.from_texts(text_chunks, embeddings)

    print("vectorstore finished!!!")

    return vector_store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()  #DaVinci
    langchain_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=langchain_memory
    )
    return conversation_chain


def handle_chat_conversation(user_input):

    #Get the AI Conversation Object
    response = st.session_state.conversationObj({'question' : user_input})

    #print(response)

    #update the chat history and display to user
    st.session_state.chat_history = response['chat_history']
 
    for i, message in enumerate(st.session_state.chat_history):

        #even numbers belong to user input messages
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #AI generated messages
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




def main():

    load_dotenv()   #load environment variables

    st.set_page_config(
        page_title="GenAIPDF",
        page_icon=im,
        layout="wide")

    #insert CSS
    st.write(css,unsafe_allow_html=True)


    #initialize the persistant global (session state) variables
    if "conversationObj" not in st.session_state:
        st.session_state.conversationObj = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    #Main Page Content
    st.header("Chat with your PDFs 😎")
    #st.write(st.__version__)

    name=''
    if name=='':
        name = st.text_input("Your name?")
    else:
        st.write(f"Hello, {name or 'there!'}!")

    # Handle user input
    user_input = st.text_input("Ask question about your PDFs:")

    if user_input:
        handle_chat_conversation(user_input)
        


    #Side bar content
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

                    # create conversation chain Object
                    # Will allow to generate new messages of the conversation
                    # Takes the conversation history and returns the next element in the conversation
                    # We make this persistant in the entire app using:
                    # st.session_state.
                    # meaning the variable must not be reinitialize
                    # and we will be able to use this variable in a global scope
                    st.session_state.conversationObj = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
