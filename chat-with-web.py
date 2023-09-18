import os 
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# load enviroment variables from .env file (Optional)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    # set the title and subtitle of the app
    st.title('Ch:red[a]t-:red[with]-W:red[e]b 💬')
    st.subheader('Input your w:red[e]bsite URL, ask question and receive answers directly from the w:red[e]bsite')

    with st.sidebar:
        url = st.text_input("Insert The website URL")
        st.button('Process')

    prompt = st.text_input("Ask a question (query/prompt)")

    if st.button("Submit Query",type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()

        #Split the loaded data 
        text_splitter= CharacterTextSplitter(separator='\n',
                                             chunk_size=1000,
                                             chunk_overlap=100)
        
        docs = text_splitter.split_documents(data)

        # create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings()

        # create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=docs,
                                         embedding=openai_embeddings,
                                         persist_directory=DB_DIR)
        
        vectordb.persist()

        # create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k":3})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        # create a retrivealqa from the model and retriver
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever)

        #run the prompt and return the response
        response = qa(prompt)
        st.write(response)

if __name__ == '__main__': 
    main()

        