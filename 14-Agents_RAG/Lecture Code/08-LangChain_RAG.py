#%% --------------------------------------------------------------------------------------------------------------------
import boto3
import configparser
from langchain_community.document_loaders import PyPDFLoader
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#%% --------------------------------------------------------------------------------------------------------------------
parser = configparser.ConfigParser()
parser.read("../../14-Agents_RAG/Lecture Code/config.ini")
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=parser["BedRock_LLM_API"]["aws_access_key_id"],
    aws_secret_access_key=parser["BedRock_LLM_API"]["aws_secret_access_key"],
    aws_session_token=parser["BedRock_LLM_API"]["aws_session_token"]
)
#%% --------------------------------------------------------------------------------------------------------------------
def create_index(path="The-Life-of-Abraham-Lincoln_page2.pdf"):
    # Load PDF
    data_load = PyPDFLoader(path)
    documents = data_load.load()

    # Split documents
    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=100,
        chunk_overlap=10
    )
    split_docs = data_split.split_documents(documents)

    # Create embeddings
    data_embedding = BedrockEmbeddings(
        client=client,
        # There are limited model_id can be used as embedding model.
        # GWU AWS Bedrock only supports "amazon.titan-embed-text-v2:0"
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )

    # Create Chroma vector store
    db_index = Chroma.from_documents(
        split_docs,
        data_embedding,
        persist_directory="./chroma_db"
    )
    return db_index

def get_llm():
    llm = ChatBedrock(
        client=client,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-east-1"
    )
    return llm

def rag_response(index, question):
    llm = get_llm()
    retriever = index.as_retriever()

    # Create RAG prompt template
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    return response

response = rag_response(create_index(), "When was Lincoln born?")
print(response)