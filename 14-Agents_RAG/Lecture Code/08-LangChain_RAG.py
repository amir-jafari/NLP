#%% --------------------------------------------------------------------------------------------------------------------
import boto3
import configparser
from langchain_community.document_loaders import PyPDFLoader
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
#%% --------------------------------------------------------------------------------------------------------------------
parser = configparser.ConfigParser()
parser.read("/tmp/pycharm_project_488/14-Agents_RAG/Lecture Code/config.ini")
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=parser["BedRock_LLM_API"]["aws_access_key_id"],
    aws_secret_access_key=parser["BedRock_LLM_API"]["aws_secret_access_key"],
    aws_session_token=parser["BedRock_LLM_API"]["aws_session_token"]
)
#%% --------------------------------------------------------------------------------------------------------------------
def create_index(path="The-Life-of-Abraham-Lincoln_page2.pdf"):
    data_load = PyPDFLoader(path)
    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=100,
        chunk_overlap=10
    )
    data_embedding = BedrockEmbeddings(
        client=client,
        # There are limited model_id can be used as embedding model.
        # GWU AWS Bedrock only supports "amazon.titan-embed-text-v2:0"
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embedding,
        vectorstore_cls=FAISS
    )
    db_index = data_index.from_loaders([data_load])
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
    response = index.query(question, llm)
    return response

response = rag_response(create_index(), "When was Lincoln born?")
print(response)