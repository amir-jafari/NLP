#%% --------------------------------------------------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
#%% --------------------------------------------------------------------------------------------------------------------
class CustomEmbeddings(Embeddings):
    """Simple fake embeddings for demonstration."""

    def __init__(self, size: int = 4):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for documents."""
        return [np.random.rand(self.size).tolist() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate random embedding for query."""
        return np.random.rand(self.size).tolist()

class CustomLLM(LLM):
    n: int
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for char in prompt[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        return "custom"
#%% --------------------------------------------------------------------------------------------------------------------
documents = [
    Document(page_content="LangChain is a framework for building 12-LLM-powered applications."),
    Document(page_content="It helps you chain together different components to create more advanced apps.")
]
embedding_function = CustomEmbeddings(size=4)  # Placeholder
vectorstore = Chroma.from_documents(documents, embedding_function)
retriever = vectorstore.as_retriever()
#%% --------------------------------------------------------------------------------------------------------------------
llm = CustomLLM(n=40)

# Create RAG prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
#%% --------------------------------------------------------------------------------------------------------------------
query = "What is LangChain and how does it help developers?"
response = rag_chain.invoke(query)
print("=== RAG Output with CustomLLM ===")
print(response)
