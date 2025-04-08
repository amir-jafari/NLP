#%% --------------------------------------------------------------------------------------------------------------------
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import FakeEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from typing import Any, Dict, Iterator, List, Optional
#%% --------------------------------------------------------------------------------------------------------------------
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
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="It helps you chain together different components to create more advanced apps.")
]
embedding_function = FakeEmbeddings(size=4)  # Placeholder
vectorstore = FAISS.from_documents(documents, embedding_function)
retriever = vectorstore.as_retriever()
#%% --------------------------------------------------------------------------------------------------------------------
llm = CustomLLM(n=40)
rag_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,)
#%% --------------------------------------------------------------------------------------------------------------------
query = "What is LangChain and how does it help developers?"
response = rag_chain.run(query)
print("=== RAG Output with CustomLLM ===")
print(response)
