# rag_example.py
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.chains import RetrievalQA
from langchain_core.docstore.document import Document
from langchain_core.vectorstores import FAISS
from langchain_core.embeddings.openai import OpenAIEmbeddings

class CustomLLM(LLM):
    n: int  # The number of characters to echo

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
        return {"model_name": "CustomChatModel"}

    @property
    def _llm_type(self) -> str:
        return "custom"


def main():
    # 1) Create example docs
    texts = [
        "Deep Lake is an awesome vector store. It can also do multi-modal data.",
        "Pinecone is another SaaS-based vector database solution.",
        "LangChain helps build LLM applications and includes RAG modules."
    ]
    docs = [Document(page_content=t) for t in texts]

    # 2) Build vector store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # 3) Custom LLM
    llm = CustomLLM(n=10)

    # 4) Build a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # 5) Query the chain
    query = "What is Pinecone?"
    result = qa_chain.run(query)
    print("Answer from RAG + Custom LLM:\n", result)

if __name__ == "__main__":
    main()
