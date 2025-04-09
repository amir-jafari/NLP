#%% --------------------------------------------------------------------------------------------------------------------
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

class CustomLLM(LLM):
    n: int
    """The number of characters from the last message of the prompt to be echoed."""

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
        """Return a dictionary of identifying parameters."""
        return {

            "model_name": "CustomChatModel",
        }
    @property
    def _llm_type(self) -> str:
        return "custom"

#%% --------------------------------------------------------------------------------------------------------------------
llm = CustomLLM(n=5)
print("======= Let's check whether we have created it=======")
print(llm)
#%% --------------------------------------------------------------------------------------------------------------------
llm_invoke_sample = llm.invoke("This is a foobar thing")
print("======= The below is the output of llm_invoke_sample=======")
print(llm_invoke_sample)
#%% --------------------------------------------------------------------------------------------------------------------
llm_batch_sample = llm.batch(["woof woof woof", "meow meow meow"])
print("======= The below is the output of llm_batch_sample=======")
print(llm_batch_sample)

