from transformers import pipeline
from transformers.utils import logging

logging.set_verbosity_error()

# "summarization" was removed in transformers v5, so we ask a small chat model instead
summarizer = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

text = """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers."""
messages = [{"role": "user", "content": f"Summarize this text in 2-3 sentences:\n{text}"}]
print(summarizer(messages, max_new_tokens=100)[0]["generated_text"][-1]["content"])
