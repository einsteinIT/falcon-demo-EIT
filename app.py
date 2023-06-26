# Сделано с любовью от: Einstein | про IT
#
# Import stuff
import os
import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain


# Setup streamlit
st.title('🤩💵 Falcon Demo ')
prompt_user = st.text_input('Plug in your prompt here')


# Token
# Сюда вставлять токен из HuggingFace
huggingfacehub_api_token="Your token here"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingfacehub_api_token

# Setup llm
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.6, "max_new_tokens":500})


# Template
template = """
You are smart and polite artificial intelligence assistant.
You should give answer to any question user could ask you. Your answers should be truthful, detailed and consice.

-----------------------

Question: {question}

Answer:
"""


# Prompt
if prompt_user:
    # Chains
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(question=prompt_user)
    st.write(response)