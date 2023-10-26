from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

import os
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import numpy as np
from prompts import *
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import requests

# OpenAi Key
#
os.environ["OPENAI_API_KEY"] = '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'

class Topic:                     # Topic class
  def __init__(self, name, description, value = None, mandatory = True):          # constructor
    self.name        = name                                                    #  name
    self.description = description                                             #  description
    self.value       = value                                                   #  value
    self.mandatory   = mandatory

topicsList = [
  Topic("company_name", "Nome da Empresa"),
  Topic("quantity_consenters", "Quantidade de Anuentes"),
  Topic("names_consenters", "Nomes dos Anuentes"),
  Topic("qualifications_consenters", "Qualificações dos Anuentes"),
  Topic("shareholding_consenters", "Participação Societária dos Anuentes"),
  Topic("name_grantee", "Nome do Outorgado"),
  Topic("shareholding_grantee", "Participação Societária do Outorgado"),
  Topic("percentage_grantee","Porcentagem que o Outorgado terá direito"),
  Topic("cliff_time", "Tempo do Período de Cliff")
]

topics = {}                           
for t in topicsList:
  topics[t.name] = t                  

#####################################################################################
from langchain.document_loaders import TextLoader
documents = []

text_path = "base.txt"
loader = TextLoader(text_path)
documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents=documents, embedding=embedding)
chatllm = OpenAI(temperature=0.0, model_name="gpt-4")
retriever = db.as_retriever(
  search_type = 'similarity',
  search_kwargs={"k": 8},
)  

###################################################################################33

qaTemplate = """You are an expert advisor talking to a human named {user}.
You are given the following extracted parts of a database and a question.
Answer the question based solely on the database.
If you don't know the answer, just say "None". Don't try to make up an answer.
Question: {question}
=========
{context}
=========
"""


chats = []                          # chat log
chatMemories = {}                  # chat memories
continueChat = True                # continue chat flag

def askBot(user: str, message: str, verb: bool = True):                      #
  global chatMemories, topics, continueChat                                  # global variables
  remainingTopics = "\n".join([f"- {t.description}" for t in topics.values() if not t.value])
  chats.append({
    'user': user,
    'message': message,
    'time': np.datetime64('now')
  })

  dHighLevel = highLevelProgram(
    message = message,
    topics = remainingTopics
  ).variables()

  if verb:
    for k in dHighLevel.keys():
      print(f">>>> {k}: {dHighLevel[k]}")

  if yesNoToTrueFalse(dHighLevel["isOffensive"]):
    dLocal = inappropriateWordsProgram(message=message).variables()
    message = dLocal['safe']
    message = f"MENSAGEM POTENCIALMENTE OFENSIVA: {message}"


  if user in chatMemories.keys():
    generalPrompt = (f"You are an expert advisor having a conversation with a human named {user}. "
                     f"Always answer in the user's own language. "
                     f"Try, when appropriate, to as for information about the following topics, one at a time. "
                     f"If the user asks for a suggestion, or to maintain the conversation flow, it is ok to talk freely. "
                     f"You do not need to follow the order of the topics: move from one to another as organically as possible."
                     f"\nTopics:\n{remainingTopics}.")
    memory = chatMemories[user]
  else:
    generalPrompt = f"You are an expert advisor and you are talking to a human called {user}. Greet the user politely and explain that your goal is to help them understand the concept of vesting and draw up a contract. Always respond in the user's language."
    if verb:
      print(f"Creating {user} memory")
    memory = ConversationSummaryBufferMemory(
      llm=OpenAI(),
      max_token_limit=10,
      memory_key="chat_history",
      return_messages=True
    )
    chatMemories[user] = memory

  if len(remainingTopics) < 1:
    continueChat = False
    generalPrompt = f"You are an expert advisor having a conversation with a human named {user}. Politely inform the user that you already have all the info needed and end the conversation."

  resultDB = ""
  if yesNoToTrueFalse(dHighLevel["isQuestion"] or dHighLevel["isDoubt"]):
    message = f"(Q?): {message}"
    QA_PROMPT = PromptTemplate(
      template=qaTemplate,
      partial_variables={"user" : user},
      input_variables=["question", "context"]
    )
    qa = RetrievalQA.from_chain_type(
      ChatOpenAI(model_name="gpt-4", temperature=0),
      chain_type="stuff",
      retriever=retriever,
      return_source_documents=True,
      chain_type_kwargs={"prompt": QA_PROMPT}
    )
    r = qa({"query": message})
    if verb:
      print(r["source_documents"][0].page_content)
      print(r["source_documents"][0].metadata['source'])
    # exit()
    resultDB = r["result"]

  if yesNoToTrueFalse(dHighLevel["containsInfo"]):
    getInfoProgram = genGetInfoProgram(topics)
    dInfo = getInfoProgram(
      message = message,
    ).variables()

    for k in dInfo.keys():
      if k not in ['llm', 'logging', 'message']:
        v = str(dInfo[k])
        if not "none" in v.strip().lower():
          topics[k].value = v

  prompt = ChatPromptTemplate.from_messages([
    ("system", generalPrompt),
    ("ai", f"summary of the conversation: {memory.moving_summary_buffer}"),
    ("ai", f"Known about the user:\n name: {user}"),
    ("ai", f"results retrieved from database: {resultDB}"),
    ("human", "{message}"),
  ])
  conversation = LLMChain(
    llm=chatllm,
    prompt=prompt,
    verbose=verb,
    memory=memory
  )
  result = conversation.predict(message=message)

  chats.append({
    'user': 'bot',
    'message': result,
    'time': np.datetime64('now')
  })
  return message, result

###########################################################################

if __name__ == '__main__':
    
    import streamlit as st
    st.title('SAFIE')
    st.write('Sistema de Assistência a Formulação de Contratos de Vesting')
    st.write('O SAFIE é um sistema de assistência a formulação de contratos de vesting. O sistema é capaz de interagir com o usuário, coletar informações e gerar um contrato de vesting.')
    openai_api_key = st.sidebar.text_input('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')

    def generate_response(input_text):
        st.info(askBot(user='Edilson', message=input_text, verb=False))

    with st.form('form_SAFIE'):
        text = st.text_area('Enter text:', 'Olá, sou o SAFIE, seu assistente pessoal')
        submitted = st.form_submit_button('Enviar')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            generate_response(text)

    for k, v in topics.items():
      print(
        f"({k}) {v.description} : {v.value}"
      )      
