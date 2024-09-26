import os
import logging
import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory


# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

file = None

with st.sidebar:
    api_key = st.text_input("1. OpenAI API keys를 입력해주세요.")

    if api_key:
        file = st.file_uploader(
            "2. 학습시킬 파일을 업로드 하세요. (.txt .pdf .docx)",
            type=["pdf", "txt", "docx"],
        )


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    openai_api_key=api_key,
)

memory = ConversationBufferMemory(llm=llm, return_messages=True)


@st.cache_resource(show_spinner="파일 임베딩 중...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"

    # 폴더 경로만 추출 (파일명 제외)
    folder_path = os.path.dirname(file_path)

    # 폴더가 존재하는지 확인하고, 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logging.info(f"folder_path: {folder_path}")
    logging.info(f"file_path: {file_path}")

    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context.
            If you don't know the answer just say you don't know. 
            DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")
st.markdown(
    """
이 챗봇은 파일의 내용을 분석해서 대답해주는 AI 챗봇입니다.

사용 방법은 다음과 같습니다.
1. SideBar에 OpenAI API Keys를 입력하세요.
2. 질문하고 싶은 내용의 파일을 업로드 하세요.
3. 무엇이든 물어보세요!

---
"""
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


if file:
    retriever = embed_file(file)
    send_message("챗봇 준비완료! 무엇이든 물어보세요!", "ai", save=False)
    paint_history()
    message = st.chat_input("업로드 된 파일에 대해 궁금한게 무엇인가요?")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": load_memory,
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
