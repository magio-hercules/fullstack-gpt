import os
from typing import Dict, List
from uuid import UUID
import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from urllib.parse import urlparse

st.set_page_config(
    page_title="Cloudflare's SiteGPT",
    page_icon="🖥️",
)

with st.sidebar:
    api_key = st.text_input(
        "Step1. OpenAI API keys를 입력해주세요.",
        type="password",
    )


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # save_message(self.message, "ai")
        pass

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        # self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    openai_api_key=api_key,
)

memory = ConversationBufferMemory(llm=llm, return_messages=True)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    # history = inputs["history"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    },
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        # "history": history,
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    # history = inputs["history"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            # "history": history,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


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


def load_memory(_):
    return memory.load_memory_variables({})["history"]


# site 이름 뽑아내기
def extract_site_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    dir_path = f"./.cache/embeddings_sites/{extract_site_name(url)}"
    # st.write(f'load_website: {extract_site_name(url)}')

    # 폴더가 존재하는지 확인하고, 없으면 생성
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cache_dir = LocalFileStore(dir_path)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    filters = (
        [
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ]
        if site_type == "Cloudflare"
        else [
            r"^(.*\/ui\/).*",
        ]
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=filters,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)

    # case1. cache 사용X
    # vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # return vector_store.as_retriever()
    # case2. cache 사용O
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


with st.sidebar:
    site_type = st.radio(label="Step2. WebSite 선택", options=["Cloudflare", "Flutter"])
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    url = st.text_input(
        "Step2-1. Site URL (고정)",
        placeholder="https://example.com",
        value=(
            "https://developers.cloudflare.com/sitemap-0.xml"
            if site_type == "Cloudflare"
            else "https://docs.flutter.dev/sitemap.xml"
        ),
        disabled=True,
    )

st.markdown(
    f"""
    # SiteGPT - {site_type} 
    ### {site_type if site_type == 'Cloudflare' else 'Flutter UI'}에 관련된 모든 질문을 해보세요.
    ---
    """
)

if not api_key:
    st.error("Step1. OpenAI API Key를 입력해주세요.")

if (not api_key) or (not url):
    st.session_state["messages"] = []
else:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)

        send_message("챗봇 준비완료! 무엇이든 물어보세요!", "ai", save=False)
        paint_history()

        message = st.chat_input("Ask a question to the website.")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                    "history": load_memory,
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            with st.chat_message("ai"):
                result = chain.invoke(message)
                result_message = result.content.replace("$", "\$")
                st.markdown(result_message)
                save_message(result_message, "ai")
