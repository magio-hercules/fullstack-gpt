import json
import logging
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


###########
# 0. Init #
###########
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

llm = None
docs = None
topic = None
type = "FunctionCalling"
difficulty = None
choice = None


############
# 1. Class #
############
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        if text:
            text = text.replace("```", "").replace("json", "")
            return json.loads(text)
        else:
            return ''

output_parser = JsonOutputParser()


#############
# 2. Define #
#############
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

function = {
    "name": "create_quiz",
    "description": "다음과 같은 조건으로 퀴즈 문제를 구성해줘.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            },
        },
        "required": ["questions"],
    },
}

if type == "FunctionCalling":
    llm = ChatOpenAI(
        temperature=0.1,
        # model="gpt-3.5-turbo-1106",
        # model="gpt-4o-mini-2024-07-18",
        model="gpt-4o-mini-2024-07-18",
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ]
    )
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

# FunctionCalling
fc_prompt = PromptTemplate.from_template(
            """
            너는 퀴즈 문제 출제자로서 주어진 내용을 가지고 그 내용을 잘 알고 있는지 테스트를 할 수 있는 문제를 만들거야.
            주어진 제시어와 난이도를 이용해 아래의 조건에 맞게 제시어와 관련된 10개의 퀴즈 문제를 만들어 줘.
            # 조건
            1. 모든 문제는 중복되지 않는다.
            2. 각 질문은 4개의 답변이 있는 형식으로 구성된다.
            3. 난이도는 '쉬움'과 '어려움'으로 구성된다.
            4. 모든 질문과 답변은 한국어로 번역해서 표시한다.
            
            #내용
            difficulty: {difficulty}
            Context: {context}
            """
        )

# PromptTemplate 
pt_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Each question not duplicated.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
    ---
    difficulty: {difficulty}
    Context: {context}
""",
        )
    ]
)

# pt_chain = {"context": format_docs} | pt_prompt | llm
pt_chain = {"context": format_docs, "difficulty": RunnablePassthrough()} | pt_prompt | llm
# pt_chain = {"context": RunnablePassthrough(), "difficulty": RunnablePassthrough()} | pt_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_path = f"./.cache/quiz_files/{file.name}"

    # 폴더 경로만 추출 (파일명 제외)
    folder_path = os.path.dirname(file_path)
    
    # 폴더가 존재하는지 확인하고, 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="퀴즈 생성중...")
def run_quiz_chain(_docs, type, difficulty, topic):
    if type == "FunctionCalling":
        chain = fc_prompt | llm
        return chain.invoke({"context": format_docs(_docs), "difficulty": difficulty})
    elif type == "PromptTemplate":
        chain = {"context": pt_chain} | formatting_chain | output_parser
        return chain.invoke(_docs)
        # chain = pt_chain | formatting_chain | output_parser
        # return chain.invoke({"context": _docs, "difficulty": difficulty})


@st.cache_data(show_spinner="Wikipedia 검색중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


#############
# 3. Layout #
#############
st.title("QuizGPT")

with st.sidebar:
    api_key = st.text_input(
        'Step1. OpenAI API keys를 입력해주세요.',
        type='password',
    )
    if api_key:
        type = st.selectbox(
            "Step2. QuizGPT 타입을 선택해주세요.",
            (
                "FunctionCalling",
                # "PromptTemplate", 
            ),
            index=0,
        )
        choice = st.selectbox(
            "Step3. 퀴즈 생성 방식을 선택해주세요.",
            (
                "Wikipedia Article",
                "File",
            ),
            index=0,
        )
        difficulty = st.selectbox(
            "Step4. 퀴즈의 난이도를 선택해주세요.",
            (
                "쉬움",
                "어려움",
            ),
            index=0,
        )
        if choice == "File":
            file = st.file_uploader(
                "Step5. 학습시킬 파일을 업로드 하세요. (.txt .pdf .docx)",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Wikipedia 검색어")
            if topic:
                docs = wiki_search(topic)
            

############
# 4. Logic #
############
if not docs:
    st.markdown(
        """
        QuizGPT는 문서나 위키피디아의 정보를 이용해서 간단히 퀴즈를 만들수 있어요.
        
        본인의 상식은 어느 정도인지 확인해보세요!
    """
    )
else:
    response = None
    if type == "FunctionCalling":
        response = run_quiz_chain(docs, type, difficulty, topic if topic else file.name)
        if response.additional_kwargs:
            response = response.additional_kwargs["function_call"]["arguments"]
        response = json.loads(response)
    elif type == "PromptTemplate":
        response = run_quiz_chain(docs, type, difficulty, topic if topic else file.name)

    with st.form("questions_form"):
        if response:
            correct_count = 0;
            for i, question in enumerate(response["questions"], start=1):
                value = st.radio(
                    f'{i}번 문제. {question["question"]}',
                    [answer["answer"] for answer in question["answers"]],
                    key=f'q{i}',
                    index=None,
                )

                if {"answer": value, "correct": True} in question["answers"]:
                    st.success(f"{i}번 정답 🤩")
                    correct_count += 1
                elif value is not None:
                    st.error(f"{i}번 오답 😱")
            
            if correct_count == len(response["questions"]):
                st.info("🎉축하합니다🎊 모든 문제를 맞추셨어요!!🥳")
                st.balloons()
            
            button = st.form_submit_button()