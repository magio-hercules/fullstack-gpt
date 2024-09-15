import yfinance
import json
import time
import streamlit as st
import openai as client
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from urllib.parse import urlparse
from typing import Any, Type
from langchain.chat_models import ChatOpenAI
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


st.set_page_config(
    page_title="InvestorGPT with OpenAI Assistants",
    page_icon="🖥️",
)

with st.sidebar:
    api_key = st.text_input(
        'Step1. OpenAI API keys를 입력해주세요.',
        type='password',
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
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    openai_api_key=api_key
)

memory = ConversationBufferMemory(
    llm=llm,
    return_messages=True
)

### assistant - start

def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

def create_run(assistant_id, thread_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    # st.write(f'create_run run_id: {run.id}')
    return run

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def create_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    # st.write(f'get_tool_outputs run: {run}')
    outputs = []

    if (run.required_action == None) or (run.required_action.submit_tool_outputs == None):
        # st.write(f'get_tool_outputs return []')
        return outputs
    
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        # print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    # st.write(f'get_tool_outputs return {outputs}')
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outpus,
    )

def get_assistant_messages(thread_id, question):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    results = []
    result_index = 0
    for index, message in enumerate(messages):
        if message.role == 'assistant':
            result_index = index
        result = message.content[0].text.value
        results.append(result)
    return results[result_index]

### assistant - end


### 채팅 - start


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


### 채팅 - end


st.markdown(
    f"""
    # InvestorGPT with OpenAI Assistants
    ### 투자와 관련된 모든 질문을 해보세요.
    ---
    """
)


if not api_key:
    st.error('Step1. OpenAI API Key를 입력해주세요.')
    st.session_state["messages"] = []
    st.session_state["assistant_id"] = []
    st.session_state["thread_id"] = []
else:
    if st.session_state["messages"] == []:
        assistant = client.beta.assistants.create(
            name="Investor Assistant",
            instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
            model="gpt-4-1106-preview",
            tools=functions,
        )
        assistant_id = assistant.id
        st.session_state["assistant_id"] = assistant_id
        # st.write(f'step1. create assistant_id: {assistant_id}')
    else:
        if st.session_state["assistant_id"] != []:
            assistant_id = st.session_state["assistant_id"]
            # st.write(f'step1. create assistant_id: {assistant_id}')


        
    send_message("챗봇 준비완료! 무엇이든 물어보세요!", "ai", save=False)
    paint_history()

    message = st.chat_input("투자를 검토하고 있는 회사에 대해 알려주세요.")
    if message:
        if st.session_state["thread_id"] != []:
            thread_id = st.session_state["thread_id"]
            # st.write(f'step2. thread_id: {thread_id}')
        else:
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            )
            thread_id = thread.id
            st.session_state["thread_id"] = thread_id
            # st.write(f'step2. create thread, thread_id: {thread_id}, thread.id: {thread.id}')
            # st.write(f'step2. create thread, thread: ({thread})')

        send_message(message, "human")
        
        # st.write(f'step !!!. check, assistant_id: {assistant_id}, thread_id: {thread_id}')

        create_message(thread_id, message)        
        run = create_run(assistant_id=assistant_id, thread_id=thread_id)
        st.write(f'step !!!. check, run_id: {run.id}')
        
        run_status = get_run(run.id, thread_id).status
        while run_status != 'completed' and run_status != 'failed':
            with st.spinner('InvestorGPT가 생각중이에요...'):
                run_status = get_run(run.id, thread_id).status
                st.write(f'check, run_status: {run_status}')
                time.sleep(1)
                if run_status == 'requires_action':
                    while run_status == 'requires_action':
                        submit_tool_outputs(run.id, thread_id)
                        time.sleep(1)
                        run_status = get_run(run.id, thread_id).status

        if run_status == 'completed':
            result = get_assistant_messages(thread_id, message)
            send_message(result, "ai")

        if run_status == 'failed':
            send_message('토큰이 부족한가??! 😇', "ai")
            