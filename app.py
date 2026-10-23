import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


@st.cache_resource
def create_graph():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=st.secrets["GROQ_API_KEY"],
    )

    def chatbot(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile()


graph = create_graph()

st.title("🤖 LangGraph + Groq Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = graph.invoke(
            {"messages": st.session_state.messages}
        )

        response = result["messages"][-1].content
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )