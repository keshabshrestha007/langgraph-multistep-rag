from typing import TypedDict,List,Literal
import sqlite3

from langchain_core.messages import SystemMessage,AIMessage,HumanMessage,BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.sqlite import SqliteSaver

from schema.validator import GradeDocument
from models.llms import llm
from tools import retriever_tool


#make sqlite connection
sqlite_conn = sqlite3.connect("1_multistep_rag.sqlite",check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)

class AgentState(TypedDict):
    """Defines the state structure for the multi-step RAG agent."""
    messages: List[BaseMessage]
    documents:List[Document]
    rephrased_question :str
    proceed_to_generate: bool
    question: HumanMessage
    uploaded_file: str
    uploaded_file_name: str


def question_rewriter(state:AgentState) -> AgentState:
    """Rewrites the input question into a single standalone question.
    Args:
        state (AgentState): The current state .
    Returns:
        state (AgentState): Updated state with rephrased question.
    """

    state["documents"] = []
    state["proceed_to_generate"] = False
    state["rephrased_question"] = ""
    state["question"] = state.get("question", HumanMessage(content=""))
    state["uploaded_file"] = state.get("uploaded_file", "")
    state["uploaded_file_name"] = state.get("uploaded_file_name", "")
    

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])
    
    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        question = state["question"].content
        messages = [
            SystemMessage(content="You are a helpful assisatant that rephrases the user's question into a standalone question optimized for retrieval.")

        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=question))
        
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        prompt = rephrase_prompt.format()

      
        response = llm.invoke(prompt)
       
        rephrased_question = response.content.strip()
        state["rephrased_question"] = rephrased_question
    
    else:
        state["rephrased_question"] = state["question"].content

    return state
    

def retrieve(state:AgentState):
    """Retrieve relevant documents using the uploaded file or default index.
    Args:
        state (AgentState): The current state.
    Returns:
        state (AgentState): Updated state with retrieved documents.
    """
    # Get file info from state
    uploaded_file = state.get("uploaded_file")
    uploaded_file_name = state.get("uploaded_file_name")
    question = state.get("rephrased_question", "")
    
    print(f"\nRetrieval step:")
    print(f"- File: {uploaded_file}")
    print(f"- Index name: {uploaded_file_name}")
    print(f"- Question: {question}")

    # Always assign a retriever (either derived from the file or default)
    try:
        if uploaded_file:
            print(f"Creating retriever for existing file: {uploaded_file}")
            retriever = retriever_tool(file=uploaded_file, file_name=uploaded_file_name.lower())
        else:
            print("Using default retriever (no file or file not found)")
            retriever = retriever_tool()
    except Exception as e:
        print(f"Failed to create retriever: {e}")
        state["documents"] = []
        state["proceed_to_generate"] = False
        return state

    # Use LangChain retriever API when available; otherwise try invoke
    documents = []
    try:
        if hasattr(retriever, 'invoke'):
            documents = retriever.invoke(state["rephrased_question"])
        
        elif hasattr(retriever, 'get_documents'):
            documents = retriever.get_documents(state["rephrased_question"])
        else:
            print("Retriever object has no supported fetch method; returning empty documents")
            documents = []
    except Exception as e:
        print(f"Error during retrieval: {e}")
        documents = []

    state["documents"] = documents
    state["proceed_to_generate"] = len(documents) > 0

    return state



def retrival_grader(state:AgentState):
    """Grades retrieved documents for relevance to the question.
    Args:
        state (AgentState): The current state.
    Returns:
        state (AgentState): Updated state with relevant documents filtered.
    """
    system_message = SystemMessage(content="""
                    You are a helpful grader accessing the relevance of a retrived document to user's question.
                    only answer with 'Yes' or 'No'.
                    If the document contains the information relevant to the user's question,respond with 'Yes',
                    otherwise respond with 'No'.""")
    
   
    
    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(content=
                                     f"User question:{state['rephrased_question']}\n\n Retrieved document:{doc.page_content}")
        
        grade_prompt = ChatPromptTemplate.from_messages([system_message,human_message])
        grader_chain = grade_prompt | structured_llm
        response = grader_chain.invoke({})

        if response.score.strip().lower()=="yes":
            relevant_docs.append(doc)
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) >0

    return state

def proceed_router(state:AgentState)->Literal["generate_answer","off-topic-response"]:
    """Route to generate_answer if we have relevant documents, otherwise off-topic.
    Args:
        state (AgentState): The current state.
    Returns:
        str: The next node to route to.
    """
    have_docs = len(state.get("documents", [])) > 0
    should_proceed = state.get("proceed_to_generate", False)
    
    print(f"Routing decision: docs={have_docs}, proceed={should_proceed}")
    
    if have_docs and should_proceed:
        return "generate_answer"
    else:
        return "off-topic-response"


def generate_answer(state:AgentState):
    """Generates an answer based on the retrieved documents and chat history.
    Args:
        state (AgentState): The current state.
    Returns:
        state (AgentState): Updated state with the generated answer.
    """
    if "messages" not in state or state["messages"] is None:
        raise ValueError("state must include 'messages'")    

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the chat history and context. Take the latest question into consideration.

        Chat history: {chat_history}
        Context: {context}
        Question: {question}

        Answer wisely and accurately.
        """
    )

    formatted_prompt = prompt.format_messages(
        chat_history=str(state["messages"]),
        context="\n\n".join([doc.page_content for doc in state["documents"]]),
        question=state["rephrased_question"]
    )

    response = llm.invoke(formatted_prompt)

    state["messages"].append(AIMessage(content=response.content.strip()))

    return state


def off_topic_response(state:AgentState):
    """Generates a response indicating the question is off-topic.
    Args:
        state (AgentState): The current state.
    Returns:
        state (AgentState): Updated state with the off-topic response."""
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry.I can't answer the question."))
    return state


workflow = StateGraph(AgentState)

# creating nodes

workflow.add_node("question_rewritter",question_rewriter)
workflow.add_node("off-topic-response", off_topic_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrival_grader)
workflow.add_node("generate_answer", generate_answer)


# connecting nodes through edges
workflow.add_edge("question_rewritter","retrieve")
workflow.add_edge("retrieve","retrieval_grader")
workflow.add_conditional_edges("retrieval_grader",proceed_router)
workflow.add_edge("generate_answer",END)
workflow.add_edge("off-topic-response",END)
workflow.set_entry_point("question_rewritter")

graph = workflow.compile(checkpointer=checkpointer)





