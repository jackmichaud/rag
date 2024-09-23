import os
import time

from langchain_core.pydantic_v1 import BaseModel, Field, conlist, ConstrainedList
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from file_management import get_embedding_function, list_uploaded_files
import json
from langchain.load import dumps
from langchain_groq import ChatGroq
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma

def stream_rag_pipeline(question: str, collection_name: str):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant whose goal is to answer a user's question given the context from the following document collection: {expertise}"),
        ("system", "Use only the context provided to develop you answer. If the context does not answer the question, say so. Do not overexplain. If you quote something from this context, copy it exactly without changing the words, and cite where you got the information from."),
        ("system", "Context: \n\n {context} \n\n The answer to your question -- {question} -- is: "),
    ])
    
    # Retrieve documents with similar embedding
    retriever = Chroma(
        persist_directory="./chroma", 
        embedding_function=get_embedding_function()
    )
    if(collection_name == "All Collections"):
        filter = None
    else:
        filter = dict(filter = "data/" + collection_name)
    similar = retriever.similarity_search(question, k=6 , filter=filter)

    # Format chunks
    delimiter = "\n\n---\n\n"
    context = delimiter.join([dumps(doc.page_content) + "\nSource: " + 
                            doc.metadata["source"] + ", Page Number: " + 
                            str(doc.metadata["page"]) + ", Chunk ID: " + 
                            doc.metadata["id"] for doc in similar])

    parser = StrOutputParser()
    model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    if(len(similar) == 0):
        return {"response": stream_data("I don't know. No context was given."), "sources": []}

    chain = prompt | model | parser

    sources = [os.path.basename(doc.metadata.get("id", None)) for doc in similar]

    return {"response": chain.stream({"question": question, "context": context, "expertise": collection_name}), "sources": sources}

def stream_rag_with_routing(question: str, collection_name: str):
    # TODO Add routing
    
    documents_list_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant whose goal is to address a prompt by listing documents/resources that will help answer their question."),
        ("ai", "Which documents can I choose from?"),
        ("system", "Here are a list of all the documents you can choose from. If no documents are relevant to the question, just say \"I don't know\". Document list: \n{documents_list}"),
        ("ai", "I'm ready to help. What is the user's question?"),
        ("human", "Hello! This is my question: {question}"),
        ("ai", "I will tell you the indexes of the relevant documents. They are: "),
    ])

    print("Your question is: ", question)
    print("Your collection is: ", collection_name)

    if(collection_name == "All Collections"):
        documents_list = json.dumps(list_uploaded_files(), indent = 4) # needs work to enumerate correctly
        print(documents_list)
    else:
        documents_list = list_uploaded_files(collection_name)
        stringified_documents_list = "\n".join([f"{i}. {doc}" for i, doc in enumerate(documents_list)]) + "\n"

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    document_list_structured_llm = llm.with_structured_output(IndexesOfDocuments)
    parser = StrOutputParser()

    document_list_chain = documents_list_prompt | document_list_structured_llm 

    # Retrieve documents with similar embedding
    retriever = Chroma(
        persist_directory="./chroma", 
        embedding_function=get_embedding_function()
    )

    prompt = ChatPromptTemplate.from_template("""Answer the following question using only the context below \n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:""")

    answers_question_prompt = ChatPromptTemplate.from_template("""Then respond with a boolean value (answers_question) if the context answers the user's quesiton:\n\nQuestion: {question}\n\nContext: {context}""")

    structured_llm = llm.with_structured_output(Response)

    chain = prompt | llm | parser

    answers_question_chain = answers_question_prompt | structured_llm

    retry_count = 0
    names_of_relevant_documents = [""]
    searched_documents = []
    response = {"response": "", "answers_question": False}
    while response["answers_question"] == False and documents_list != [] and retry_count < 3:
        print("Iteration #" + str(retry_count))
        print("Possible documents are: ", documents_list)
        stringified_documents_list = "\n".join([f"{i}. {doc}" for i, doc in enumerate(documents_list)]) + "\n"
        indexes_of_relevant_documents = document_list_chain.invoke({"question": question, "documents_list": stringified_documents_list})
        names_of_relevant_documents = [documents_list[index] for index in indexes_of_relevant_documents.indexes]
        print("Documents that may be relevant: " + str(names_of_relevant_documents))

        similar = []
        for name in names_of_relevant_documents:
            similar.extend(
                retriever.similarity_search(question, k=6, filter={"source": "data/" + collection_name + "/" + name}) 
            )
        
        # Format chunks
        delimiter = "\n\n---\n\n"
        context = delimiter.join([dumps(doc.page_content) + "\nSource: " + 
                                doc.metadata["source"] + ", Page Number: " + 
                                str(doc.metadata["page"]) + ", Chunk ID: " + 
                                doc.metadata["id"] for doc in similar])

        print("Checking context relevance")
        # Check context relevance with llm
        context_relevance = answers_question_chain.invoke({"question": question, "context": context})
        context_relevance = context_relevance.answers_question

        if(context_relevance == True):
            print("Generating response...")
            return {"response": chain.stream({"question": question, "context": context}), "sources": names_of_relevant_documents}

        retry_count += 1
        documents_list = [doc for doc in documents_list if doc not in names_of_relevant_documents]
        searched_documents = searched_documents + names_of_relevant_documents

    print("No relevant documents found")
    return {"response": stream_data("I don't know based on the context provided"), "sources": searched_documents}
        
def stream_data(data: str):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)

class IndexesOfDocuments(BaseModel):
    indexes: List[int]

class Response(BaseModel):
    answers_question: bool