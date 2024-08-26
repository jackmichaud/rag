import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


def get_embedding_function():
    embeddings = OllamaEmbeddings(
      model="nomic-embed-text"
    )
    return embeddings

def list_uploaded_files(collection_name = None):
    files = {}
    if collection_name is not None:
        files.update({collection_name: [f for f in os.listdir(f"app/data/{collection_name}") if f.endswith(".pdf")]})
        return files[collection_name]
    else:
        for collection in os.listdir("app/data"):
            files.update({collection: [f for f in os.listdir(f"app/data/{collection}") if f.endswith(".pdf")]})
        return files  
    
    
def delete_file(file_path):
    print(f"Deleting file: {file_path}")

    file_path = "app/" + file_path

    # Remove local file
    os.remove(file_path)

    # Remove file from vectorstore
    vectorstore = Chroma(
        persist_directory="./app/chroma", 
        embedding_function=get_embedding_function()
    )
    existing_items = vectorstore.get(where={"source": file_path})
    ids = list(existing_items.values())[0]
    vectorstore.delete(ids=ids)

    # Find the index of the last slash
    last_slash_index = file_path.rfind('/')
    # Slice the string up to the last slash
    directory = file_path[:last_slash_index]

    print("🗑️ Deleted " + str(len(existing_items)) + " document chunks from vectorstore")

    # Check if directory is empty
    if not os.listdir(directory) and len(directory) >= 9:
        os.rmdir(directory)
        print("🗑️ Deleted " + directory)



def update_vectorstore_collection(collection_name: str):
    # Load documents in a given colleciton
    document_loader = PyPDFDirectoryLoader(os.path.join("app", "data", collection_name))
    docs = document_loader.load()
    print("Loaded", len(docs), "documents")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        is_separator_regex = False
    )
    chunks = text_splitter.split_documents(docs)
    print("Documents split into " + str(len(chunks)) + " chunks")

    # Calculate chunk ids
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Find the index of the last slash
        last_slash_index = source.rfind('/')
        # Slice the string up to the last slash
        filter = source[:last_slash_index]

        # Create metadata that will be used for filtering during retrieval
        chunk.metadata["filter"] = filter

    vectorstore = Chroma(
        persist_directory="./app/chroma", 
        embedding_function=get_embedding_function()
    )

    # Add or Update the documents.
    existing_items = vectorstore.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    print("🏷️ Generating metadata")
    new_chunks = []
    summaries = ""
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            # Add metadata to the document
            chunk_summary = generate_metadata(chunk)
            summaries += "Chunk " + chunk.metadata["id"] + " summary: " + chunk_summary + "\n\n"
            chunk.metadata["summary"] = chunk_summary

            new_chunks.append(chunk)

    print("📝 Summarizing document")
    document_summary = generate_document_summary(summaries)
    for chunk in new_chunks:
        chunk.metadata["document_summary"] = document_summary
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vectorstore.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def generate_metadata(chunk):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following document chunk in one or two sentences: {context}")
    ])

    model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    parser = StrOutputParser()

    chain = prompt | model | parser

    summary = chain.invoke({"context": chunk.page_content})

    return summary

def generate_document_summary(summary):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following document in up to four sentences: {context}")
    ])

    model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    parser = StrOutputParser()

    chain = prompt | model | parser

    summary = chain.invoke({"context": summary})

    return summary