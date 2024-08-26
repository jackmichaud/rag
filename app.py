from rag import stream_rag_pipeline, stream_rag_with_routing
from file_management import update_vectorstore_collection, delete_file
from streamlit_components import chatbot, file_uploader, file_explorer

# Runnable streamlit app
# To run this app, run `streamlit run app/app.py` in the terminal

chatbot(stream_rag_with_routing)

file_uploader(update_vectorstore_collection)

file_explorer(delete_file)
