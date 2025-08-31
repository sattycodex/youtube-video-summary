from langchain_core.prompts import ChatPromptTemplate
from utils.embedding_model import EuriEmbeddings
from utils.llm_model import EuriLLM
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel 

def store_embeddings(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(texts)
    embeddings = EuriEmbeddings()
    vector_store= FAISS.from_texts(chunks, embeddings)
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retrieved_docs=retriever.invoke("give me summary of this video")
    print("sample retrieved docs: ",retrieved_docs)
    return retriever


def load_youtube_transcript(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except Exception as e:
        print("No captions available for this video.")


def get_response(question):
    llm = EuriLLM()
    retriever=st.session_state.retriever
    prompt_template = """You are a helpful assistant that helps people find information.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer in a concise manner.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    def get_retreieved_docs(question):
        docs=retriever.invoke(question)
        context="\n".join([doc.page_content for doc in docs])
        return context
   

    conext_and_query = RunnableParallel({
        'context': RunnableLambda(get_retreieved_docs),
        'question': RunnablePassthrough()
    })

    chain = conext_and_query | prompt | llm | output_parser
    response=chain.invoke({'question': question})
    print("response is: ",response)
    return response




st.sidebar.title("youtube video Id")
video_id=st.sidebar.text_input("Enter youtube video id", key="video_id")

## load transcript and store embeddings and return retriever
isLoaded=False
if 'retriever' not in st.session_state:
    st.session_state.retriever=None

if st.sidebar.button("Process", key="process"):
    if video_id == "":
        st.write("Please enter a valid youtube video id")
    else:
        with st.spinner("Processing..."):
            transcript = load_youtube_transcript(video_id)
            if transcript:
                st.toast("Transcript loaded successfully",icon="✅")
                retriever=store_embeddings(transcript[0:1000])
                st.session_state.retriever=retriever
                isLoaded=True
            else:
                st.toast("No transcript available for this video",icon="⚠️")




st.title("Youtube Video Summary")
st.write("Enter youtube video id in the sidebar to get the summary of the video")		
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Your Questions...",disabled=(isLoaded==False)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = get_response(prompt)
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})	