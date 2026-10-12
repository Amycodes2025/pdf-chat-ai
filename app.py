# 1.Import Libraries

import io
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai 
from google.genai import errors
from google.genai import types

# Settings
MAX_MB = 10             
MAX_PAGES = 100        
CHUNK_SIZE = 1000       
CHUNK_OVERLAP = 200     
TOP_K = 4               
EMBED_MODEL = "gemini-embedding-001"     
CHAT_MODEL = "gemini-flash-lite-latest"   

# 2.Page Setup
st.set_page_config(page_title = "PDF Chat", layout = "centered")

st.markdown("""
<style>

.block-container{padding-top: 2rem; max-width:800px;}

h1 {text-align: center;}
            
.subtitle{text-align: center; color:gray; margin-bottom: 2rem;}
            
[data-testid = "stSidebar"]{
    background-color: #000000;
}
[data-testid = "stSidebar"] * {
    color: #FFFFFF;
}

[data-testid = "stFileUploaderDropzone"]{
    background-color: #1A1A1A;
    border: 1px dashed #555555;
}

[data-testid = "stSidebar"] button {
    background-color: #222222;
    border: 1px solid #555555;
}

[data-testid = "stChatInput"],
[data-testid = "stChatInput"] div,
[data-testid = "stChatInput"] textarea {
    background-color: #F0F0F0 !important;
    color: #1F1F1F !important;
    border-radius: 5px;
}
               
[data-testid = "stFileUploaderFile"],
[data-testid = "stFileUploaderFile"] div {
    background-color: #00000;
    border-radius: 8px;
    border-color: black;
}
            
[data-testid="stChatMessage"] :not(pre) > code {
    background-color: transparent !important;
    color: inherit !important;
    font-weight: 600;
}

[data-testid="stChatMessage"] pre {
    background-color: #1E1E1E !important;   
    border-radius: 10px;                   
}
[data-testid="stChatMessage"] pre code {
    color: #F8F8F2 !important;              
}

</style>
""", unsafe_allow_html = True) 

# 3.Connect to Gemini
client = genai.Client(api_key = st.secrets["GEMINI_API_KEY"])

# 4. Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Read the PDF only once
@st.cache_data
def read_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []    
        for i, page in enumerate(reader.pages):  
            pages.append({
                "page": i + 1,                     
                "text": page.extract_text() or "",  
            })
        return pages
    except Exception:
        return None    

def split_into_chunks(pages):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE]     
            if piece.strip():                            
                chunks.append({"page": p["page"], "text" : piece})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

@st.cache_data
def embed_chunks(chunk_texts):
    vectors = []
    for i in range(0, len(chunk_texts), 100):
        batch = chunk_texts[i: i + 100]
        result = client.models.embed_content(
            model = EMBED_MODEL,
            contents = batch,
            config = types.EmbedContentConfig(task_type = "RETRIEVAL_DOCUMENT"),    
        )
        vectors += [e.values for e in result.embeddings]
    return np.array(vectors)       

def find_relevant_chunks(question, chunks, vectors):
    result = client.models.embed_content(
        model = EMBED_MODEL,
        contents = question,
        config = types.EmbedContentConfig(task_type = "RETRIEVAL_QUERY"),
    )
    q = np.array(result.embeddings[0].values)
    scores = vectors @ q / (np.linalg.norm(vectors, axis = 1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in best]


# 6. Sidebar
with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload a PDF", type = "pdf")

    st.caption("Don't upload private or sensitive files")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.divider()

    with st.expander("How to use"):
        st.write("1. Upload a PDF")
        st.write("2. Ask any question about it")
        st.write("3. Get answers from AI instantly")

    st.caption(f"Limits: {MAX_MB} MB . {MAX_PAGES} pages")
    st.caption("Built with Streamlit + Gemini")

# 7. Main Page Header 
st.title("Chat with your PDF")
st.markdown('<p class = "subtitle">Upload a PDF and ask anything about it </p>', unsafe_allow_html = True) 

if not uploaded_file:
    st.info("Upload a file from side bar to get started")
    st.stop()

# 8. Get PDF Text 
if uploaded_file.size > MAX_MB * 1024 * 1024:
    st.error(f"File too big. Max size is {MAX_MB} MB.")
    st.stop()

pages = read_pdf(uploaded_file.getvalue())

if pages is None:
    st.error("Couldn't read this PDF. It may be damaged or password-protected")
    st.stop()

if len(pages) > MAX_PAGES:
    st.error(f"Too many pages. Max is {MAX_PAGES} pages.")
    st.stop()

chunks = split_into_chunks(pages)

if not chunks:
    st.error("No text found. This may be a scanned PDF (images only)")
    st.stop()
try:
    with st.spinner("Preparing your PDF..."):
        vectors = embed_chunks([c["text"] for c in chunks])
except Exception:
    st.error("Couldn't prepare the PDF right now. Please try again in a minute")
    st.stop()
    
st.sidebar.success(f"{len(pages)} pages . {len(chunks)} chunks ready")


# 9. Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# 10. Chat input box
question = st.chat_input("Ask a question about your PDF...")

if question:
     st.session_state.messages.append({"role" : "user", "content" : question})
     with st.chat_message("user"):
         st.write(question)

     with st.chat_message("assistant"):
        try:
            relevant = find_relevant_chunks(question, chunks, vectors)

            context = "\n\n".join(f"[Page{c['page']}]\n{c['text']}" for c in relevant)

            source_pages = ", ".join(str(p) for p in sorted({c["page"] for c in relevant}))

            prompt = f"""Answer the question using only this document. 
            If the answer is not in the document, say "I couldn't find that in the PDF"

            Document:
            {context}

            Question: {question}"""

            stream = client.models.generate_content_stream(
                model = CHAT_MODEL,
                contents = prompt,
            )
            answer = st.write_stream(chunk.text or "" for chunk in stream)

            st.caption(f"Sources: pages {source_pages}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_pages
             })
        except errors.ClientError as e:
            if e.code == 429:
                st.error("Free limit reached. Please wait a minute and try again")
            else:
                st.error("Request problem. Check your API key")
        except errors.ServerError:
            st.error("Free limit reached. Please wait a minute and try again")
        except Exception as e:
            st.error(f"Something went wrong. Please try again {e}")


          
         