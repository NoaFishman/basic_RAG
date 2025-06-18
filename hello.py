from sentence_transformers import SentenceTransformer 
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_ollama import OllamaLLM 
from langchain.chains import RetrievalQA 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader  
from langchain_community.document_loaders import PyPDFLoader

# 1. Load and split documents 
loader = PyPDFLoader("docs/Markets_CPS Protection Platforms.pdf") 
docs = loader.load() 
splitter = RecursiveCharacterTextSplitter(     
    chunk_size=500,     
    chunk_overlap=100,     
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] 
) 
split_docs = splitter.split_documents(docs)  

# 2. Create embeddings 
embedding_model = HuggingFaceEmbeddings(     
    model_name="all-MiniLM-L6-v2",     
    model_kwargs={"device": "cpu"} 
) 
db = FAISS.from_documents(split_docs, embedding_model)  

# 3. Setup LLM 
llm = OllamaLLM(model="mistral")  

# 4. Setup RetrievalQA chain 
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())  

# 5. Ask a question 
result = qa.invoke("What is the main topic of this document?") 
print(result)  
