# 1. CARGA DE DEPENDENCIAS
# -------------------------

# Configuración de API Key y variables del entorno
from dotenv import load_dotenv
import os

load_dotenv()

# Procesamiento de Documentos PDF (Chunking)
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Conocimiento vectorial y Embeddings
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# Costrucción del Agente con LangGraph
from typing import Annotated, List, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from IPython.display import Image, display, Markdown

# 2. CONFIGURACIÓN DE RUTAS Y DE CONSTANTES
# -----------------------------------------

# Configuración de rutas
PDF_DIR = './docs'

PDF_FILES = [
    'Doc1_Fundamentos_Nutricion_Deportiva.pdf',
    'Doc2_Planificacion_Entrenamiento.pdf',
    'Doc3_Recuperacion_Suplementacion.pdf'
]

# API Key
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_EMBEDDING = 'gemini-embedding-001'
MODEL_LLM = 'gemini-2.5-flash'

# Creación y carga de colección ChormaDB
CHROMA_DIR = './chroma_db'
COLLECTION_NAME = 'nutricion_deportiva'

# 3. CARGA DE PDFs
# ----------------

documentos = []

for pdf_file in PDF_FILES:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    
    if not os.path.exists(pdf_path):
        print(f'No encontrado: {pdf_path} - asegúrate de colocar los PDFs en "{PDF_DIR}"')
        continue
    
    carga = PyPDFLoader(pdf_path)
    docs = carga.load()
    
    # Añadir metadatos a cada página
    # for doc in docs:
    #     doc.metadata["source_file"] = pdf_file
    #     doc.metadata["domain"] = "nutricion_deportiva"
    
    documentos.extend(docs)

# 4. CHUNKING
# -----------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,           # 800 caracteres por chunk
    chunk_overlap=100,        # Solapamiento para preservar contexto
    length_function=len,
    separators=['\n\n', '\n', '. ', ' ', '']  # Separación semántica
)

chunks = text_splitter.split_documents(documentos)

# 5. CREACIÓN DE LA BASE DE CONOCIMIENTO VECOTRIAL (CHROMADB + GEMINI EMBEDDINGS)
# -------------------------------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model=MODEL_EMBEDDING,
    google_api_key=API_KEY
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# 6. DISEÑO DEL SYSTEM PROMPT
# ---------------------------
