import os
# import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. Configuración 
load_dotenv()

# Compatibilidad local (.env) y Streamlit Cloud (secrets)
API_KEY = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', '')

MODEL_EMBEDDING = 'gemini-embedding-001'
CHROMA_DIR      = './chroma_db'
COLLECTION_NAME = 'nutricion_deportiva'

# 2. Rotación de modelos (free tier)
MODELS = [
    {'name': 'gemini-2.5-flash',      'rpd': 20},
    {'name': 'gemini-2.5-flash-lite', 'rpd': 1000},
]

# Contadores por sesión (se resetean al recargar la app)
if 'usage' not in st.session_state:
    st.session_state.usage = {m['name']: {'requests_today': 0} for m in MODELS}

def _modelo_disponible(model: dict) -> bool:
    return st.session_state.usage[model['name']]['requests_today'] < model['rpd']

def _registrar_uso(model_name: str):
    st.session_state.usage[model_name]['requests_today'] += 1

def _tiempo_reset() -> str:
    pacific      = ZoneInfo('America/Los_Angeles')
    now          = datetime.now(pacific)
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    segundos = int((next_midnight - now).total_seconds())
    return f'{segundos // 3600}h {(segundos % 3600) // 60}min'

# 3. System Prompt
SYSTEM_PROMPT = '''
Eres un asistente experto en nutrición deportiva y planificación del entrenamiento físico.
Tu base de conocimiento incluye información detallada sobre macronutrientes, periodización del
entrenamiento, suplementación deportiva, recuperación y estrategias de composición corporal.

## TU ROL
Actúas como un entrenador personal y nutricionista deportivo con formación científica. Respondes
preguntas sobre nutrición, entrenamiento, suplementación y recuperación de forma clara, precisa
y basada en evidencia.

## CÓMO RESPONDER
1. Basa SIEMPRE tus respuestas en el CONTEXTO RECUPERADO que se te proporciona.
2. Incluye datos concretos cuando estén disponibles (dosis, rangos, porcentajes, tiempos).
3. Estructura tus respuestas con claridad: usa listas o párrafos según convenga.
4. Si el contexto no contiene información suficiente, dilo explícitamente.
   No inventes datos ni hagas suposiciones sin base.
5. Usa el HISTORIAL DE CONVERSACIÓN para mantener coherencia entre turnos.
6. Adapta el nivel técnico al usuario.

## LIMITACIONES
- No eres médico. Para condiciones médicas específicas recomienda consultar un profesional.
- No prescribes medicamentos ni tratas enfermedades.
- Si preguntan sobre algo fuera de tu dominio, indícalo amablemente.

## TONO
Cercano, motivador y profesional. Usa un lenguaje accesible pero técnicamente riguroso.
'''

# 4. Estado del Agente
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context:  str
    question: str

# 5. Inicialización (cacheada para no recargar en cada rerun)
@st.cache_resource
def init_rag_system():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODEL_EMBEDDING,
        google_api_key=API_KEY
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 4}
    )
    return retriever

# 6. Grafo LangGraph
nl='\n'
def retrieve_node(state: AgentState) -> dict:
    retriever = init_rag_system()
    docs = retriever.invoke(state['question'])
    context_parts = [
        f'[{d.metadata.get('source', '?')} | pág.{d.metadata.get('page', '?')}]{nl}{d.page_content}'
        for d in docs
    ]
    return {'context': '\n\n---\n\n'.join(context_parts)}

def generate_node(state: AgentState) -> dict:
    rag_prompt = (
        f'CONTEXTO RECUPERADO DE LA BASE DE CONOCIMIENTO:\n'
        f'─────────────────────────────────────────\n'
        f'{state['context']}\n'
        f'─────────────────────────────────────────\n\n'
        f'PREGUNTA DEL USUARIO: {state['question']}\n\n'
        f'Responde basándote principalmente en el contexto proporcionado arriba.'
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state['messages'],
        HumanMessage(content=rag_prompt)
    ]

    # Rotación de modelos
    for model in MODELS:
        if not _modelo_disponible(model):
            continue

        llm = ChatGoogleGenerativeAI(
            model=model['name'],
            google_api_key=API_KEY,
            temperature=0.3,
            max_output_tokens=1024,
        )
        try:
            _registrar_uso(model['name'])
            response = llm.invoke(messages)
            return {
                'messages': [
                    HumanMessage(content=state['question']),
                    AIMessage(content=response.content)
                ]
            }
        except ChatGoogleGenerativeAIError as e:
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                # Marcar como agotado realmente y probar el siguiente
                st.session_state.usage[model['name']]['requests_today'] = model['rpd']
                continue
            raise

    # Todos los modelos agotados
    msg = (
        f'⏳ Has alcanzado el límite diario de consultas en todos los modelos disponibles.\n\n'
        f'El límite se restablece a medianoche (hora del Pacífico).\n'
        f'**Tiempo restante: {_tiempo_reset()}**\n\n'
        f'Inténtalo de nuevo más tarde.'
    )
    return {
        'messages': [
            HumanMessage(content=state['question']),
            AIMessage(content=msg)
        ]
    }

# 7. Construcción del grafo
@st.cache_resource
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()

# 8. UI Streamlit
st.set_page_config(
    page_title='Asistente Deportivo',
    page_icon='🏋️',
    layout='centered'
)

st.title('🏋️ Asistente Deportivo')
st.caption('Asistente experto en nutrición y entrenamiento deportivo · Powered by Gemini + RAG')
st.divider()

# Inicializar sistema RAG
if not API_KEY:
    st.error('⚠️ Configura GEMINI_API_KEY en los secrets de Streamlit o en el archivo .env')
    st.stop()

# Inicializar grafo
graph = build_graph()

# Estado de sesión
if 'conv_state' not in st.session_state:
    st.session_state.conv_state = {'messages': [], 'context': '', 'question': ''}
if 'display_messages' not in st.session_state:
    st.session_state.display_messages = []

# Historial de mensajes
for message in st.session_state.display_messages:
    with st.chat_message(message['role'], avatar='👤' if message['role']=='user' else '🤖'):
        st.markdown(message['content'])

# Input del usuario
if prompt := st.chat_input('Pregunta sobre nutrición o entrenamiento...'):
    # Mostrar mensaje del usuario
    with st.chat_message('user', avatar='👤'):
        st.markdown(prompt)
    st.session_state.display_messages.append({'role': 'user', 'content': prompt})

    # Generar respuesta
    with st.chat_message('assistant', avatar='🤖'):
        with st.spinner('Consultando la base de conocimiento...'):
            st.session_state.conv_state['question'] = prompt
            result = graph.invoke(st.session_state.conv_state)
            st.session_state.conv_state = result
            response = result['messages'][-1].content
        st.markdown(response)

    st.session_state.display_messages.append({'role': 'assistant', 'content': response})

# Sidebar con info
with st.sidebar:
    st.header('ℹ️ Acerca de NutriCoach')
    st.markdown('''
    **Base de conocimiento:**
    - 📄 Fundamentos de Nutrición Deportiva
    - 📄 Planificación del Entrenamiento
    - 📄 Recuperación y Suplementación

    **Stack tecnológico:**
    - 🤖 Google Gemini (LLM + Embeddings)
    - 🗄️ ChromaDB (Vector Store)
    - 🔗 LangGraph (Agente con memoria)
    ''')
    st.divider()

    # Métricas de uso
    st.subheader("📊 Uso de modelos (sesión)")
    for model in MODELS:
        used = st.session_state.usage[model["name"]]["requests_today"]
        rpd  = model["rpd"]
        st.progress(
            min(used / rpd, 1.0),
            text=f"{model['name']}: {used}/{rpd} req"
        )

    st.divider()
    turns = len(st.session_state.conv_state['messages']) // 2
    st.metric('Turnos de conversación', turns)

    if st.button('🔄 Nueva conversación'):
        st.session_state.conv_state = {'messages': [], 'context': '', 'question': ''}
        st.session_state.display_messages = []
        st.rerun()


# Guardar el archivo app.py
# with open('app.py', 'w', encoding='utf-8') as f:
#     f.write(streamlit_app_code)

print('✅ Archivo app.py generado correctamente')
print('\nPara ejecutar Streamlit localmente:')
print('  $ streamlit run app.py')
print('\nPara desplegar en Streamlit Cloud:')
print('  1. Sube el repositorio a GitHub')
print('  2. Ve a share.streamlit.io')
print('  3. Conecta el repo y añade GEMINI_API_KEY en Secrets')