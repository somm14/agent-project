import os
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

# 1. CONFIGURACIÓN
# ────────────────
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', '')

MODEL_EMBEDDING = 'gemini-embedding-001'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, 'chroma_db')

# 2. ROTACIÓN DE MODELOS (free tier)
# ──────────────────────────────────
MODELS = [
    {'name': 'gemini-2.5-flash',      'rpd': 20,    'label': 'gemini-2.5-flash',      'tier': 'primary'},
    {'name': 'gemini-2.5-flash-lite', 'rpd': 20,    'label': 'gemini-2.5-flash-lite', 'tier': 'fallback'},
]

if 'usage' not in st.session_state:
    st.session_state.usage = {m['name']: {'requests_today': 0} for m in MODELS}

if 'active_model' not in st.session_state:
    st.session_state.active_model = MODELS[0]['name']

if 'model_rotated' not in st.session_state:
    st.session_state.model_rotated = False

def _modelo_disponible(model: dict) -> bool:
    return st.session_state.usage[model['name']]['requests_today'] < model['rpd']


def _registrar_uso(model_name: str):
    st.session_state.usage[model_name]['requests_today'] += 1
    st.session_state.active_model = model_name

# Contador de tiempo de reseteo
def tiempo_reset() -> str:
    pacific      = ZoneInfo('America/Los_Angeles')
    now          = datetime.now(pacific)
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    segundos = int((next_midnight - now).total_seconds())
    return f'{segundos // 3600}h {(segundos % 3600) // 60}min'

# 3. System Prompt
# ────────────────
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

# 4. ESTADO DEL AGENTE
# ────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context:  str
    question: str

# 5. INICIALIZACIÓN (cacheada)
# ────────────────────────────
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

retriever = init_rag_system()

# 6. NODOS DEL GRAFO LANGGRAPH
# ─────────────────────────────
nl='\n'
def retrieve_node(state: AgentState) -> dict:
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

    previous_model = st.session_state.active_model

    # Rotación de modelos
    for model in MODELS:
        if not _modelo_disponible(model):
            continue
        llm = ChatGoogleGenerativeAI(
            model=model['name'],
            google_api_key=API_KEY,
            temperature=0.3,
        )
        try:
            _registrar_uso(model['name'])

            # Detectar rotación al modelo de fallback
            if model['name'] != previous_model and model['tier'] == 'fallback':
                st.session_state.model_rotated = True
            else:
                st.session_state.model_rotated = False

            response = llm.invoke(messages)
            return {
                'messages': [
                    HumanMessage(content=state['question']),
                    AIMessage(content=response.content),
                ],
                '_model_used': model['name'],
            }
        except ChatGoogleGenerativeAIError as e:
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                st.session_state.usage[model['name']]['requests_today'] = model['rpd']
                continue
            raise

    # Todos los modelos agotados
    msg = (
        f'⏳ Has alcanzado el límite diario de consultas en todos los modelos disponibles.\n\n'
        f'El límite se restablece a medianoche (hora del Pacífico).\n'
        f'**Tiempo restante: {tiempo_reset()}**\n\n'
        f'Inténtalo de nuevo más tarde.'
    )
    return {
        'messages': [
            HumanMessage(content=state['question']),
            AIMessage(content=msg)
        ]
    }

# 7. GRAFO
# ─────────
@st.cache_resource
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node('retrieve', retrieve_node)
    builder.add_node('generate', generate_node)
    builder.add_edge(START, 'retrieve')
    builder.add_edge('retrieve', 'generate')
    builder.add_edge('generate', END)
    return builder.compile()

# 8. UI-CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ───────────────────────────────────────
st.set_page_config(
    page_title='Asistente Deportivo',
    page_icon='🏋️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
<style>
    /* ── Stat cards ── */
    .stat-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: bold; color: #4285f4; }
    .stat-label  { color: #8b8fa8; font-size: 0.8rem; margin-top: 4px; }

    /* ── Model badge ── */
    .model-badge-primary {
        display: inline-flex; align-items: center; gap: 6px;
        background: #1a2a4a; border: 1px solid #2d5aa0;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.78rem; font-weight: 600; color: #4285f4;
    }
    .model-badge-fallback {
        display: inline-flex; align-items: center; gap: 6px;
        background: #2a1a00; border: 1px solid #a05a00;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.78rem; font-weight: 600; color: #f4a400;
    }

    /* ── Rotation banner ── */
    .rotation-banner {
        background: linear-gradient(90deg, #2a1a00, #3d2800);
        border: 1px solid #f4a400;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0 6px 0;
        color: #f4a400;
        font-size: 0.88rem;
    }

    /* ── System prompt box ── */
    .prompt-box {
        background: #1a1f35;
        border: 1px solid #3d4166;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .prompt-box span { color: #c8cadd; font-size: 0.88rem; }

    /* ── Separator ── */
    hr { border-color: #3d4166 !important; }
</style>
''', unsafe_allow_html=True)

# 9. SIDEBAR
# ────────────
with st.sidebar:
    st.title('🏋️ Asistente Deportivo')
    st.divider()

    # Modelo activo 
    st.subheader('🤖 Modelo activo')
    active = st.session_state.get('active_model', MODELS[0]['name'])
    is_fallback = active == MODELS[1]['name']

    if is_fallback:
        st.markdown(
            f'<div class='model-badge-fallback'>⚡ {active} &nbsp;·&nbsp; fallback</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class='model-badge-primary'>✦ {active} &nbsp;·&nbsp; primario</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Info base de conocimiento
    st.subheader('📚 Base de conocimiento')
    st.markdown('''
- 📄 Fundamentos de Nutrición Deportiva
- 📄 Planificación del Entrenamiento
- 📄 Recuperación y Suplementación
- 📄 Psicología del Rendimiento
- 📄 Cronobiología y Dietas Especiales
- 📄 Entrenamiento Funcional y Movilidad
''')
    
    # Métricas de uso
    st.subheader('📊 Uso de modelos (sesión)')
    for model in MODELS:
        used = st.session_state.usage[model['name']]['requests_today']
        rpd  = model['rpd']
        pct  = min(used / rpd, 1.0)
        color = 'normal' if pct < 0.7 else ('off' if pct >= 1.0 else 'normal')
        st.progress(pct, text=f'{model['name']}: {used}/{rpd} req')

    st.divider()

    # Stats de conversación
    turns = len(st.session_state.get('conv_state', {}).get('messages', [])) // 2
    msgs  = len(st.session_state.get('display_messages', []))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class='stat-card'>'
            f'<div class='stat-number'>{turns}</div>'
            f'<div class='stat-label'>Turnos</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class='stat-card'>'
            f'<div class='stat-number'>{msgs}</div>'
            f'<div class='stat-label'>Mensajes</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button('🔄 Nueva conversación', use_container_width=True, type='secondary'):
        st.session_state.conv_state        = {'messages': [], 'context': '', 'question': ''}
        st.session_state.display_messages  = []
        st.session_state.model_rotated     = False
        st.rerun()

        #  Stack técnico 
    with st.expander('🛠️ Stack tecnológico'):
        st.markdown('''
- 🤖 Google Gemini (LLM + Embeddings)
- 🗄️ ChromaDB (Vector Store)
- 🔗 LangGraph (Agente con memoria)
- 🐍 LangChain · Streamlit
''')

    # Historial RAW
    with st.expander('🔍 Historial raw (JSON)'):
        st.json(st.session_state.get('display_messages', [])
                
# 10. CUERPO PRINCIPAL
# ──────────────────────
st.title('🏋️ NutriCoach IA')
st.caption(
    'Asistente experto en nutrición y entrenamiento deportivo · '
    'Powered by Google Gemini + RAG (ChromaDB) + LangGraph'
)

# Inicializar RAG (y verificar que ChromaDB devuelve resultados)
try:
    retriever = init_rag_system()
    test = retriever.invoke('proteína deportista fuerza')
    if not test:
        st.error(
            '⚠️ ChromaDB no devuelve resultados. '
            'Verifica que `chroma_db/` está en el mismo directorio que `app.py`.'
        )
        st.stop()
except Exception as e:
    st.error(f'⚠️ Error al inicializar el sistema RAG: {e}')
    st.stop()

# Inicializar grafo
graph = build_graph()

# Estado de sesión
if 'conv_state' not in st.session_state:
    st.session_state.conv_state = {'messages': [], 'context': '', 'question': ''}
if 'display_messages' not in st.session_state:
    st.session_state.display_messages = []

# Banner de rotación de modelo
if st.session_state.get('model_rotated'):
    st.markdown(
        '<div class='rotation-banner'>'
        '⚠️ <strong>Modelo cambiado automáticamente</strong> · '
        f'<code>gemini-2.5-flash</code> ha alcanzado su límite diario. '
        f'Usando <code>gemini-2.5-flash-lite</code> como fallback. '
        f'El límite se restablece en <strong>{_tiempo_reset()}</strong> (hora del Pacífico).'
        '</div>',
        unsafe_allow_html=True,
    )
# System prompt box
with st.expander('🎭 Ver System Prompt activo', expanded=False):
    st.code(SYSTEM_PROMPT.strip(), language='markdown')

# Mensaje de bienvenida 
if not st.session_state.display_messages:
    st.info(
        '👋 **¡Bienvenido a tu Asistente deportivo!**  \n'
        'Pregúntame sobre nutrición deportiva, planificación del entrenamiento, '
        'suplementación, recuperación, psicología del rendimiento o cualquier tema '
        'relacionado con el rendimiento físico.'
    )

# Historial de mensajes
for message in st.session_state.display_messages:
    avatar = '👤' if message['role'] == 'user' else '🤖'
    with st.chat_message(message['role'], avatar=avatar):
        st.markdown(message['content'])
        if 'model' in message:
            is_fb = message['model'] == MODELS[1]['name']
            badge_cls = 'model-badge-fallback' if is_fb else 'model-badge-primary'
            icon      = '⚡' if is_fb else '✦'
            st.markdown(
                f'<div style="margin-top:6px">'
                f'<span class="{badge_cls}">{icon} {message["model"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# Input del usuario
if prompt := st.chat_input('Pregunta sobre nutrición, entrenamiento, suplementación...'):

    # Mostrar mensaje del usuario
    with st.chat_message('user', avatar='👤'):
        st.markdown(prompt)
    st.session_state.display_messages.append({'role': 'user', 'content': prompt})

    # Generar respuesta
    with st.chat_message('assistant', avatar='🤖'):
        with st.spinner('Consultando la base de conocimiento...'):
            st.session_state.conv_state['question'] = prompt
            result   = graph.invoke(st.session_state.conv_state)
            st.session_state.conv_state = result
            response = result['messages'][-1].content
            model_used = st.session_state.active_model

        st.markdown(response)

        is_fb     = model_used == MODELS[1]['name']
        badge_cls = 'model-badge-fallback' if is_fb else 'model-badge-primary'
        icon      = '⚡' if is_fb else '✦'
        st.markdown(
            f'<div style="margin-top:6px">'
            f'<span class="{badge_cls}">{icon} {model_used}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.session_state.display_messages.append(
        {'role': 'assistant', 'content': response, 'model': model_used}
    )

    # Si hubo rotación, forzar rerun para mostrar el banner
    if st.session_state.get('model_rotated'):
        st.rerun()


# 11. SECCIÓN EDUCATIVA
# ──────────────────────

with st.expander('📖 ¿Cómo funciona el sistema RAG? — Concepto clave'):
    st.markdown('''
    ### Arquitectura del Asistente

    Este asistente combina **RAG (Retrieval-Augmented Generation)** con un agente **LangGraph**:

    ```
    Usuario
      │
      ▼
    [Chat Input]
      │
      ▼
    LangGraph Agent (StateGraph)
      │
      ├── Nodo: retrieve
      │     └── ChromaDB similarity search (top-4 chunks)
      │
      └── Nodo: generate
            └── Rotación de modelos Gemini
                  ├── System Prompt (agente experto)
                  ├── Contexto RAG (chunks recuperados)
                  └── Historial de conversación (memoria)
    ```

    **¿Por qué RAG?** El LLM tiene conocimiento general pero puede alucinar datos específicos
    (dosis de suplementos, rangos de RM, protocolos concretos). Con RAG, las respuestas se
    anclan en los documentos de la base de conocimiento, reduciendo las alucinaciones.

    **Rotación de modelos:** Cuando `gemini-2.5-flash` (20 RPD en free tier) agota su cuota
    diaria, el agente rota automáticamente a `gemini-2.5-flash-lite` (1.000 RPD) sin
    interrumpir la conversación. Un banner en la UI notifica el cambio.
    ''')