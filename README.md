# 🏋️ Asistente Experto en Nutrición y Entrenamiento Deportivo

> Proyecto Modular - IA Generativa | Máster en Data e IA  
> Stack: Google Gemini · ChromaDB · LangGraph · LangChain · Streamlit

---

## **Descripción del Dominio**

El agente está especializado en **nutrición deportiva y planificación del entrenamiento físico**, un dominio con alta demanda de información precisa, técnica y personalizada. Se ha elegido este tema porque:

- Combina datos numéricos concretos (dosis, rangos, porcentajes) que permiten evaluar la precisión del RAG
- Tiene preguntas con respuestas claramente recuperables de los documentos y preguntas abiertas que requieren síntesis
- Es un dominio con alta variabilidad de perfiles de usuario (principiantes, intermedios, élite) que permite demostrar la adaptabilidad del agente

### Base de Conocimiento (3 documentos, 29 páginas)

| Archivo | Páginas | Temas principales |
|---------|---------|-------------------|
| `Doc1_Fundamentos_Nutricion_Deportiva.pdf` | 10 | Balance energético, macronutrientes, micronutrientes, hidratación, nutrición perioperativa |
| `Doc2_Planificacion_Entrenamiento.pdf` | 9 | Principios del entrenamiento, periodización (lineal, DUP, bloques), fuerza, resistencia, composición corporal |
| `Doc3_Recuperacion_Suplementacion.pdf` | 10 | Recuperación, sueño, suplementos (creatina, cafeína, beta-alanina), planes dietéticos, casos prácticos, lesiones |

---

## **Arquitectura del Sistema**

```
Usuario
  │
  ▼
[Celda de chat / Streamlit UI]
  │
  ▼
LangGraph Agent (StateGraph)
  │
  ├── Nodo: retrieve
  │     └── ChromaDB similarity search (top-4 chunks)
  │           └── Gemini text-embedding-004
  │
  └── Nodo: generate
        └── Gemini 1.5 Flash
              ├── System Prompt (NutriCoach)
              ├── Contexto RAG (chunks recuperados)
              └── Historial de conversación (memoria)
```

**Flujo de datos:** `START → retrieve → generate → END`

---

## **Instalación y Ejecución**

### Requisitos previos

- Python 3.10+
- API Key de Google Gemini ([obtener aquí](https://aistudio.google.com/app/apikey))
- `uv` instalado ([instrucciones](https://docs.astral.sh/uv/getting-started/installation/)) - recomendado

### 1. Clonar el repositorio

```bash
git clone https://github.com/somm14/agent-project
cd nutricoach-rag
```

### 2. Instalar dependencias

El proyecto usa `uv` para la gestión del entorno. El archivo `pyproject.toml` incluido define todas las dependencias con versiones fijadas, garantizando un entorno reproducible en cualquier máquina.

**Con uv (recomendado):**
```bash
uv sync
```
Esto crea automáticamente un entorno virtual `.venv/` e instala todas las dependencias. No necesitas hacer nada más.

**Con pip (alternativa):**
```bash
pip install -r requirements.txt
```
El `requirements.txt` es compatible con cualquier entorno pip estándar.

### 3. Configurar la API Key

Crea un archivo `.env` en la raíz del proyecto:

```bash
# .env
GOOGLE_API_KEY=tu_clave_de_gemini_aqui
```

> ⚠️ **Nunca subas el archivo `.env` a GitHub.** Está incluido en `.gitignore`.

### 4. Preparar los documentos (MODIFICAR)

Coloca los 3 PDFs en la carpeta `docs/`:

```
agent-project/
├── docs/
│   ├── Doc1_Fundamentos_Nutricion_Deportiva.pdf
│   ├── Doc2_Planificacion_Entrenamiento.pdf
│   └── Doc3_Recuperacion_Suplementacion.pdf
├── asistente_deportivo_rag.ipynb
├── app.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 5. Ejecutar el notebook

**Con uv (lanza el notebook dentro del entorno automáticamente):**
```bash
uv run jupyter notebook asistente_deportivo_rag.ipynb
```

**Con el entorno activado manualmente:**
```bash
# Linux / macOS
source .venv/bin/activate
jupyter notebook asistente_deportivo_rag.ipynb

# Windows
.venv\Scripts\activate
jupyter notebook asistente_deportivo_rag.ipynb
```

Ejecuta las celdas en orden. ChromaDB creará automáticamente la carpeta `chroma_db/` con la base vectorial indexada.

### 6. (Opcional) Ejecutar la interfaz Streamlit

```bash
# Con uv:
uv run streamlit run app.py

# Con entorno activado:
streamlit run app.py
```

---

## **System Prompt - Diseño y Justificación**

```
Eres un asistente experto en nutrición deportiva y planificación del entrenamiento físico.
Tu base de conocimiento incluye información detallada sobre macronutrientes, periodización del entrenamiento, suplementación deportiva, recuperación y estrategias de composición corporal.

*[Ver el prompt completo en el notebook, celda PASO 4]*
```

### Decisiones de diseño justificadas

| Decisión | Justificación técnica |
|----------|----------------------|
| **Sección "CÓMO RESPONDER"** | Instrucciones explícitas sobre cómo usar el contexto RAG reducen las alucinaciones y mejoran la precisión de recuperación |
| **"Prioriza el contexto recuperado"** | Instrucción crítica para RAG: sin ella, el modelo tiende a ignorar el contexto y responder desde su conocimiento paramétrico |
| **Admite incertidumbre explícita** | Siguiendo el principio de honestidad: es preferible decir "no tengo esa información" a inventar datos numéricos de salud |
| **Conciencia del historial** | Instrucción explícita de usar el historial para coherencia entre turnos, que es el mecanismo de memoria del agente |
| **Temperatura = 0.3** | Baja temperatura para respuestas más deterministas y precisas; valores altos generan variabilidad innecesaria en información nutricional |
| **Disclaimer médico** | Las recomendaciones de salud tienen implicaciones legales y éticas; el aviso de consultar profesionales es obligatorio |
| **Tono "cercano pero técnico"** | El público objetivo valora tanto la accesibilidad como la precisión científica |

---

## **Decisiones Técnicas**

### Chunking
- **Tamaño:** 800 caracteres con overlap de 100
- **Justificación:** Los documentos tienen tablas y párrafos densos; chunks pequeños (< 400 chars) partirían las tablas perdiendo contexto; chunks grandes (> 1200 chars) reducen la precisión del retrieval

### Embeddings
- **Modelo:** `gemini-embedding-001` (Google)
- **Dimensiones:** 3,072
- **Justificación:** Modelo de incrustación de texto de alto rendimiento de Google (lanzado en octubre de 2025) que ofrece soporte multilingüe avanzado para búsqueda, recuperación y clasificación semántica

### Retrieval
- **Top-k:** 4 chunks por consulta
- **Tipo:** Similarity search (cosine distance)
- **Justificación:** 4 chunks ofrecen suficiente contexto sin superar el límite de tokens del prompt; más chunks aumentan el ruido

### LLM
- **Modelo:** `gemini-1.5-flash`
- **Justificación:** Equilibrio óptimo entre velocidad, coste y calidad; gemini-1.5-pro sería excesivo para este caso de uso

### Memoria
- **Mecanismo:** `add_messages` reducer de LangGraph en el estado del grafo
- **Justificación:** El estado del grafo persiste entre invocaciones; `add_messages` aplica un reducer que concatena mensajes en lugar de sobreescribirlos

---

## **Estructura del Proyecto** (MODIFICAR)

```
agent-project/
├── docs/                          # PDFs de la base de conocimiento
│   ├── Doc1_Fundamentos_Nutricion_Deportiva.pdf
│   ├── Doc2_Planificacion_Entrenamiento.pdf
│   └── Doc3_Recuperacion_Suplementacion.pdf
├── chroma_db/                     # Base vectorial (generada al ejecutar el notebook)
├── asistente_deportivo_rag.ipynb  # Notebook principal
├── app.py                         # Interfaz Streamlit (bonus)
├── pyproject.toml                 # Dependencias gestionadas con uv
├── requirements.txt               # Exportado desde pyproject.toml (compatibilidad pip)
├── .env                           # API keys (NO subir a git)
├── .gitignore
└── README.md
```

---

## **Dependencias**

El proyecto se gestiona con `uv`. El `pyproject.toml` es la fuente de verdad; el `requirements.txt` se genera a partir de él para compatibilidad con pip.

**`pyproject.toml`** (generado con `uv init` + `uv add`):
```toml
[project]
name = "agent-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=1.5.8",
    "dotenv>=0.9.9",
    "langchain>=1.2.17",
    "langchain-community>=0.4.1",
    "langchain-core>=1.3.2",
    "langchain-google-genai>=4.2.2",
    "langchain-text-splitters>=1.1.2",
    "langgraph>=1.1.10",
    "pypdf>=6.10.2",
]
```

Para regenerar el `requirements.txt` desde `pyproject.toml`:
```bash
uv export --format requirements-txt > requirements.txt
```

---

## 🌐 Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub (sin el `.env` y sin `chroma_db/`)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub y selecciona el repositorio
4. En **Advanced settings → Secrets**, añade:
   ```toml
   GOOGLE_API_KEY = "tu_clave_aqui"
   ```
5. El archivo principal es `app.py`

> **Nota:** La carpeta `chroma_db/` debe estar disponible en el despliegue. Para Streamlit Cloud, se recomienda ejecutar el proceso de indexación como parte del `init` de la app (usando `@st.cache_resource`).

---

*Proyecto Modular - IA Generativa · Máster en Data e IA*
