# 🏋️ Asistente Experto en Nutrición y Entrenamiento Deportivo

> Proyecto Modular - IA Generativa | Máster en Data e IA  
> *Elaborado por Soraya Malpica*  
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
  │           └── Gemini text-embedding-001
  │
  └── Nodo: generate
        └── Rotación de modelos Gemini (free tier)
              ├── gemini-2.5-flash      (prioritario, 20 RPD)
              └── gemini-2.5-flash-lite (fallback, 1.000 RPD)
                    ├── System Prompt (agente experto)
                    ├── Contexto RAG (chunks recuperados)
                    └── Historial de conversación (memoria)
```

**Flujo de datos:** `START → retrieve → generate → END`

---

## **Instalación y Ejecución**

### Requisitos previos

- Python 3.12+
- API Key de Google Gemini ([obtener aquí](https://aistudio.google.com/app/apikey))
- `uv` instalado ([instrucciones](https://docs.astral.sh/uv/getting-started/installation/)) - recomendado

### 1. Clonar el repositorio

```bash
git clone https://github.com/somm14/agent-project
cd 01_desarrollo_asistente.ipynb
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
GEMINI_API_KEY=tu_clave_de_gemini_aqui
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
├── 01_desarrollo_asistente.ipynb
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
jupyter notebook 01_desarrollo_asistente.ipynb

# Windows
.venv\Scripts\activate
jupyter notebook asistente_deportivo_rag.ipynb
```

Ejecuta las celdas en orden. ChromaDB creará automáticamente la carpeta `chroma_db/` con la base vectorial indexadala primera vez. En ejecuciones posteriores, la base ya está lista y no necesita regenerarse.

> ⚠️ **Nota sobre los límites del free tier:** Los contadores de solicitudes se reinician al reiniciar el kernel. Si alcanzas el límite diario real (20 req para `gemini-2.5-flash`), el agente rotará automáticamente a `gemini-2.5-flash-lite`. Si se agotan ambos, mostrará un mensaje indicando cuándo se restablece el límite (medianoche hora del Pacífico).

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
| **Rol de entrenador experto** | Define un perfil de autoridad en el dominio sin ser médico, equilibrando credibilidad con responsabilidad |
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
- **Separadores:** `['\n\n', '\n', '. ', ' ', '']` — prioriza separación semántica por párrafos
- **Resultado real:** 77 chunks con tamaño medio de 626 caracteres
- **Justificación:** Los documentos tienen tablas y párrafos densos; chunks 
pequeños (< 400 chars) partirían las tablas perdiendo contexto; chunks grandes (> 1200 chars) reducen la precisión del retrieval

### Embeddings
- **Modelo:** `gemini-embedding-001` (Google)
- **Justificación:** Modelo de embeddings de última generación de Google con soporte multilingüe avanzado, optimizado para búsqueda semántica y recuperación de información en español.

### Retrieval
- **Top-k:** 4 chunks por consulta
- **Tipo:** Similarity search (cosine distance)
- **Almacenamiento:** ChromaDB persistente en `./chroma_db/`
- **Justificación:** 4 chunks ofrecen suficiente contexto sin superar el límite de tokens del prompt; más chunks aumentan el ruido en la respuesta.

### LLM y Rotación de Modelos (Free Tier)

El agente implementa rotación automática de modelos para gestionar los límites del free tier de Gemini. Cuando un modelo alcanza su cuota diaria real (detectada mediante el error `429 RESOURCE_EXHAUSTED` de la API), rota automáticamente al siguiente sin interrumpir la conversación:

| Prioridad | Modelo | RPD free tier | Rol |
|-----------|--------|--------------|-----|
| 1º | `gemini-2.5-flash` | 20 req/día | Prioritario - mayor calidad de respuesta |
| 2º | `gemini-2.5-flash-lite` | 1.000 req/día | Fallback - mayor disponibilidad |

Si ambos modelos se agotan, el agente devuelve un mensaje informativo al usuario indicando el tiempo restante hasta el reset (medianoche, hora del Pacífico). Los contadores internos del notebook no persisten entre reinicios del kernel; la cuota real la gestiona Google en su backend.

### Memoria
- **Mecanismo:** `add_messages` reducer de LangGraph en el estado del grafo
- **Justificación:** El estado del grafo persiste entre invocaciones; `add_messages` aplica un reducer que concatena mensajes en lugar de sobreescribirlos
- **Decisión clave:** En el historial se guarda la pregunta limpia del usuario (sin el bloque de contexto RAG), para no inflar el contexto en turnos sucesivos y mantener la ventana de tokens bajo control

---

## **Estructura del Proyecto** (MODIFICAR)

```
agent-project/
├── docs/                          # PDFs de la base de conocimiento
│   ├── Doc1_Fundamentos_Nutricion_Deportiva.pdf
│   ├── Doc2_Planificacion_Entrenamiento.pdf
│   └── Doc3_Recuperacion_Suplementacion.pdf
├── chroma_db/                     # Base vectorial (generada al ejecutar el notebook)
|   ├── chroma.sqlite3                 # Metadatos, textos e índice de colecciones
│   └── [uuid]/                        # Índice HNSW para búsqueda vectorial eficiente
│       ├── header.bin
│       ├── data_level0.bin
│       ├── length.bin
│       └── link_list.bin
├── 01_desarrollo_asistente.ipynb  # Notebook principal
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
