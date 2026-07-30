# 🐍 Python & Inteligencia Artificial Aplicada

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg?logo=openai&logoColor=white)](https://platform.openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-black.svg?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic--AI-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-red.svg)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-PostgreSQL-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-Package--Manager-de5fe9.svg)](https://github.com/astral-sh/uv)

Repositorio con el código fuente y proyectos prácticos desarrollados durante el curso de **Python & Inteligencia Artificial Aplicada** de **DevTalles**.

---

## 📌 Tabla de Contenidos

1. [Visión General del Curso](#-visión-general-del-curso)
2. [Índice del Curso](#-índice-del-curso)
   - [Sección 1: Introducción al curso](#1---introducción-al-curso)
   - [Sección 2: Fundamentos](#2---fundamentos)
   - [Sección 3: Prompts para desarrolladores](#3---prompts-para-desarrolladores)
   - [Sección 4: Técnicas de Prompts avanzados (JSON, Function calling)](#4---técnicas-de-prompts-avanzados-json-function-calling)
   - [Sección 5: Memoria y Contexto - RAG (Retrieval Augmented Generation)](#5---memoria-y-contexto---rag-retrieval-augmented-generation)
   - [Sección 6: Proyecto - RAG - Chatea con PDFs](#6---proyecto---rag---chatea-con-pdfs)
   - [Sección 7: Introducción a LangChain](#7---introducción-a-langchain)
   - [Sección 8: LangChain memoria persistente](#8---langchain-memoria-persistente)
   - [Sección 9: LangChain + RAG](#9---langchain--rag)
   - [Sección 10: LangGraph](#10---langgraph)
   - [Sección 11: Despedida del curso](#11---despedida-del-curso)
3. [🚀 Estructura del Repositorio](#-estructura-del-repositorio)
4. [🛠️ Requisitos Previos e Instalación](#️-requisitos-previos-e-instalación)
5. [🖥️ Ejecución de Proyectos y Demos](#️-ejecución-de-proyectos-y-demos)

---

## 💡 Visión General del Curso

En este curso se aborda la integración de modelos de lenguaje (LLMs) en aplicaciones reales con Python, pasando desde las bases de las APIs y la ingeniería de prompts hasta arquitecturas avanzadas como **RAG (Retrieval-Augmented Generation)**, **LangChain (LCEL)** y flujos de agentes estado a estado con **LangGraph**.

### 🌟 Proyectos y Hitos Principales
- 💬 **CLI Chatbot**: Interfaz de comandos interactiva con cálculo de tokens y estimación de costos en tiempo real.
- 🛠️ **Function Calling & Agent Tooling**: Invocación de herramientas personalizadas y consumo de APIs externas (e.g. extractor de noticias).
- 📄 **RAG "Chatea con PDFs"**: Indexación vectorial semántica con ChromaDB, splitting de documentos PDF y consulta por similitud de coseno.
- 🧠 **Memoria Persistente**: Gestión de sesiones de chat conectadas a bases de datos relacionales (SQLite y PostgreSQL en Docker).
- 🕸️ **Agentes con LangGraph**: Grafos orientados a estados (Nodes, Edges condicionales y Memory Checkpointers) para respuestas precisas y controladas.

---

## 📚 Índice del Curso

<details open>
<summary><b>1 - Introducción al curso</b></summary>

- [x] 1. Bienvenida al curso
- [x] 2. ¿Cómo funcionará el curso?
- [x] 3. ¿Cómo hacer preguntas?
- [x] 4. Requisitos del curso y software a utilizar
- [x] 5. Instalaciones recomendadas
</details>

<details open>
<summary><b>2 - Fundamentos</b></summary>

- [x] 6. Introducción a la sección
- [x] 8. ¿Qué vamos a construir?
- [x] 9. IA Aplicada vs Machine Learning
- [x] 10. Creando proyecto
- [x] 11. Obtener APIKey de OpenAI
- [x] 12. Tu primera llamada a la API
- [x] 13. Solución al ejercicio y uso de tokens
- [x] 14. Manejo de errores con la API
- [x] 15. Temperature
- [x] 16. Proyecto - CLI Chatbot - Crear clase
- [x] 17. Proyecto - CLI Chatbot - Obtener costos
- [x] 18. Proyecto - CLI Chatbot - Función main
</details>

<details open>
<summary><b>3 - Prompts para desarrolladores</b></summary>

- [x] 20. Introducción a la sección
- [x] 22. ¿Qué vamos a construir?
- [x] 23. Prompts para desarrolladores
- [x] 24. Rol User
- [x] 25. Rol System
- [x] 26. Rol Assistant
- [x] 27. Aplicando roles y prompts
- [x] 28. Helper Creación de cliente
- [x] 29. Técnicas de prompting
- [x] 30. Zero-shot
- [x] 31. Few-shot
- [x] 32. Chain-of-Thought
- [x] 33. Prompt Templates - Creando template
- [x] 34. Prompt Templates - Ejecutar prompt
</details>

<details open>
<summary><b>4 - Técnicas de Prompts avanzados (JSON, Function calling)</b></summary>

- [x] 36. Introducción a la sección
- [x] 38. ¿Qué vamos a construir?
- [x] 39. Retornar formato texto
- [x] 40. Retorno formato JSON
- [x] 41. Extractor de noticias
- [x] 42. Servicio - Obtener noticias por API
- [x] 43. Extractor de noticias por API
- [x] 44. Function Calling - Introducción
- [x] 45. Function Calling - Tools y función principal
- [x] 46. Function Calling - Dispatcher y loop de ejecución
- [x] 47. Function Calling - Ejecutando tool
- [x] 48. Function Calling - Servicio API
- [x] 49. Function Calling - Utilizando servicio
</details>

<details open>
<summary><b>5 - Memoria y Contexto - RAG (Retrieval Augmented Generation)</b></summary>

- [x] 51. Introducción a la sección
- [x] 53. ¿Qué vamos a construir?
- [x] 54. Introducción a RAG
- [x] 55. El problema de la ventana de contexto
- [x] 56. Embeddings
- [x] 57. Similitud coseno
- [x] 58. Similitud coseno - parte 2
- [x] 59. Demostración de búsqueda semántica
- [x] 60. Bases de datos vectoriales
- [x] 61. ChromaDB - Teoría
- [x] 62. ChromaDB - Configurando la base de datos
- [x] 63. ChromaDB - Agregar documentos
- [x] 64. ChromaDB - Buscar documentos similares
- [x] 65. ChromaDB - Demo completa
- [x] 66. ChromaDB - Analizando base de datos vectorial
- [x] 67. RAG pipeline - Crear proyecto
- [x] 68. RAG pipeline - Indexación
- [x] 69. RAG pipeline - Consulta
- [x] 70. RAG pipeline - Respuesta - Contexto
- [x] 71. RAG pipeline - Generar respuesta con el LLM
- [x] 72. RAG pipeline - Demo completa
</details>

<details open>
<summary><b>6 - Proyecto - RAG - Chatea con PDFs</b></summary>

- [x] 74. Introducción a la sección
- [x] 76. ¿Qué vamos a construir?
- [x] 77. Imports y configuración
- [x] 78. PDFProcessor
- [x] 79. IndexRegistry - Guardar de archivo a memoria
- [x] 80. IndexRegistry - Guardar de memoria a archivo 
- [x] 81. IndexRegistry - Métodos de indexado y properties
- [x] 82. ChatWithPDFs - Constructor
- [x] 83. Indexar PDFs - Verificar PDFs indexados
- [x] 84. Indexar PDFs - Generar chunks de PDFs
- [x] 85. Mostrar estatus
- [x] 86. Método Chat
- [x] 87. Método Chat - Parte 2
- [x] 88. Main y Demo
</details>

<details open>
<summary><b>7 - Introducción a LangChain</b></summary>

- [x] 90. Introducción a la sección
- [x] 92. ¿Qué vamos a construir?
- [x] 93. Introducción a LangChain
- [x] 94. Arquitectura del proyecto
- [x] 95. LangChain charla rápida (opcional)
- [x] 96. UV Gestor de paquetes
- [x] 97. Creando proyecto y primeras instalaciones
- [x] 98. Archivo de configuración
- [x] 99. Cliente LLM (centralizado)
- [x] 100. Demo LCEL - Main Function
- [x] 101. Cadena simple
- [x] 102. Inspección de pasos
- [x] 103. Batch
- [x] 104. Streaming
- [x] 105. Passthrough
</details>

<details open>
<summary><b>8 - LangChain memoria persistente</b></summary>

- [x] 107. Introducción a la sección
- [x] 109. ¿Qué vamos a construir?
- [x] 110. Memoria persistente
- [x] 111. Interfaz base para memoria
- [x] 112. Memoria usando SQLite
- [x] 113. Instalación PostgreSQL con Docker
- [x] 115. Memoria usando PostgreSQL
- [x] 116. Módulo de memoria para importación
- [x] 117. Cadena base del asistente
- [x] 118. Demo Chatbot - Construir chatbot
- [x] 119. Demo Chatbot - Chat session
- [x] 120. Demo Chatbot - Chat session - parte 2
- [x] 121. Demo Chatbot - Main y pruebas
- [x] 122. Demo Chatbot - Pruebas con PostgreSQL
</details>

<details open>
<summary><b>9 - LangChain + RAG</b></summary>

- [x] 124. Introducción a la sección
- [x] 126. ¿Qué vamos a construir?
- [x] 127. Embeddings - Vector Store
- [x] 128. Pipeline RAG - Format Docs
- [x] 129. Pipeline RAG con LangChain
- [x] 130. Document loader - Cargar archivo
- [x] 131. Document loader - Cargar directorio
- [x] 132. Document loader - Split documents
- [x] 133. Demo RAG - Indexar documentos
- [x] 134. Demo RAG - Fuentes consultadas
- [x] 135. Demo RAG - Comandos especiales
- [x] 136. Demo RAG - Generar respuesta
</details>

<details open>
<summary><b>10 - LangGraph</b></summary>

- [x] 138. Introducción a la sección
- [x] 140. ¿Qué vamos a construir?
- [x] 141. ¿Qué es LangGraph?
- [x] 142. Estados
- [x] 143. Nodo - Analizar pregunta
- [x] 144. Nodo - Recuperar documentos
- [x] 145. Nodo - Generar respuesta
- [x] 146. Función de decisión para Edge Conditional
- [x] 147. Construir agente RAG
- [x] 148. Módulo de grafos LangGraph
- [x] 149. Demo LangGraph - Setup Vectorstore
- [x] 150. Demo LangGraph - Memory Backend
- [x] 151. Demo LangGraph - Gestión de sesiones
- [x] 152. Demo LangGraph - Cargar historial
- [x] 153. Demo LangGraph - Guardar mensajes
- [x] 154. Demo LangGraph - Run Chat - Comandos especiales
- [x] 155. Demo LangGraph - Run Chat - Invocar Grafo
- [x] 156. Demo LangGraph - Run Chat - Respuesta
- [x] 157. Demo LangGraph - Main
- [x] 158. Demo LangGraph - Solucionando errores
- [x] 159. Demo LangGraph - Probando flujo completo
</details>

<details open>
<summary><b>11 - Despedida del curso</b></summary>

- [x] 161. Despedida del curso
</details>

---

## 🚀 Estructura del Repositorio

```directory
.
├── data/                       # Archivos fuente (PDFs de prueba, etc.)
│   └── files/pdfs/
├── src/                        # Proyecto base: Fundamentos OpenAI, Prompts y RAG nativo
│   ├── chatbot_cli.py          # Chatbot interactivo CLI
│   ├── hello_ia.py             # Primera llamada a OpenAI API
│   ├── hello_error_managment.py # Manejo de excepciones y errores de API
│   ├── hello_languages.py      # Pruebas iniciales de prompts e idioma
│   ├── prompts/                # Prompts avanzados (System, User, JSON, Function Calling)
│   └── rag/                    # Pipeline RAG nativo (Embeddings, ChromaDB, Chat with PDFs)
└── python-ia-langchain/        # Proyecto LangChain & LangGraph (gestionado con uv)
    ├── docker-compose.yml      # Servicio PostgreSQL para memoria persistente
    ├── pyproject.toml          # Configuración de dependencias (uv)
    └── src/
        └── langchain_section/
            ├── chains/         # LCEL Chains (Base y RAG)
            ├── config/         # Configuración global y cliente LLM
            ├── core/           # Loaders y Document Splitters
            ├── memory/         # Adapters de memoria (SQLite & Postgres)
            ├── graphs/         # Agente RAG orientado a estados (LangGraph)
            └── demos/          # Puntos de entrada ejecutables (LCEL, Memory, RAG, LangGraph)
```

---

## 🛠️ Requisitos Previos e Instalación

### 1. Clonar el Repositorio e Instalar Dependencias

Asegurate de contar con **Python 3.12+**. Se recomienda utilizar [uv](https://github.com/astral-sh/uv) como gestor rápido de paquetes:

```bash
# Instalación con uv en el módulo de LangChain
cd python-ia-langchain
uv sync
```

O si utilizás el entorno de Python estándar en la raíz:

```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/macOS
pip install -r requirements.txt
```

### 2. Configuración de Variables de Entorno

Copiá el archivo de ejemplo `.env.example` a `.env` e ingresá tu API Key de OpenAI:

```bash
cp .env.example .env
```

Configuración requerida en `.env`:
```env
OPENAI_API_KEY=tu_api_key_aqui
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=langchain_memory
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 3. PostgreSQL con Docker (Opcional para Memoria Persistente)

Para levantar la base de datos PostgreSQL requerida en los ejercicios de memoria persistente:

```bash
cd python-ia-langchain
docker compose up -d
```

---

## 🖥️ Ejecución de Proyectos y Demos

### 1. Demos Fundamentos (OpenAI Directo)
```bash
# Probar llamada básica
python src/hello_ia.py

# Iniciar CLI Chatbot
python src/chatbot_cli.py

# Sistema RAG "Chatea con PDFs"
python src/rag/chat_with_pdfs.py
```

### 2. Demos LangChain & LangGraph (con `uv`)
```bash
cd python-ia-langchain

# Demo de LCEL (Streaming, Batch, Passthrough)
uv run python src/langchain_section/demos/demo_lcel.py

# Demo de Memoria Persistente (SQLite / Postgres)
uv run python src/langchain_section/demos/demo_memory.py

# Demo de RAG con LangChain
uv run python src/langchain_section/demos/demo_rag.py

# Demo de Agente RAG interactivo con LangGraph
uv run python src/langchain_section/demos/demo_langgraph.py
```

---

## 👨‍💻 Créditos y Reconocimientos

- **Curso**: Python y IA Aplicada
- **Instructor / Plataforma**: [DevTalles](https://devtalles.com/)
