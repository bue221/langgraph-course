# LangGraph Learning Course

Este repositorio contiene una colección de notebooks y ejemplos prácticos para aprender **LangGraph**, la librería de los creadores de LangChain que permite crear flujos complejos usando grafos con modelos de inteligencia artificial.

## 📋 Descripción

LangGraph permite construir aplicaciones de IA robustas mediante:

- **State**: Contexto compartido del grafo que actúa como memoria
- **Nodes**: Funciones que procesan información y modifican el estado
- **Edges**: Conexiones entre nodos que definen el flujo de ejecución
- **Conditional Edges**: Caminos condicionales basados en lógica específica
- **Tools**: Herramientas externas que pueden usar los agentes

## 🚀 Instalación

### Requisitos

- Python >= 3.12
- Poetry (recomendado) o pip

### Configuración del entorno

1. **Clona el repositorio:**

```bash
git clone <repository-url>
cd langgraph
```

2. **Instala las dependencias:**

```bash
# Con Poetry (recomendado)
poetry install

# Con pip
pip install -e .
```

3. **Configura las variables de entorno:**

```bash
# Crea un archivo .env en la raíz del proyecto
cp .env.example .env
# Edita .env con tus claves de API
```

### Variables de entorno requeridas

```
OPENAI_API_KEY=tu_clave_openai
LANGSMITH_API_KEY=tu_clave_langsmith  # opcional
```

## 📚 Estructura del Proyecto

```
langgraph/
├── notebooks/
│   ├── basic/                    # Ejemplos básicos
│   │   ├── beginners_steps.ipynb     # Conceptos fundamentales y primer "Hola Mundo"
│   │   ├── sequentials_nodes.ipynb   # Nodos secuenciales básicos
│   │   ├── conditional_edges.ipynb   # Edges condicionales
│   │   ├── looping_graph.ipynb       # Grafos con bucles
│   │   └── guess_number_game.ipynb   # Juego interactivo de adivinanza
│   └── with_llms/               # Ejemplos avanzados con LLMs
│       ├── first_steps.ipynb         # Chatbot básico con LangGraph
│       └── re_act_agents.ipynb       # Agentes ReAct
├── projects/                    # Proyectos prácticos
├── langgraph.json              # Configuración de LangGraph
└── pyproject.toml              # Dependencias del proyecto
```

## 🎯 Guía de Aprendizaje

### 1. Conceptos Básicos (`notebooks/basic/`)

#### 📖 **beginners_steps.ipynb**

- Fundamentos de Python: Typing, Lambdas
- Conceptos core de LangGraph: State, Nodes, Graphs
- Tu primer "Hola Mundo" con LangGraph

#### ⛓️ **sequentials_nodes.ipynb**

- Creación de flujos secuenciales
- Manejo del estado entre nodos
- Patrones básicos de ejecución

#### 🔀 **conditional_edges.ipynb**

- Implementación de lógica condicional
- Múltiples caminos de ejecución
- Toma de decisiones basada en el estado

#### 🔄 **looping_graph.ipynb**

- Grafos con bucles y recursión
- Control de flujo avanzado
- Condiciones de salida

#### 🎮 **guess_number_game.ipynb**

- Proyecto práctico interactivo
- Integración de conceptos aprendidos
- Manejo de entrada del usuario

### 2. LLMs y Agentes (`notebooks/with_llms/`)

#### 🤖 **first_steps.ipynb**

- Chatbot básico con OpenAI
- Integración de LLMs en grafos
- Manejo de mensajes y conversaciones

#### 🧠 **re_act_agents.ipynb**

- Implementación del patrón ReAct
- Agentes con capacidad de razonamiento
- Uso de herramientas externas

## 🛠️ Uso

### Ejecutar notebooks

```bash
# Activa el entorno virtual
poetry shell

# Inicia Jupyter
jupyter notebook notebooks/
```

### Ejecutar con LangGraph CLI

```bash
# Ejecutar un grafo específico
langgraph dev

# Usar el grafo configurado en langgraph.json
langgraph run my_agent
```

## 📦 Dependencias Principales

- **langgraph**: Framework principal para grafos de IA
- **langchain**: Componentes base para LLMs
- **langchain-openai**: Integración con OpenAI
- **langsmith**: Herramientas de monitoreo y debugging
- **python-dotenv**: Manejo de variables de entorno
- **langgraph-swarm**: Extensiones para múltiples agentes

## 🎯 Ejemplos Rápidos

### Hola Mundo Básico

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    message: str

def greetings(state: AgentState) -> AgentState:
    state["message"] = f"Hello, {state['message']}!"
    return state

graph = StateGraph(AgentState)
graph.add_node("greetings", greetings)
graph.add_edge(START, "greetings")
graph.add_edge("greetings", END)

workflow = graph.compile()
result = workflow.invoke({"message": "World"})
```

### Chatbot con LLM

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = init_chat_model("openai:gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

## 🤝 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'feat: add nueva caracteristica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Andrés Camilo Plaza Jiménez**

- Email: camiloplaza3@gmail.com

---

## 🔗 Enlaces Útiles

- [Documentación oficial de LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

**¡Feliz aprendizaje con LangGraph! 🚀**
