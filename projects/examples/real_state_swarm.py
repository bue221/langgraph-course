import json
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Dict, List, Optional
from langchain_core.runnables.utils import Output
from langchain_core.tools import InjectedToolCallId

# Imports para Command y Send
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ===============================
# ESTADO ENRIQUECIDO CON COMMAND SUPPORT
# ===============================

class RealEstateSwarmState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Información del cliente
    client: Optional[Dict]
    
    # Estado de propiedades
    current_property_id: Optional[str]
    current_property_data: Optional[Dict]
    user_confirmed_interest: bool
    confirmation_timestamp: Optional[str]
    
    # Proyectos y visitas
    projects: List[Dict]
    active_project_id: Optional[str]
    visits: List[Dict]
    pending_visit_request: Optional[Dict]
    
    # Contexto de conversación
    conversation_context: Dict
    last_action: Optional[str]
    escalation_requested: bool
    
    # Métricas y analytics
    interaction_count: int
    session_start_time: str
    
    # Agente activo
    active_agent: Optional[str]

# ===============================
# MOCK DATABASE CON EVENTOS
# ===============================

class MockDatabase:
    def __init__(self):
        self.clients = {}
        self.projects = {}
        self.properties = {}
        self.visits = {}
        self.events = []  # Para tracking de eventos
        self._init_mock_data()
    
    def _init_mock_data(self):
        # Propiedades mock
        mock_properties = {
            "prop_001": {
                "id": "prop_001",
                "url": "https://example.com/property/luxury-apartment-zona-rosa",
                "title": "Apartamento de Lujo en Zona Rosa",
                "price": 850000000,
                "property_type": "apartamento",
                "bedrooms": 3,
                "bathrooms": 2,
                "area": 120,
                "zone": "Zona Rosa",
                "address": "Calle 93 #15-47, Zona Rosa, Bogotá",
                "description": "Hermoso apartamento moderno con vista panorámica",
                "features": ["balcón", "parqueadero", "gimnasio", "seguridad 24h"],
                "available": True
            },
            "prop_002": {
                "id": "prop_002", 
                "url": "https://example.com/property/casa-chapinero",
                "title": "Casa Familiar en Chapinero",
                "price": 650000000,
                "property_type": "casa",
                "bedrooms": 4,
                "bathrooms": 3,
                "area": 180,
                "zone": "Chapinero",
                "address": "Carrera 13 #63-28, Chapinero, Bogotá",
                "description": "Casa tradicional ideal para familias",
                "features": ["jardín", "garaje doble", "chimenea", "estudio"],
                "available": True
            }
        }
        
        self.properties = mock_properties
    
    def get_property_by_url(self, url: str) -> Optional[Dict]:
        for prop in self.properties.values():
            if prop["url"] == url:
                return prop
        return None
    
    def get_property_by_id(self, property_id: str) -> Optional[Dict]:
        return self.properties.get(property_id)
    
    def log_event(self, event_type: str, data: Dict):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        self.events.append(event)
        return event
    
    def save_project(self, project: Dict):
        self.projects[project["id"]] = project
        self.log_event("project_created", {"project_id": project["id"]})
    
    def save_visit(self, visit: Dict):
        self.visits[visit["id"]] = visit
        self.log_event("visit_scheduled", {"visit_id": visit["id"]})

# Instancia global
mock_db = MockDatabase()

# ===============================
# COMMAND TOOLS - MODIFICAN ESTADO DIRECTAMENTE CON TOOL MESSAGES
# ===============================

@tool
def analyze_rejection_and_update_preferences(tool_call_id: Annotated[str, InjectedToolCallId],rejection_reason: str, property_data: str = "") -> Command:
    """
    Analiza por qué el cliente rechazó una propiedad Y actualiza preferencias
    Retorna Command para actualizar el perfil del cliente
    """
    try:
        # Analizar la razón del rechazo
        rejection_analysis = {
            "reason": rejection_reason,
            "rejected_at": datetime.now().isoformat(),
            "property_context": json.loads(property_data) if property_data else {}
        }
        
        # Inferir preferencias basadas en el rechazo
        inferred_preferences = {}
        
        if "muy caro" in rejection_reason.lower() or "presupuesto" in rejection_reason.lower():
            inferred_preferences["budget_concern"] = True
            inferred_preferences["suggested_action"] = "lower_budget_search"
            
        elif "zona" in rejection_reason.lower() or "ubicación" in rejection_reason.lower():
            inferred_preferences["location_concern"] = True
            inferred_preferences["suggested_action"] = "alternative_zones"
            
        elif "pequeño" in rejection_reason.lower() or "área" in rejection_reason.lower():
            inferred_preferences["size_concern"] = True
            inferred_preferences["suggested_action"] = "larger_properties"
            
        elif "tipo" in rejection_reason.lower():
            inferred_preferences["property_type_concern"] = True
            inferred_preferences["suggested_action"] = "different_property_type"
        
        # Log del evento
        mock_db.log_event("property_rejected", {
            "rejection_reason": rejection_reason,
            "inferred_preferences": inferred_preferences
        })
        
        return Command(
            update={
                "messages": [ToolMessage(content="REJECTION_ANALYZED", tool_call_id=tool_call_id)],
                "last_action": "rejection_analyzed",
                "conversation_context": {
                    "rejection_analysis": rejection_analysis,
                    "inferred_preferences": inferred_preferences,
                    "profiling_status": "preferences_updated",
                    "next_step": "create_personalized_project"
                },
                "client": {
                    # Actualizar preferencias del cliente basadas en el rechazo
                    "rejection_history": [rejection_analysis],
                    "inferred_preferences": inferred_preferences
                }
            }
        )
        
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(content=f"ANALYSIS_ERROR: {str(e)}", tool_call_id=tool_call_id)],
                "last_action": "rejection_analysis_failed",
                "conversation_context": {
                    "error": str(e),
                    "fallback_action": "manual_profiling_required"
                }
            }
        )

@tool
def get_property_info_and_set_context(url: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Obtiene información de propiedad Y actualiza el estado con los datos
    Retorna Command para modificar el estado directamente
    """
    property_data = mock_db.get_property_by_url(url)
    
    if not property_data:
        return Command(
            update={
                "messages": [ToolMessage(content="PROPERTY_NOT_FOUND", tool_call_id=tool_call_id)],
                "last_action": "property_search_failed",
                "conversation_context": {
                    "error": "Propiedad no encontrada",
                    "searched_url": url
                }
            }
        )
    
    # Log del evento
    mock_db.log_event("property_viewed", {
        "property_id": property_data["id"],
        "url": url
    })
    
    # Actualizar estado con información completa
    return Command(
        goto="PropertyAgent",
        graph=Command.PARENT,
        update={
            "messages": [ToolMessage(content="PROPERTY_LOADED", tool_call_id=tool_call_id)],
            "current_property_id": property_data["id"],
            "current_property_data": property_data,
            "last_action": "property_loaded",
            "conversation_context": {
                "property_title": property_data["title"],
                "property_price": property_data["price"],
                "property_zone": property_data["zone"]
            }
        }
    )

@tool
def confirm_property_interest(tool_call_id: Annotated[str, InjectedToolCallId], interest_level: str, notes: str = "") -> Command:
    """
    Confirma el interés del usuario en la propiedad actual
    Actualiza el estado con la confirmación
    """
    confirmation_data = {
        "interest_level": interest_level,  # "high", "medium", "low"
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }
    
    # Log del evento
    mock_db.log_event("interest_confirmed", confirmation_data)
    
    return Command(
        update={
            "messages": [ToolMessage(content="INTEREST_CONFIRMED", tool_call_id=tool_call_id)],
            "user_confirmed_interest": True,
            "confirmation_timestamp": confirmation_data["timestamp"],
            "conversation_context": {
                **confirmation_data,
                "next_suggested_action": "schedule_visit" if interest_level == "high" else "explore_alternatives"
            },
            "last_action": "interest_confirmed"
        }
    )

@tool
def create_project_and_set_active(project_data: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Crea un proyecto Y lo establece como activo en el estado
    """
    try:
        data = json.loads(project_data)
        project = {
            "id": f"proj_{uuid.uuid4().hex[:8]}",
            "client_id": data["client_id"],
            "name": data["name"],
            "budget_min": data["budget_min"],
            "budget_max": data["budget_max"],
            "property_type": data["property_type"],
            "zones": data.get("zones", []),
            "bedrooms": data.get("bedrooms"),
            "bathrooms": data.get("bathrooms"),
            "area_min": data.get("area_min"),
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        
        mock_db.save_project(project)
        
        return Command(
            update={
                "messages": [ToolMessage(content="PROJECT_CREATED", tool_call_id=tool_call_id)],
                "active_project_id": project["id"],
                "projects": [project],  # Se agregará a la lista existente
                "last_action": "project_created",
                "conversation_context": {
                    "created_project": project["name"],
                    "project_id": project["id"]
                }
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(content=f"PROJECT_ERROR: {str(e)}", tool_call_id=tool_call_id)],
                "last_action": "project_creation_failed",
                "conversation_context": {
                    "error": str(e)
                }
            }
        )

@tool
def request_visit_and_update_state(
    property_id: str,
    preferred_date: str,
    preferred_time: str,
    notes: str,
    tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Procesa solicitud de visita Y actualiza estado con la solicitud pendiente
    """
    try:
        visit_request = {
            "property_id": property_id,
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "client_notes": notes,
            "status": "pending_confirmation",
            "requested_at": datetime.now().isoformat()
        }
        
        return Command(
            update={
                "messages": [ToolMessage(content="VISIT_REQUESTED", tool_call_id=tool_call_id)],
                "pending_visit_request": visit_request,
                "last_action": "visit_requested",
                "conversation_context": {
                    "visit_property_id": visit_request["property_id"],
                    "preferred_date": visit_request["preferred_date"],
                    "status": "awaiting_confirmation"
                }
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(content=f"VISIT_ERROR: {str(e)}", tool_call_id=tool_call_id)],
                "last_action": "visit_request_failed",
                "conversation_context": {"error": str(e)}
            }
        )

@tool
def confirm_visit_and_schedule(confirmation: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Confirma y agenda la visita pendiente
    """
    if confirmation.lower() in ["sí", "si", "yes", "confirmar", "ok"]:
        visit_id = f"visit_{uuid.uuid4().hex[:8]}"
        
        # Crear visita confirmada
        visit = {
            "id": visit_id,
            "property_id": "current_property",  # Se tomará del estado
            "date": "fecha_solicitada",  # Se tomará del pending_visit_request
            "time": "hora_solicitada",
            "status": "confirmed",
            "confirmed_at": datetime.now().isoformat()
        }
        
        mock_db.save_visit(visit)
        
        return Command(
            update={
                "messages": [ToolMessage(content="VISIT_CONFIRMED", tool_call_id=tool_call_id)],
                "visits": [visit],  # Se agregará a la lista
                "pending_visit_request": None,  # Limpiar la solicitud pendiente
                "last_action": "visit_confirmed",
                "conversation_context": {
                    "visit_id": visit_id,
                    "status": "confirmed"
                }
            }
        )
    else:
        return Command(
            update={
                "messages": [ToolMessage(content="VISIT_CANCELLED", tool_call_id=tool_call_id)],
                "pending_visit_request": None,
                "last_action": "visit_cancelled",
                "conversation_context": {
                    "status": "cancelled_by_user"
                }
            }
        )

@tool
def escalate_with_context(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Escala a humano Y prepara todo el contexto necesario
    """
    escalation_id = f"esc_{uuid.uuid4().hex[:8]}"
    
    escalation_context = {
        "escalation_id": escalation_id,
        "escalated_at": datetime.now().isoformat(),
        "reason": "user_request",
        "estimated_wait": "5-10 minutes"
    }
    
    mock_db.log_event("escalated_to_human", escalation_context)
    
    return Command(
        update={
            "messages": [ToolMessage(content="ESCALATED_TO_HUMAN", tool_call_id=tool_call_id)],
            "escalation_requested": True,
            "last_action": "escalated_to_human",
            "conversation_context": escalation_context
        }
    )

# ===============================
# TOOLS CON SEND - COMUNICACIÓN ENTRE AGENTES
# ===============================

@tool
def notify_scheduling_agent_of_interest() -> Send:
    """
    Envía notificación al SchedulingAgent cuando hay interés confirmado
    Usa Send para comunicación directa entre agentes
    """
    return Send(
        node="SchedulingAgent",
        arg={
            "messages": [HumanMessage(content="Cliente confirmó interés en propiedad. Proceder con agendamiento.")],
            "trigger": "interest_confirmed",
            "from_agent": "PropertyAgent"
        }
    )

@tool
def send_property_data_to_profiling() -> Send:
    """
    Envía datos de propiedad rechazada al ProfilingAgent
    """
    return Send(
        node="ProfilingAgent", 
        arg={
            "messages": [HumanMessage(content="Cliente no está interesado en la propiedad. Iniciar perfilado.")],
            "trigger": "property_rejected",
            "from_agent": "PropertyAgent"
        }
    )

# ===============================
# AGENTES CON COMMAND TOOLS
# ===============================

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# PropertyAgent mejorado con Command tools
property_agent_with_commands = create_react_agent(
    model,
    tools=[
        get_property_info_and_set_context,  # Command tool
        confirm_property_interest,  # Command tool
        notify_scheduling_agent_of_interest,  # Send tool
        send_property_data_to_profiling,  # Send tool
        create_handoff_tool(agent_name="SchedulingAgent"),
        create_handoff_tool(agent_name="ProfilingAgent"),
        create_handoff_tool(agent_name="RoutingAgent")
    ],
    prompt="""Eres el especialista en análisis de propiedades con capacidades avanzadas.

HERRAMIENTAS ESPECIALES:
- get_property_info_and_set_context: Obtiene info Y actualiza el estado automáticamente
- confirm_property_interest: Registra interés Y actualiza estado con confirmación
- notify_scheduling_agent_of_interest: Envía notificación directa al SchedulingAgent
- send_property_data_to_profiling: Envía datos al ProfilingAgent si no hay interés

IMPORTANTE SOBRE TOOL RESPONSES:
- Los tools devuelven mensajes técnicos como "PROPERTY_LOADED", "INTEREST_CONFIRMED"
- NUNCA uses estos mensajes técnicos como respuesta final
- Usa la información del ESTADO para generar respuestas naturales y atractivas
- Ejemplo: Si ves "PROPERTY_LOADED", di "¡Excelente! He analizado esta propiedad..."

FLUJO:
1. Usa get_property_info_and_set_context para obtener y contextualizar propiedad
2. Presenta características de manera atractiva basándote en current_property_data del estado
3. Si hay interés: usa confirm_property_interest Y notify_scheduling_agent_of_interest
4. Si no hay interés: usa send_property_data_to_profiling

El estado se actualiza automáticamente con tus acciones.""",
    name="PropertyAgent"
)

# ProjectAgent con Command tools
project_agent_with_commands = create_react_agent(
    model,
    tools=[
        create_project_and_set_active,  # Command tool
        create_handoff_tool(agent_name="RoutingAgent"),
        create_handoff_tool(agent_name="EscalationAgent")
    ],
    prompt="""Eres el especialista en proyectos con capacidades de modificación de estado.

HERRAMIENTAS ESPECIALES:
- create_project_and_set_active: Crea proyecto Y lo establece como activo automáticamente

IMPORTANTE SOBRE TOOL RESPONSES:
- Los tools devuelven mensajes técnicos como "PROJECT_CREATED"
- NUNCA uses estos mensajes técnicos como respuesta final
- Usa la información del ESTADO para generar respuestas naturales
- Ejemplo: Si ves "PROJECT_CREATED", di "¡Perfecto! He creado tu proyecto personalizado..."

FLUJO:
1. Recopila información del proyecto
2. Usa create_project_and_set_active para crear Y activar el proyecto
3. El estado se actualiza automáticamente
4. Genera respuesta basada en conversation_context.created_project

Cuando crees un proyecto, el sistema lo marcará como activo automáticamente.""",
    name="ProjectAgent"
)

# SchedulingAgent con Command tools
scheduling_agent_with_commands = create_react_agent(
    model,
    tools=[
        request_visit_and_update_state,  # Command tool
        confirm_visit_and_schedule,  # Command tool
        create_handoff_tool(agent_name="RoutingAgent"),
        create_handoff_tool(agent_name="EscalationAgent")
    ],
    prompt="""Eres el especialista en agendamiento con capacidades de estado avanzadas.

HERRAMIENTAS ESPECIALES:
- request_visit_and_update_state: Procesa solicitud Y actualiza estado con request pendiente
- confirm_visit_and_schedule: Confirma visita Y actualiza estado con visita confirmada

IMPORTANTE SOBRE TOOL RESPONSES:
- Los tools devuelven mensajes técnicos como "VISIT_REQUESTED", "VISIT_CONFIRMED"
- NUNCA uses estos mensajes técnicos como respuesta final
- Usa la información del ESTADO para generar respuestas naturales
- Ejemplo: Si ves "VISIT_CONFIRMED", di "¡Excelente! Tu visita ha sido confirmada para..."

FLUJO:
1. Usa request_visit_and_update_state para procesar solicitud inicial
2. Confirma detalles con el cliente  
3. Usa confirm_visit_and_schedule para finalizar agendamiento
4. El estado se actualiza automáticamente en cada paso
5. Genera respuestas basadas en pending_visit_request y visits del estado

Puedes ver el estado actual de solicitudes pendientes en el contexto.""",
    name="SchedulingAgent"
)

# EscalationAgent con Command
escalation_agent_with_commands = create_react_agent(
    model,
    tools=[
        escalate_with_context,  # Command tool
        create_handoff_tool(agent_name="RoutingAgent")
    ],
    prompt="""Eres el especialista en escalamiento con preparación automática de contexto.

HERRAMIENTAS ESPECIALES:
- escalate_with_context: Escala Y prepara todo el contexto automáticamente

IMPORTANTE SOBRE TOOL RESPONSES:
- Los tools devuelven mensajes técnicos como "ESCALATED_TO_HUMAN"
- NUNCA uses estos mensajes técnicos como respuesta final
- Usa la información del ESTADO para generar respuestas naturales
- Ejemplo: Si ves "ESCALATED_TO_HUMAN", di "He escalado tu consulta a un agente humano..."

FLUJO:
1. Entiende la razón del escalamiento
2. Usa escalate_with_context para procesar Y actualizar estado
3. El contexto completo se prepara automáticamente para el agente humano
4. Genera respuesta basada en conversation_context.escalation_id

El estado se marca automáticamente como escalado.""",
    name="EscalationAgent"
)

# ProfilingAgent con Command tools
profiling_agent_with_commands = create_react_agent(
    model,
    tools=[
        create_project_and_set_active,  # Command tool - reutilizado
        analyze_rejection_and_update_preferences,  # Command tool nuevo
        create_handoff_tool(agent_name="RoutingAgent"),
        create_handoff_tool(agent_name="ProjectAgent"),
        create_handoff_tool(agent_name="EscalationAgent")
    ],
    prompt="""Eres el especialista en perfilado de clientes con capacidades de análisis avanzadas.

HERRAMIENTAS ESPECIALES:
- analyze_rejection_and_update_preferences: Analiza rechazo Y actualiza preferencias automáticamente
- create_project_and_set_active: Crea proyecto basado en perfilado Y lo activa

IMPORTANTE SOBRE TOOL RESPONSES:
- Los tools devuelven mensajes técnicos como "REJECTION_ANALYZED", "PROJECT_CREATED"
- NUNCA uses estos mensajes técnicos como respuesta final
- Usa la información del ESTADO para generar respuestas naturales
- Ejemplo: Si ves "REJECTION_ANALYZED", di "Entiendo tus preferencias. Basándome en tu feedback..."

FLUJO:
1. Recibe información de propiedad rechazada (puede venir vía Send)
2. Usa analyze_rejection_and_update_preferences para entender por qué no le gustó
3. Recopila preferencias reales del cliente
4. Usa create_project_and_set_active para crear proyecto personalizado
5. El estado se actualiza automáticamente con las nuevas preferencias
6. Genera respuestas basadas en conversation_context.inferred_preferences

Tu función es convertir un rechazo en una oportunidad de entender mejor al cliente.""",
    name="ProfilingAgent"
)

# ===============================
# ROUTING AGENT CON ESTADO CONSCIENTE
# ===============================

routing_agent_state_aware = create_react_agent(
    model,
    tools=[
        create_handoff_tool(agent_name="PropertyAgent"),
        create_handoff_tool(agent_name="ProjectAgent"),
        create_handoff_tool(agent_name="SchedulingAgent"),
        create_handoff_tool(agent_name="ProfilingAgent"),
        create_handoff_tool(agent_name="EscalationAgent")
    ],
    prompt="""Eres el coordinador consciente del estado del sistema.

INFORMACIÓN DISPONIBLE EN EL ESTADO:
- current_property_id: ID de propiedad actual (si hay)
- user_confirmed_interest: Si el usuario confirmó interés
- pending_visit_request: Solicitud de visita pendiente
- active_project_id: Proyecto activo
- last_action: Última acción realizada
- escalation_requested: Si se solicitó escalamiento
- inferred_preferences: Preferencias inferidas del perfilado

LÓGICA DE ROUTING INTELIGENTE:
- Si hay pending_visit_request → SchedulingAgent
- Si user_confirmed_interest=true y no hay visita → SchedulingAgent  
- Si current_property_id está presente → PropertyAgent puede continuar
- Si escalation_requested=true → EscalationAgent
- Si last_action="rejection_analyzed" → ProfilingAgent para crear proyecto
- URL de propiedad nueva → PropertyAgent
- Crear/editar proyecto → ProjectAgent
- Cliente no interesado o rechazo → ProfilingAgent

Usa la información del estado para dar continuidad inteligente.""",
    name="RoutingAgent",
)

# ===============================
# CREAR SWARM CON COMMAND SUPPORT
# ===============================

def create_command_aware_swarm():
    """Crea Swarm con agentes que usan Command para modificar estado"""
    
    swarm = create_swarm(
        agents=[
            routing_agent_state_aware,
            property_agent_with_commands,
            project_agent_with_commands, 
            scheduling_agent_with_commands,
            profiling_agent_with_commands,
            escalation_agent_with_commands
        ],
        default_active_agent="RoutingAgent",
        state_schema=RealEstateSwarmState,
    )
    
    return swarm

graph = create_command_aware_swarm()

# ===============================
# FUNCIÓN PRINCIPAL CON ESTADO OBSERVABLE
# ===============================

async def run_command_swarm():
    """Ejecuta Swarm con Command tools y estado observable"""
    
    swarm_graph = create_command_aware_swarm()
    memory = MemorySaver()
    app = swarm_graph.compile(checkpointer=memory)
    
    # Estado inicial enriquecido
    initial_state = {
        "messages": [],
        "client": {
            "id": "client_001",
            "name": "María González",
            "email": "maria@email.com"
        },
        "current_property_id": None,
        "current_property_data": None,
        "user_confirmed_interest": False,
        "confirmation_timestamp": None,
        "projects": [],
        "active_project_id": None,
        "visits": [],
        "pending_visit_request": None,
        "conversation_context": {},
        "last_action": None,
        "escalation_requested": False,
        "interaction_count": 0,
        "session_start_time": datetime.now().isoformat()
    }
    
    print("🏠 Real Estate Swarm con Command Tools y ToolMessages")
    print("=" * 60)
    print("✨ Los tools modifican el estado automáticamente")
    print("🔧 ToolMessages técnicos NO aparecen como respuestas finales")
    print("🤖 Los agentes generan respuestas naturales basadas en el estado")
    print("=" * 60)
    print("\nEjemplos:")
    print("1. 'https://example.com/property/luxury-apartment-zona-rosa'")
    print("2. 'Me interesa esta propiedad'")
    print("3. 'Quiero agendar una visita'")
    print("4. 'No me gusta, busco otra cosa'")
    print("\nEscribe 'estado' para ver el estado actual")
    print("Escribe 'salir' para terminar")
    print("-" * 60)
    
    config: RunnableConfig = {"configurable": {"thread_id": "command_swarm_001"}}
    
    while True:
        user_input = input("\n👤 Tú: ").strip()
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Gracias por probar nuestro Command Swarm con ToolMessages!")
            break
            
        if user_input.lower() == 'estado':
            # Obtener estado actual
            current_state = app.get_state(config)
            state_values = current_state.values
            
            print("\n📊 ESTADO ACTUAL:")
            print(f"🏠 Propiedad actual: {state_values.get('current_property_id', 'Ninguna')}")
            print(f"💚 Interés confirmado: {state_values.get('user_confirmed_interest', False)}")
            print(f"📅 Visita pendiente: {'Sí' if state_values.get('pending_visit_request') else 'No'}")
            print(f"📋 Proyecto activo: {state_values.get('active_project_id', 'Ninguno')}")
            print(f"🎯 Última acción: {state_values.get('last_action', 'Ninguna')}")
            print(f"🆘 Escalamiento: {'Sí' if state_values.get('escalation_requested') else 'No'}")
            print(f"💬 Interacciones: {state_values.get('interaction_count', 0)}")
            
            # Mostrar últimos mensajes para verificar ToolMessages
            messages = state_values.get('messages', [])
            if messages:
                print(f"\n📝 Últimos 3 mensajes:")
                for msg in messages[-3:]:
                    msg_type = type(msg).__name__
                    content = getattr(msg, 'content', '')[:50]
                    print(f"   {msg_type}: {content}...")
            continue
        
        if not user_input:
            continue
        
        try:
            # Procesar con Command Swarm
            response = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # Mostrar respuesta (filtrar ToolMessages técnicos)
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                # Solo mostrar si NO es un ToolMessage técnico
                if not (isinstance(last_message, ToolMessage) and 
                       last_message.content in ["PROPERTY_LOADED", "INTEREST_CONFIRMED", 
                                              "PROJECT_CREATED", "VISIT_REQUESTED", 
                                              "VISIT_CONFIRMED", "ESCALATED_TO_HUMAN",
                                              "REJECTION_ANALYZED"]):
                    print(f"\n🤖 Agente: {last_message.content}")
            
            # Mostrar cambios de estado relevantes
            if response.get("last_action"):
                print(f"🔄 Acción: {response['last_action']}")
                
            if response.get("user_confirmed_interest"):
                print("✅ ¡Interés confirmado en la propiedad!")
                
            if response.get("pending_visit_request"):
                print("📅 Solicitud de visita registrada")
                
            if response.get("escalation_requested"):
                print("🆘 Escalamiento a agente humano procesado")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

# ===============================
# PUNTO DE ENTRADA
# ===============================

if __name__ == "__main__":
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Configura OPENAI_API_KEY")
        print("\nEste ejemplo muestra Command y Send con ToolMessages correctos")
    else:
        import asyncio
        asyncio.run(run_command_swarm())