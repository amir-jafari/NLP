import boto3
from langgraph import Graph, PythonNode, LLMNode

# Initialize AWS Bedrock client (ensure AWS creds & region are set)
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
# Configure LLM (e.g., Amazon Titan model on Bedrock)
llm_config = {"model": "amazon.titan-text-large", "client": bedrock}

# Define specialized agent functions:
def events_database_tool(state):
    city = state["city"]
    events = query_local_events(city)
    if not events:
        events = search_online_events(city)  # use Tavily API or similar
    return {"events_info": events}

def get_weather_info(state):
    city = state["city"]
    weather = fetch_weather(city)  # call OpenWeatherMap API
    return {"weather_info": weather}

def get_restaurant_recs(state):
    city = state["city"]
    recs = generate_restaurant_recs(city)  # RAG: query vector DB + LLM
    return {"restaurant_recs": recs}

# Create LangGraph nodes for each agent:
events_node   = PythonNode(name="EventsAgent", func=events_database_tool)
weather_node  = PythonNode(name="WeatherAgent", func=get_weather_info)
rest_node     = PythonNode(name="RestaurantsAgent", func=get_restaurant_recs)
analysis_node = LLMNode(name="AnalysisAgent",
                        prompt="Provide a detailed city report using {events_info}, "
                               "{weather_info}, {restaurant_recs}.",
                        llm=llm_config)

# Build and connect the graph:
graph = Graph()
graph.add_node(events_node, inputs=["city"])
graph.add_node(weather_node, inputs=["city"])
graph.add_node(rest_node, inputs=["city"])
graph.add_node(analysis_node, 
    inputs=["city", "events_info", "weather_info", "restaurant_recs"])
# Connect outputs to the Analysis agent for final answer
graph.connect(events_node, analysis_node)
graph.connect(weather_node, analysis_node)
graph.connect(rest_node, analysis_node)

# Run the multi-agent system:
result = graph.run({"city": "New York"})
print(result["AnalysisAgent"])