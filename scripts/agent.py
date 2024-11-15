from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from recommendation import HybridRecommender

@dataclass
class AgentState:
    """State management for the AI agent"""
    messages: List[BaseMessage]
    user_profile: Dict
    current_recommendations: List[Dict]
    feedback: Optional[Dict] = None
    context: Dict = None

class SchemeRecommenderAgent:
    def __init__(self, recommender: HybridRecommender, openai_api_key: str):
        """Initialize the AI agent with recommender system and tools"""
        self.recommender = recommender
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        # Initialize the graph
        self.graph = self.create_workflow_graph()
        
        # Initialize tools
        self.tools = self.create_tools()
        self.tool_executor = ToolExecutor(self.tools)

    def create_tools(self) -> Dict:
        """Create specialized tools for the agent"""
        return {
            "get_recommendations": {
                "name": "get_recommendations",
                "description": "Get scheme recommendations based on user profile and query",
                "func": self.recommender.get_recommendations,
            },
            "analyze_user_needs": {
                "name": "analyze_user_needs",
                "description": "Analyze user needs and context",
                "func": self.analyze_user_needs,
            },
            "refine_recommendations": {
                "name": "refine_recommendations",
                "description": "Refine recommendations based on user feedback",
                "func": self.refine_recommendations,
            }
        }

    def analyze_user_needs(self, messages: List[BaseMessage]) -> Dict:
        """Analyze user needs from conversation history"""
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the user's needs and context from their messages."),
            ("human", "{messages}")
        ])
        
        response = self.llm.invoke(
            analysis_prompt.format_messages(
                messages=str([m.content for m in messages if isinstance(m, HumanMessage)])
            )
        )
        
        return JsonOutputParser().parse(response.content)

    def refine_recommendations(
        self,
        recommendations: List[Dict],
        feedback: Dict,
        context: Dict
    ) -> List[Dict]:
        """Refine recommendations based on user feedback and context"""
        # Implement recommendation refinement logic
        if feedback.get("relevance_score", 0) < 0.7:
            # Adjust weights and get new recommendations
            new_weights = {
                "vector_weight": context.get("vector_weight", 0.5) + 0.1,
                "semantic_weight": context.get("semantic_weight", 0.5) - 0.1
            }
            return self.recommender.get_hybrid_recommendations(
                user_profile=context["user_profile"],
                query=context["query"],
                vector_weight=new_weights["vector_weight"]
            )
        return recommendations

    def should_continue(self, state: AgentState) -> bool:
        """Determine if the conversation should continue"""
        if not state.messages:
            return False
            
        last_message = state.messages[-1]
        if isinstance(last_message, HumanMessage):
            return True
            
        # Check for conversation ending indicators
        end_indicators = ["thank you", "goodbye", "that's all"]
        return not any(indicator in last_message.content.lower() for indicator in end_indicators)

    def process_user_input(self, state: AgentState) -> AgentState:
        """Process user input and update state"""
        # Analyze user needs
        context = self.tool_executor.execute(
            "analyze_user_needs",
            messages=state.messages
        )
        
        # Get recommendations
        recommendations = self.tool_executor.execute(
            "get_recommendations",
            user_profile=state.user_profile,
            query=context.get("query", "")
        )
        
        # Update state
        state.context = context
        state.current_recommendations = recommendations
        
        return state

    def generate_response(self, state: AgentState) -> Tuple[AgentState, str]:
        """Generate AI response based on current state"""
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful government scheme recommendation assistant.
            Provide clear and concise explanations of recommended schemes.
            Format the response in a user-friendly way."""),
            ("human", """User Profile: {user_profile}
            Context: {context}
            Recommendations: {recommendations}
            Generate a helpful response.""")
        ])
        
        response = self.llm.invoke(
            response_prompt.format_messages(
                user_profile=state.user_profile,
                context=state.context,
                recommendations=state.current_recommendations
            )
        )
        
        state.messages.append(AIMessage(content=response.content))
        return state, response.content

    def create_workflow_graph(self) -> StateGraph:
        """Create the workflow graph using LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", self.process_user_input)
        workflow.add_node("generate_response", self.generate_response)
        
        # Add edges
        workflow.add_edge("process_input", "generate_response")
        workflow.add_conditional_edges(
            "generate_response",
            self.should_continue,
            {
                True: "process_input",
                False: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("process_input")
        
        return workflow

    def run(self, user_profile: Dict, initial_message: str) -> List[str]:
        """Run the agent with initial user input"""
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content=initial_message)],
            user_profile=user_profile,
            current_recommendations=[],
            context={}
        )
        
        # Execute workflow
        final_state = self.graph.run(state)
        
        # Return conversation history
        return [msg.content for msg in final_state.messages] 