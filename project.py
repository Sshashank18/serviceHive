import os
import json
import time
from google import genai
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- Global Config ---
API_KEY = "YOUR API KEY HERE" 
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-flash" 

# Local knowledge base to satisfy RAG requirements
KB_CONTENT = {
    "pricing": {
        "Basic Plan": "$29/month, 10 videos/month, 720p resolution",
        "Pro Plan": "$79/month, Unlimited videos, 4K resolution, AI captions"
    },
    "policies": {
        "Refunds": "No refunds after 7 days.",
        "Support": "24/7 support is available only on the Pro plan."
    }
}

# Dump to file so the agent reads from a "database"
with open("autostream_kb.json", "w") as f:
    json.dump(KB_CONTENT, f)

def mock_lead_capture(name, email, platform):
    """Simulates a backend API call for lead ingestion"""
    print(f"\n" + "!"*50)
    print(f"SUCCESS: Lead captured for {name} ({email}) on {platform}")
    print("!"*50 + "\n")

class AgentState(TypedDict):
    """LangGraph state schema"""
    messages: List[dict]
    intent: Literal["Greeting", "Inquiry", "High-Intent", "Unknown"]

class AutoStreamAgent:
    def __init__(self):
        # Read the local store on init
        with open("autostream_kb.json", "r") as f:
            self.kb = json.load(f)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _call_gemini(self, contents, system_instruction):
        """Wrapper for API calls with backoff logic for 503/429 errors"""
        return client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config={'system_instruction': system_instruction}
        )

    def run_agent(self, state: AgentState):
        """Main node logic: classifies intent and generates context-aware replies"""
        prompt = f"""
        You are an AI for AutoStream. 
        Use the following KB for technical/pricing info: {json.dumps(self.kb)}

        Workflow:
        1. Classify intent: 'Greeting', 'Inquiry', or 'High-Intent'.
        2. Use KB for 'Inquiry'.
        3. If 'High-Intent', you must get: Name, Email, and Creator Platform.
        
        Mandatory Format:
        Intent: [Classification]
        Response: [Your Message]
        
        Trigger lead capture ONLY if all 3 fields are present:
        LEAD_COMPLETE: [Name], [Email], [Platform]
        """
        
        # Map message history for the Gemini SDK
        history = []
        for msg in state['messages']:
            role = "user" if msg['role'] == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg['content']}]})
            
        try:
            response = self._call_gemini(history, prompt)
            raw_text = response.text
            print(raw_text)
            
            # Simple string parsing for intent tracking
            detected_intent = "Unknown"
            for itype in ["Greeting", "Inquiry", "High-Intent"]:
                if f"Intent: {itype}" in raw_text:
                    detected_intent = itype
            
            # Extract just the chat part for the user
            clean_reply = raw_text.split("Response:")[-1].strip()
            return {
                "messages": [{"role": "model", "content": clean_reply}], 
                "intent": detected_intent
            }
        
        except Exception as e:
            return {
                "messages": [{"role": "model", "content": f"System error: {str(e)}"}], 
                "intent": "Unknown"
            }

# Setup the graph
builder = StateGraph(AgentState)
logic = AutoStreamAgent()
builder.add_node("agent", logic.run_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# In-memory checkpointer for session persistence
app = builder.compile(checkpointer=MemorySaver())

def run_chat():
    """CLI loop for testing the agent workflow"""
    # Unique thread ID allows the agent to 'remember' across turns
    config = {"configurable": {"thread_id": "test_session_01"}}
    print("--- AutoStream AI Online (Resilient Mode) ---")
    
    while True:
        u_in = input("\nYou: ")
        if u_in.lower() in ['exit', 'quit']: 
            break
        
        # Invoke graph with user input
        state_update = {"messages": [{"role": "user", "content": u_in}]}
        output = app.invoke(state_update, config)
        
        bot_reply = output["messages"][-1]["content"]
        print(f"[Internal Intent: {output['intent']}]")
        print(f"Agent: {bot_reply}")      
        
        # Hook for external function calls
        if "LEAD_COMPLETE:" in bot_reply:
            parts = bot_reply.split("LEAD_COMPLETE:")[1].strip().split(",")
            if len(parts) >= 3:
                mock_lead_capture(parts[0].strip(), parts[1].strip(), parts[2].strip())
                break

if __name__ == "__main__":
    run_chat()
