import os
import json
import time
from google import genai
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- Global Config ---
API_KEY = "Your API key" 
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash" 

# Local knowledge base
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

with open("autostream_kb.json", "w") as f:
    json.dump(KB_CONTENT, f)

def mock_lead_capture(name, email, platform):
    print(f"\n" + "!"*50)
    print(f"SUCCESS: Lead captured for {name} ({email}) on {platform}")
    print("!"*50 + "\n")

class AgentState(TypedDict):
    # add_messages handles the conversion of dicts to Message Objects
    messages: Annotated[List[dict], add_messages]
    intent: Literal["Greeting", "Inquiry", "High-Intent", "Unknown"]

class AutoStreamAgent:
    def __init__(self):
        with open("autostream_kb.json", "r") as f:
            self.kb = json.load(f)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _call_gemini(self, contents, system_instruction):
        return client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config={'system_instruction': system_instruction}
        )

    def run_agent(self, state: AgentState):
        prompt = f"""
        You are an AI for AutoStream. 
        KB Info: {json.dumps(self.kb)}

        Workflow:
        1. Classify intent: 'Greeting', 'Inquiry', or 'High-Intent'.
        2. Use KB for 'Inquiry'.
        3. For 'High-Intent', check the history. If Name, Email, or Platform is missing, ask for it.
        
        Mandatory Format:
        Intent: [Classification]
        Response: [Your Message]
        
        Only if ALL 3 (Name, Email, Platform) have been provided in the history:
        LEAD_COMPLETE: [Name], [Email], [Platform]
        """
        
        # 1. Properly map LangGraph Objects to Gemini Format
        history = []
        for msg in state['messages']:
            # Handle objects (HumanMessage/AIMessage) or dicts
            m_type = getattr(msg, 'type', None) or msg.get('role')
            m_content = getattr(msg, 'content', None) or msg.get('content')
            
            # Gemini expects "user" and "model"
            role = "user" if m_type in ["human", "user"] else "model"
            history.append({"role": role, "parts": [{"text": m_content}]})
            
        try:
            response = self._call_gemini(history, prompt)
            raw_text = response.text
            
            detected_intent = "Unknown"
            for itype in ["Greeting", "Inquiry", "High-Intent"]:
                if f"Intent: {itype}" in raw_text:
                    detected_intent = itype
            
            clean_reply = raw_text.split("Response:")[-1].strip()
            
            # 2. Return with "ai" role (LangGraph standard) instead of "model"
            return {
                "messages": [{"role": "ai", "content": clean_reply}], 
                "intent": detected_intent
            }
        
        except Exception as e:
            return {
                "messages": [{"role": "ai", "content": f"System error: {str(e)}"}], 
                "intent": "Unknown"
            }

# Setup the graph
builder = StateGraph(AgentState)
logic = AutoStreamAgent()
builder.add_node("agent", logic.run_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

app = builder.compile(checkpointer=MemorySaver())

def run_chat():
    config = {"configurable": {"thread_id": "session_final_fix"}}
    print("--- AutoStream AI Online  ---")
    
    while True:
        u_in = input("\nYou: ")
        if u_in.lower() in ['exit', 'quit']: 
            break
        
        # Use "user" role here, which LangGraph accepts
        output = app.invoke({"messages": [{"role": "user", "content": u_in}]}, config)
        
        bot_reply = output["messages"][-1].content
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
