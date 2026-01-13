# 🚀 AutoStream AI Agent

An intelligent, stateful lead generation and customer support agent powered by **Gemini 2.5 Flash** and **LangGraph**. This agent handles user inquiries using a local Knowledge Base (RAG) and automatically captures high-intent leads through multi-turn conversation.

---

### 🛠 How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sshashank18/serviceHive
    cd serviceHive
    ```

2.  **Set Up Environment:**
    Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    Replace the `API_KEY` variable in the script with your Google AI Studio key or set it as an environment variable.

5.  **Start the Agent:**
    ```bash
    python project.py
    ```

---

### 🏗 Architecture Explanation

The AutoStream Agent is built using **LangGraph** rather than a simpler sequential chain or AutoGen because of its ability to manage **cyclical state machines** and robust persistence.



* **Why LangGraph?** While AutoGen is excellent for multi-agent social dynamics, LangGraph provides a lower-level, control-oriented approach. This project requires a strict business workflow: **Classify Intent → Reference Knowledge Base → Audit Lead Progress**. LangGraph’s directed graph structure ensures the agent follows this logic reliably, preventing the "hallucinated departures" common in simpler chatbot architectures.
* **State Management:** The agent utilizes a `TypedDict` called `AgentState`. This central schema carries the message history and the detected `intent` throughout the graph's lifecycle. We use **MemorySaver** (an in-memory checkpointer) to provide session persistence. By using a unique `thread_id`, the agent can remember past interactions with a specific user even if the execution loop is interrupted, making the experience feel seamless and professional.

---

### 📱 WhatsApp Deployment (via Webhooks)

To move this agent from the command line to a production environment like WhatsApp, we would implement a **Webhook architecture** using a provider like **Twilio**.



1.  **Incoming Message:** When a user sends a WhatsApp message, Twilio receives it and sends an **HTTP POST request** (the Webhook) to our hosted server (e.g., via FastAPI or Flask).
2.  **Session Mapping:** The server extracts the user's phone number and uses it as the `thread_id`. This allows the LangGraph engine to automatically retrieve that specific user's conversation history from the checkpointer.
3.  **Agent Logic:** The message content is passed into the graph. The agent determines if it needs to provide pricing info or if it's time to trigger the `LEAD_COMPLETE` logic.
4.  **Outgoing Response:** Our server receives the agent's output and sends it back to Twilio as a response. Twilio then delivers the message to the user’s WhatsApp chat.

---