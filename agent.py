import os
import json
import re
from typing import TypedDict, Optional, List

from langdetect import detect, DetectorFactory
from duckduckgo_search import DDGS
import google.generativeai as genai

DetectorFactory.seed = 0  # deterministic langdetect results

# ==========================================
# 1. LLM CLIENT (Google Gemini)
# ==========================================
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# If running in Google Colab and the key still isn't set, try pulling it
# straight from Colab's secrets manager (key icon in the left sidebar).
if not GOOGLE_API_KEY:
    try:
        from google.colab import userdata
        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY") or userdata.get("GEMINI_API_KEY")
    except Exception:
        pass

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


def call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Single helper to call the Gemini generate_content endpoint."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Export it as an environment variable "
            "before running the agent."
        )
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature),
    )
    return (response.text or "").strip()


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class HealthAgentState(TypedDict):
    user_query: str
    detected_language: str          # "english" | "hindi" | "telugu"
    lang_code: str                  # "en" | "hi" | "te"
    is_health_related: bool
    needs_search: bool
    search_results: Optional[str]
    final_answer: Optional[str]
    next_step: Optional[str]


# ==========================================
# 3. OFF-TOPIC REFUSAL (fixed, respectful, per language)
# ==========================================
OFF_TOPIC_MESSAGE = {
    "en": (
        "I'm here to help only with health-related information. "
        "Could you please share a question related to health so I can "
        "assist you better?"
    ),
    "hi": (
        "मैं केवल स्वास्थ्य से संबंधित जानकारी में आपकी सहायता करने के लिए "
        "बना हूँ। कृपया स्वास्थ्य से जुड़ा प्रश्न पूछें, ताकि मैं आपकी बेहतर "
        "सहायता कर सकूँ।"
    ),
    "te": (
        "నేను కేవలం ఆరోగ్యానికి సంబంధించిన సమాచారంలో మాత్రమే మీకు సహాయం "
        "చేయడానికి రూపొందించబడ్డాను. దయచేసి ఆరోగ్యానికి సంబంధించిన ప్రశ్న "
        "అడగండి, తద్వారా నేను మీకు మెరుగ్గా సహాయం చేయగలను."
    ),
}

LANG_NAME = {"en": "english", "hi": "hindi", "te": "telugu"}


# ==========================================
# 4. GRAPH NODES
# ==========================================
def language_detection_node(state: HealthAgentState):
    print("\n[1/5] Detecting language...")
    query = state["user_query"]
    try:
        code = detect(query)
    except Exception:
        code = "en"

    # Collapse anything unsupported down to English, and normalise
    # common langdetect variants for Hindi / Telugu.
    if code not in ("en", "hi", "te"):
        code = "en"

    print(f"      -> detected: {LANG_NAME[code]} ({code})")
    return {"lang_code": code, "detected_language": LANG_NAME[code]}


def question_classification_node(state: HealthAgentState):
    print("[2/5] Classifying question (health-related? needs live search?)...")
    system_prompt = (
        "You are a strict JSON-only classifier for a health-information "
        "assistant. Given a user question (which may be in English, Hindi, "
        "or Telugu), decide:\n"
        "1. is_health_related: true if the question is about health, "
        "medicine, symptoms, diseases, wellness, nutrition, mental health, "
        "medications, or healthcare in general. false otherwise.\n"
        "2. needs_search: true only if is_health_related is true AND "
        "answering well requires current/real-time information (e.g. "
        "recent outbreaks, new drug approvals, latest guidelines, current "
        "statistics). false if it can be answered from general medical "
        "knowledge.\n"
        'Respond with ONLY this JSON object and nothing else: '
        '{"is_health_related": true|false, "needs_search": true|false}'
    )
    raw = call_gemini(system_prompt, state["user_query"], temperature=0)
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(cleaned)
        is_health = bool(parsed.get("is_health_related", False))
        needs_search = bool(parsed.get("needs_search", False))
    except Exception:
        # Fail safe: treat unparsable output as health-related, no search
        is_health, needs_search = True, False

    print(f"      -> is_health_related={is_health}, needs_search={needs_search}")
    return {"is_health_related": is_health, "needs_search": needs_search}


def off_topic_node(state: HealthAgentState):
    print("[!] Off-topic question detected. Returning polite refusal.")
    message = OFF_TOPIC_MESSAGE[state["lang_code"]]
    return {"final_answer": message}


def duckduckgo_search_node(state: HealthAgentState):
    print("[3/5] Searching DuckDuckGo for up-to-date information...")
    query = state["user_query"]
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        formatted = "\n".join(
            f"- {h.get('title', '')}: {h.get('body', '')} (source: {h.get('href', '')})"
            for h in hits
        )
        if not formatted:
            formatted = "No relevant search results were found."
    except Exception as e:
        formatted = f"Search failed: {e}"

    return {"search_results": formatted}


def knowledge_base_node(state: HealthAgentState):
    print("[4/5] Answering from the model's own medical knowledge...")
    lang = state["detected_language"]
    system_prompt = (
        f"You are a careful, respectful health-information assistant. "
        f"Answer the user's health question using your own general medical "
        f"knowledge. Respond ONLY in {lang}. Keep the answer clear and "
        f"well organised, and end with a brief reminder to consult a "
        f"qualified doctor for personal medical advice or diagnosis."
    )
    answer = call_gemini(system_prompt, state["user_query"])
    return {"final_answer": answer}


def generate_answer_node(state: HealthAgentState):
    print("[4/5] Generating answer from search results via Gemini...")
    lang = state["detected_language"]
    system_prompt = (
        f"You are a careful, respectful health-information assistant. "
        f"Use the search results below (if relevant) plus your own medical "
        f"knowledge to answer the user's health question. Respond ONLY in "
        f"{lang}. Keep the answer clear and well organised, and end with a "
        f"brief reminder to consult a qualified doctor for personal medical "
        f"advice or diagnosis.\n\nSEARCH RESULTS:\n{state.get('search_results', '')}"
    )
    answer = call_gemini(system_prompt, state["user_query"])
    return {"final_answer": answer}


def return_response_node(state: HealthAgentState):
    print("[5/5] Returning response to user.\n")
    print("=" * 60)
    print(state["final_answer"])
    print("=" * 60)
    return {}


# ==========================================
# 5. GRAPH CONSTRUCTION & ROUTING
# ==========================================
from langgraph.graph import StateGraph, START, END


def build_graph():
    graph = StateGraph(HealthAgentState)

    graph.add_node("language_detection", language_detection_node)
    graph.add_node("question_classification", question_classification_node)
    graph.add_node("off_topic", off_topic_node)
    graph.add_node("duckduckgo_search", duckduckgo_search_node)
    graph.add_node("knowledge_base", knowledge_base_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("return_response", return_response_node)

    graph.add_edge(START, "language_detection")
    graph.add_edge("language_detection", "question_classification")

    def route_after_classification(state: HealthAgentState):
        if not state["is_health_related"]:
            return "off_topic"
        if state["needs_search"]:
            return "duckduckgo_search"
        return "knowledge_base"

    graph.add_conditional_edges(
        "question_classification",
        route_after_classification,
        {
            "off_topic": "off_topic",
            "duckduckgo_search": "duckduckgo_search",
            "knowledge_base": "knowledge_base",
        },
    )

    graph.add_edge("duckduckgo_search", "generate_answer")
    graph.add_edge("generate_answer", "return_response")
    graph.add_edge("knowledge_base", "return_response")
    graph.add_edge("off_topic", "return_response")
    graph.add_edge("return_response", END)

    return graph.compile()


health_app = build_graph()


# ==========================================
# 6. PUBLIC HELPER (used by deployment wrappers)
# ==========================================
def answer_query(user_query: str) -> dict:
    """Run one query through the agent and return the full final state."""
    result = health_app.invoke({"user_query": user_query})
    return result


# ==========================================
# 7. CLI EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("Multilingual Health Information Agent (English / Hindi / Telugu)")
    print("Type 'exit' to quit.\n")
    try:
        while True:
            q = input("You: ").strip()
            if q.lower() == "exit":
                print("Goodbye!")
                break
            if not q:
                continue
            answer_query(q)
    except KeyboardInterrupt:
        print("\nStopped by user.")