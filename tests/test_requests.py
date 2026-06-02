import requests

URL = "http://127.0.0.1:8000/chat"

general_msgs = [
    "Tell me about emergency services",
    "Summarize hospital contact options",
    "How do I reach local fire services?",
    "What should I know about ambulance response?",
    "Explain when to call emergency services",
    "Give me a short overview of public safety support",
    "What are the main emergency numbers I should know?",
    "Describe first response services in simple terms",
    "How do police, fire, and medical dispatch differ?",
    "What is the role of emergency operators?",
    "Explain how urgent care differs from the ER",
    "What information should I give during an emergency call?",
    "How can I prepare for a medical emergency?",
    "What are the steps after calling emergency services?",
    "Give a brief summary of disaster response services",
    "When should I avoid calling emergency services?",
    "What does public emergency infrastructure cover?",
    "How do I stay calm during an emergency call?",
    "What should I do while waiting for responders?",
    "Explain emergency services in one paragraph",
]

stress_msgs = [
    "Stress test 01: provide a terse one-line response about emergency services",
    "Stress test 02: summarize emergency dispatch in under 20 words",
    "Stress test 03: answer briefly about local ambulance services",
    "Stress test 04: return a compact emergency services overview",
    "Stress test 05: give a fast factual reply about fire and rescue",
    "Stress test 06: respond with a short safety-focused explanation",
    "Stress test 07: provide a minimal answer about first responders",
    "Stress test 08: compress the topic into a single sentence",
    "Stress test 09: keep the answer concise about emergency hotlines",
    "Stress test 10: state the role of emergency services briefly",
    "Stress test 11: low latency check for chat node behavior",
    "Stress test 12: quick response requested for emergency support",
    "Stress test 13: short factual note about calling for help",
    "Stress test 14: terse summary of medical emergency response",
    "Stress test 15: one-line answer about public safety response",
    "Stress test 16: short answer on ambulance dispatch handling",
    "Stress test 17: minimal wording for emergency operator duties",
    "Stress test 18: compact explanation of urgent incident response",
    "Stress test 19: brief note on what emergency services provide",
    "Stress test 20: quick check of node throughput with a short answer",
    "Stress test 21: respond quickly with a safety answer",
    "Stress test 22: keep the reply short about fire departments",
    "Stress test 23: concise response for police and medical response",
    "Stress test 24: fast answer about crisis dispatch",
    "Stress test 25: short answer about when to call 911",
    "Stress test 26: single-sentence response about responder arrival",
    "Stress test 27: brief note on emergency call handling",
    "Stress test 28: compact answer on public emergency coordination",
    "Stress test 29: low-latency test for the chat endpoint",
    "Stress test 30: short explanation of emergency triage",
    "Stress test 31: terse response about first response teams",
    "Stress test 32: one-line answer on emergency readiness",
    "Stress test 33: concise info on ambulance and fire response",
    "Stress test 34: quick summary about urgent incident support",
    "Stress test 35: short factual response about emergency contact",
    "Stress test 36: compact note on disaster response teams",
    "Stress test 37: brief response about public safety dispatch",
    "Stress test 38: minimal answer on emergency coordination",
    "Stress test 39: fast one-line note about help lines",
    "Stress test 40: throughput test with a short emergency reply",
]


def post_req(conversation_id: str, message: str) -> int:
    try:
        response = requests.post(
            URL,
            json={
                "message": message,
                "conversation_id": conversation_id,
            },
            timeout=30,
        )
        return response.status_code
    except Exception as e:
        print(f"{conversation_id} ERROR: {e}")
        return 0


def run_batch(messages, prefix):
    for i, msg in enumerate(messages, start=1):
        cid = f"{prefix}-{i}"
        code = post_req(cid, msg)
        print(f"{cid} {code}")


if __name__ == "__main__":
    run_batch(general_msgs, "general")
    run_batch(stress_msgs, "stress")