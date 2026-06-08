import requests

URL = "http://127.0.0.1:8000/chat"

general_msgs = [
    "Tell me about Acme Technologies",
    "Summarize Acme's core product categories",
    "What products does Acme Technologies offer?",
    "Explain the features of AuroraLamp",
    "What are the specifications of EcoTherm X2?",
    "Describe SecureCam Pro and its capabilities",
    "What features does SmartLock Ultra provide?",
    "How do I set up an AuroraLamp?",
    "What should I do if my device is offline?",
    "How can I fix Bluetooth pairing issues?",
    "What are the recommended steps for a firmware update failure?",
    "Explain the Wi-Fi performance recommendations",
    "What are the account security best practices?",
    "How do I reset an AuroraLamp?",
    "What warranty does AuroraLamp include?",
    "How can I contact Acme technical support?",
    "What is the emergency hotline number for Acme?",
    "Explain the support escalation priorities",
    "What qualifies as a Priority P1 incident?",
    "Summarize Acme customer support options"
]

stress_msgs = [
    "Provide a one-line summary of Acme Technologies",
    "Summarize AuroraLamp in under 20 words",
    "Briefly describe EcoTherm X2",
    "Return a compact SecureCam Pro overview",
    "Give a fast factual reply about SmartLock Ultra",
    "Provide a short answer about Acme support channels",
    "Summarize AuroraLamp setup in one sentence",
    "Compress device offline troubleshooting into a single sentence",
    "Briefly explain Bluetooth pairing troubleshooting",
    "State AuroraLamp warranty information briefly",
    "Low latency check for technical support contact information",
    "Quick response requested for Acme emergency hotline",
    "Short factual note about firmware update failures",
    "Terse summary of Wi-Fi optimization recommendations",
    "One-line answer about account security best practices",
    "Short answer on EcoTherm X2 features",
    "Minimal wording for SecureCam Pro capabilities",
    "Compact explanation of SmartLock Ultra access methods",
    "Brief note on Acme business hours",
    "Quick check of support escalation procedures",
    "Respond quickly with a summary of Priority P1 incidents",
    "Keep the reply short about Priority P2 incidents",
    "Concise response about Priority P3 incidents",
    "Fast answer about Acme product support contacts",
    "Short answer about AuroraLamp support information",
    "Single-sentence response about EcoTherm X2 support",
    "Brief note on SecureCam Pro support options",
    "Compact answer on SmartLock Ultra support channels",
    "Low-latency test for customer service information",
    "Short explanation of support escalation response times",
    "Terse response about Acme smart home products",
    "One-line answer on AuroraLamp features",
    "Concise info on EcoTherm X2 energy-saving capabilities",
    "Quick summary about SecureCam Pro surveillance features",
    "Short factual response about SmartLock Ultra security features",
    "Compact note on account security recommendations",
    "Brief response about Wi-Fi performance guidance",
    "Minimal answer on device troubleshooting steps",
    "Fast one-line note about Acme corporate support",
    "Throughput test with a short Acme product reply",
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