from dataclasses import dataclass

@dataclass
class Plan:
    route: str
    reason: str

def make_plan(query: str, has_local: bool, use_web: bool, use_local: bool) -> Plan:
    if use_local and use_web:
        return Plan(route="hybrid", reason="Use local + web for full coverage.")
    if use_local:
        return Plan(route="local", reason="Use local SCET docs.")
    if use_web:
        return Plan(route="web", reason="Use live web info.")
    return Plan(route="llm", reason="Use LLM directly.")
