from dataclasses import dataclass

@dataclass
class Route:
    use_local: bool
    use_web: bool
    reason: str

SCET_HINTS = {
    "scet", "sarvajanik", "surat", "vivaksha", "jariwala",
    "it department", "computer engineering", "placements", "hod", "faculty"
}

def pick_route(query: str, has_local: bool) -> Route:
    q = (query or "").lower()

    if any(k in q for k in SCET_HINTS):
        if has_local:
            return Route(use_local=True, use_web=True, reason="SCET query → hybrid")
        else:
            return Route(use_local=False, use_web=True, reason="SCET query → web only")

    return Route(use_local=has_local, use_web=True, reason="Default hybrid")
