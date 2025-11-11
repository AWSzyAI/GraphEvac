# src/configs.py
def load_config():
    layout = {
        "rooms": [
            {"id": "R1", "pos": 10, "entry": 10, "occupants": [{"id": "o1", "status": "inside"}]},
            {"id": "R2", "pos": 20, "entry": 20, "occupants": [{"id": "o2", "status": "inside"}]}
        ]
    }
    responders = [
        {"id": "A", "pos": 0, "view": 15, "alpha": 0.2, "H_limit": 0.7, "detection_prob": 0.8},
        {"id": "B", "pos": 30, "view": 15, "alpha": 0.2, "H_limit": 0.7, "detection_prob": 0.8}
    ]
    occupants = [o for r in layout["rooms"] for o in r["occupants"]]
    hazard = {r["id"]: 0.1 for r in layout["rooms"]}
    return {
        "layout": layout,
        "responders": responders,
        "occupants": occupants,
        "hazard": hazard,
        "dt": 1.0
    }
