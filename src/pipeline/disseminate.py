"""
Farmer dissemination simulator (feature F27): SMS, WhatsApp and IVR.

**Nothing is sent.** For every registered (mock) subscriber of an *officer-published* issue, the
message that would be sent is generated in the subscriber's language and written to the
``outbox`` table with its character and segment count. In deployment, replace
``simulate``'s queue call with a gateway client (e.g. the NIC SMS gateway, or the WhatsApp
Business API via a DLT-registered template).

* SMS: the bulletin's prioritised SMS text (<=160 characters).
* WhatsApp: a richer text (headline + weather table + top advisories).
* IVR: a read-aloud script with numbers spelt out as plain units.
"""

from __future__ import annotations

from collections.abc import Callable

from src.common.config import Config
from src.dashboard.db import Store


def whatsapp_text(b: dict) -> str:
    L = b["labels"]
    lines = [f"*{L['bulletin_title']}* - {b['gp_name']} ({b['block_name']})",
             f"{L['overall_status']}: *{b['overall_label']}*", ""]
    for r in b["table"]:
        lines.append(f"{r['label']}: {r['rain']} mm, {r['tmax']}/{r['tmin']} °C")
    lines.append("")
    for a in [*b["general"], *b["crop_advisories"], *b["disease_alerts"]][:4]:
        if a["severity"] != "green" or a["rule"] in ("normal", "spray_window"):
            lines.append(f"• *{a['title']}* {a['action']}")
    return "\n".join(lines)[:1500]


def ivr_script(b: dict) -> str:
    L = b["labels"]
    parts = [f"{L['bulletin_title']}. {L['gp']} {b['gp_name']}.", f"{L['overall_status']}: {b['overall_label']}.",
             b["summary"]]
    for a in [*b["general"], *b["crop_advisories"], *b["disease_alerts"]][:5]:
        parts.append(f"{a['title']}. {a['text']} {a['action']}".strip())
    return "\n".join(p.replace("°C", " degree").replace("mm", " millimetre") for p in parts)


def simulate(cfg: Config, store: Store, issue: str, get_bulletin: Callable[[str, str], dict]) -> dict:
    subs = store.subscribers()
    counts = {"sms": 0, "whatsapp": 0, "ivr": 0}
    cache: dict = {}
    for s in subs:
        key = (s["gp_code"], s["lang"])
        if key not in cache:
            cache[key] = get_bulletin(*key)
        b = cache[key]
        msg = {"sms": b["sms"], "whatsapp": whatsapp_text(b), "ivr": ivr_script(b)}[s["channel"]]
        store.queue(issue, s, msg)
        counts[s["channel"]] += 1
    store.log("system", "disseminate(simulated)", {"issue_date": issue, **counts})
    return {"issue_date": issue, "subscribers": len(subs), "queued": counts,
            "note": "simulation only - no messages were sent"}
