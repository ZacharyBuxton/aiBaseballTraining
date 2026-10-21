"""
coaching.py -- turn SwingMetrics into a few plain-language notes.

Right now this is rule-based (works offline, deterministic, no API key). The team plans to add
LLM-generated coaching via OpenRouter later; to do that, implement another class with the same
`notes(metrics) -> list[str]` method and select it in dckft.pipeline. Keep the rule-based coach
as the fallback when the network/API is unavailable.

Reference ranges are typical published values for high-school/college hitters (bat-sensor
literature and vendor benchmarks). Tune them with the coaches the team interviewed.
"""

from __future__ import annotations

import math
from typing import List, Protocol

from .metrics import SwingMetrics


class Coach(Protocol):
    def notes(self, m: SwingMetrics) -> List[str]: ...


RANGES = {
    "bat_speed_mph": (65.0, 80.0),
    "attack_angle_deg": (5.0, 20.0),
    "vertical_bat_angle_deg": (-45.0, -15.0),
    "time_to_contact_s": (0.13, 0.20),
}


class RuleBasedCoach:
    max_notes = 3

    def notes(self, m: SwingMetrics) -> List[str]:
        out: List[str] = []

        def bad(x):
            return x is None or (isinstance(x, float) and math.isnan(x))

        lo, hi = RANGES["attack_angle_deg"]
        if not bad(m.attack_angle_deg):
            if m.attack_angle_deg < lo - 5:
                out.append(f"Attack angle {m.attack_angle_deg:+.0f}° is steep downward. Let the barrel "
                           "work up through the zone: match the pitch plane instead of chopping.")
            elif m.attack_angle_deg < lo:
                out.append(f"Attack angle {m.attack_angle_deg:+.0f}° is a bit flat. A slightly more "
                           "upward path (5–20°) keeps the barrel in the zone longer.")
            elif m.attack_angle_deg > hi + 5:
                out.append(f"Attack angle {m.attack_angle_deg:+.0f}° is very uphill, so expect pop-ups. "
                           "Stay through the ball rather than lifting early.")

        lo, hi = RANGES["time_to_contact_s"]
        if not bad(m.time_to_contact_s) and m.time_to_contact_s > hi:
            out.append(f"Time to contact {m.time_to_contact_s:.2f}s is long. Shorten the move "
                       "to the ball (hands inside, less barrel drag) to buy decision time.")

        lo, hi = RANGES["bat_speed_mph"]
        if not bad(m.bat_speed_mph) and m.bat_speed_mph < lo:
            if not bad(m.peak_bat_speed_mph) and m.peak_bat_speed_mph - m.bat_speed_mph > 5:
                out.append(f"Bat speed peaked at {m.peak_bat_speed_mph:.0f} mph but was "
                           f"{m.bat_speed_mph:.0f} mph at contact. You're releasing early, so time "
                           "the hip/hand sequence so peak speed lands at the ball.")
            else:
                out.append(f"Bat speed {m.bat_speed_mph:.0f} mph. Build speed from the ground up: "
                           "load the back hip, then let the hips lead the hands.")

        lo, hi = RANGES["vertical_bat_angle_deg"]
        if not bad(m.vertical_bat_angle_deg) and m.vertical_bat_angle_deg > hi + 10:
            out.append(f"Bat is nearly flat at contact ({m.vertical_bat_angle_deg:+.0f}°). That's fine on high "
                       "pitches; on lower pitches tilt the shoulders to get the barrel under.")

        if not out:
            out.append(f"Solid swing: {m.bat_speed_mph:.0f} mph bat speed, "
                       f"{m.attack_angle_deg:+.0f}° attack angle, {m.time_to_contact_s:.2f}s to contact. "
                       "Keep repeating it.")
        return out[: self.max_notes]
