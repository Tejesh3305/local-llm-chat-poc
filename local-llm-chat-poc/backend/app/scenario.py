"""Scenario models: plain calculations over data/scenario.json and data/rules.json.

No model is involved here. Every figure a workflow shows comes from these
functions, so the results hold up whether or not an LLM is running.
"""

import math
from typing import Any

Scenario = dict[str, Any]
Rules = dict[str, Any]

RISK_ORDER = ["High", "Considerable", "Moderate", "Low"]


# ---------- formatting helpers ----------


def rnd(value: float) -> int:
    """Round half up, so displayed figures do not depend on banker's rounding."""
    return math.floor(value + 0.5)


def to_min(time: str) -> int:
    hours, _, minutes = str(time).partition(":")
    return int(hours or 0) * 60 + int(minutes or 0)


def clock(minutes: float) -> str:
    """Minutes after midnight as '0530 h'."""
    total = rnd(minutes) % 1440
    return f"{total // 60:02d}{total % 60:02d} h"


def span(hours: float) -> str:
    return f"{rnd(hours * 60)} min" if hours < 1.5 else f"{hours:.1f} h"


def deg(celsius: float) -> str:
    return f"{'−' if celsius < -0.5 else ''}{abs(rnd(celsius))} °C"


def fmt(number: float) -> str:
    return f"{rnd(number):,}"


def and_list(items: list[str]) -> str:
    return "".join(items) if len(items) < 2 else f"{', '.join(items[:-1])} and {items[-1]}"


def risk_tone(risk: str) -> str:
    return "bad" if risk == "High" else "good" if risk == "Low" else "warn"


def eff_tone(percent: float, rules: Rules) -> str:
    if percent >= 95:
        return "good"
    if percent >= 85:
        return ""
    return "warn" if percent >= rules["force"]["reinforce_below_pct"] else "bad"


def flies(slot: dict, rules: Rules) -> bool:
    limits = rules["aviation"]
    return slot["vis_m"] >= limits["min_vis_m"] and slot["wind_kmh"] <= limits["max_wind_kmh"]


# ---------- models ----------


def weather_model(s: Scenario, rules: Rules) -> dict:
    weather = s["weather"]
    temp, wind, snowfall = weather["temp_c"], weather["wind_kmh"], weather["snowfall_cm"]
    # Wind chill index, Environment Canada / US NWS formula (deg C and km/h)
    if wind >= 5 and temp <= 10:
        wind_chill = 13.12 + 0.6215 * temp - 11.37 * wind**0.16 + 0.3965 * temp * wind**0.16
    else:
        wind_chill = temp
    cold = next(band for band in rules["cold"] if band["above"] is None or wind_chill > band["above"])

    sectors = []
    for sector in s["sectors"]:
        new_snow = rnd(snowfall * sector["snow_factor"])
        slope = sector["slope_deg"]
        if 30 <= slope <= 45 and new_snow >= 60:
            risk = "High"
        elif 25 <= slope <= 50 and new_snow >= 30:
            risk = "Considerable"
        else:
            risk = "Moderate" if new_snow >= 30 else "Low"
        sectors.append({**sector, "new_snow_cm": new_snow, "risk": risk, "tone": risk_tone(risk)})
    worst = min(sectors, key=lambda sector: RISK_ORDER.index(sector["risk"]))

    # Flying window: the longest unbroken run of flyable hours
    best = run = None
    for slot in weather["hourly"]:
        run = (run[0] if run else slot["hour"], slot["hour"] + 1) if flies(slot, rules) else None
        if run and (best is None or run[1] - run[0] > best[1] - best[0]):
            best = run
    window = None
    if best:
        window = {"from": best[0] * 60, "to": best[1] * 60, "text": f"{clock(best[0] * 60)[:4]} to {clock(best[1] * 60)}"}
    return {"wind_chill": wind_chill, "cold": cold, "sectors": sectors, "worst": worst, "window": window}


def mobility_model(s: Scenario, rules: Rules) -> dict:
    weather, move = weather_model(s, rules), s["movement"]
    factor = next(band["factor"] for band in rules["snow_speed"] if band["up_to_cm"] is None or s["weather"]["snowfall_cm"] <= band["up_to_cm"])
    speed = move["base_speed_kmh"] * factor
    risk_of = {sector["name"]: sector["risk"] for sector in weather["sectors"]}
    routes = [
        {
            **route,
            "open": route["status"] == "open",
            "transit_h": route["km"] / speed,
            "clearing_h": route["chokepoints"] * move["clearing_min_per_chokepoint"] / 60,
            "avalanche": risk_of.get(route["sector"], "Low"),
        }
        for route in s["routes"]
    ]
    # Open routes, those under a high-risk slope last, then by time
    usable = sorted((route for route in routes if route["open"]), key=lambda route: (route["avalanche"] == "High", route["transit_h"]))
    chosen = usable[0] if usable else None
    fastest = min(routes, key=lambda route: route["transit_h"])

    carriers = math.ceil(move["troops"] / rules["seats_per_vehicle"])
    snow_vehicles = chosen["chokepoints"] + 1 if chosen else 0
    fleet = [{**vehicle, "cleared": bool(chosen) and vehicle["weight_t"] <= chosen["bridge_class_t"]} for vehicle in s["vehicles"]]
    depart = to_min(move["departure"])
    timings = None
    if chosen:
        timings = {
            "clearing_start": depart - 60 - chosen["clearing_h"] * 60,
            "cleared_by": depart - 60,
            "probe": depart - 45,
            "depart": depart,
            "arrive": depart + chosen["transit_h"] * 60,
        }
    per_trip_l = (carriers + snow_vehicles) * chosen["km"] * 2 * move["litres_per_km"] if chosen else 0
    round_trips = math.floor(move["fuel_stock_l"] / per_trip_l) if per_trip_l else 0
    return {
        "weather": weather, "speed": speed, "routes": routes, "chosen": chosen, "fastest": fastest,
        "carriers": carriers, "snow_vehicles": snow_vehicles, "fleet": fleet, "timings": timings,
        "per_trip_l": per_trip_l, "round_trips": round_trips,
    }


def force_model(s: Scenario, rules: Rules) -> dict:
    forward = [unit for unit in s["own"] if unit["role"] == "forward"]
    reserve = next(unit for unit in s["own"] if unit["role"] == "reserve")
    own_fit = sum(unit["fit"] for unit in forward)
    ratio = s["enemy"]["strength"] / max(1, own_fit)
    return {
        "forward": forward, "reserve": reserve, "own_fit": own_fit, "ratio": ratio,
        "deficit": ratio > rules["force"]["reinforce_above_ratio"],
        "ratio_with_reserve": s["enemy"]["strength"] / max(1, own_fit + reserve["fit"]),
        "reaction_min": reserve["km_to_axes"] / mobility_model(s, rules)["speed"] * 60,
        "main": next((unit for unit in forward if unit["at"] == s["enemy"]["at"]), forward[0]),
    }


def casualty_model(s: Scenario, rules: Rules) -> dict:
    weather, c, med = weather_model(s, rules), s["casualties"], s["medical"]
    helo = med["helo"]
    lost = c["cat_a"] + c["cat_b"] + c["cat_c"]
    units = []
    for unit in s["own"]:
        fit_after = max(0, unit["fit"] - (lost if unit["name"] == c["unit"] else 0))
        units.append({**unit, "fit_after": fit_after, "before": 100 * unit["fit"] / unit["establishment"], "after": 100 * fit_after / unit["establishment"]})
    unit = next((entry for entry in units if entry["name"] == c["unit"]), units[0])

    # Cat A go by helicopter when every lift can reach hospital inside the flying window and that beats the road
    at, window = to_min(c["time"]), weather["window"]
    road_arrives = at + med["road_hours"] * 60
    sorties = math.ceil(c["cat_a"] / helo["stretchers"])
    lift_at = first_at_hospital = last_at_hospital = None
    by_helo = False
    if window and c["cat_a"] > 0:
        lift_at = max(at + helo["notice_min"], window["from"]) + helo["flight_min"]
        first_at_hospital = lift_at + helo["load_min"] + helo["to_hospital_min"]
        last_at_hospital = first_at_hospital + (sorties - 1) * (2 * helo["to_hospital_min"] + helo["load_min"])
        by_helo = last_at_hospital <= window["to"] and first_at_hospital <= road_arrives

    # Strength of the affected unit over 48 hours, with and without rotation
    rate = weather["cold"]["attrition_pct"] / 100
    trigger = rules["force"]["reinforce_below_pct"]

    def project(per_step: float) -> list[float]:
        return [unit["after"] * (1 - per_step) ** (hours / 12) for hours in (0, 12, 24, 48)]

    if unit["after"] <= trigger:
        hours_to_trigger = 0.0
    else:
        hours_to_trigger = 12 * math.log(trigger / unit["after"]) / math.log(1 - rate)

    nine_line = "\n".join([
        f"1  Location            {unit['at']}",
        f"2  Net                 {med['net']}, authenticated",
        f"3  Precedence          {c['cat_a']} x Cat A, {c['cat_b']} x Cat B, {c['cat_c']} x Cat C",
        "4  Special equipment   None",
        f"5  Patients            {c['cat_a'] + c['cat_b']} stretcher, {c['cat_c']} walking",
        f"6  Security at site    {'Enemy in the area, escort advised' if unit['at'] == s['enemy']['at'] else 'No enemy reported at the site'}",
        f"7  Marking             {med['marking']}",
        "8  Status              Own troops",
        "9  Terrain             Snow-covered; cleared landing zone of 50 m x 50 m needed",
    ])
    return {
        "weather": weather, "units": units, "unit": unit, "lost": lost, "at": at, "window": window,
        "by_helo": by_helo, "sorties": sorties, "lift_at": lift_at, "last_at_hospital": last_at_hospital,
        "road_arrives": road_arrives, "nine_line": nine_line,
        "in_window": bool(window) and window["from"] <= at < window["to"],
        "no_rotation": project(rate), "rotated": project(rate * rules["rotation_benefit"]),
        "hours_to_trigger": hours_to_trigger,
    }


def comms_model(s: Scenario, rules: Rules) -> dict:
    c, rule = s["comms"], rules["cyber"]
    hit = [node for node in c["nodes"] if node in c["compromised"]]
    nets = [{**net, "compromised": any(node in hit for node in net["nodes"])} for net in c["nets"]]
    affected = [net for net in nets if net["compromised"]]
    active = next(net for net in nets if not net["compromised"])
    standby = next((net for net in nets if not net["compromised"] and net is not active and net["nodes"]), None)
    t0 = to_min(c["detected"])
    lost = and_list([net["name"] for net in affected])
    actions: list[tuple[float, str]] = []
    if hit:
        note = f" ({active['note']})" if active["note"] else ""
        first = f"Stop all traffic on {lost}; {active['name']} stays in use" if active["order"] == "Primary" else f"Stop all traffic on {lost} and move to {active['name']}{note}"
        actions = [
            (t0, first),
            (t0 + rule["notify_min"], f"Tell formation headquarters on {active['name']}"),
            (t0 + rule["isolate_min"], f"Disconnect {and_list([f'node {node}' for node in hit])} from the data network"),
            (t0 + rule["codes_min"], "Change all authentication codes"),
            (t0 + rule["report_min"], "Send the incident report to DGIS"),
        ]
    main_share = math.ceil(c["signal_teams"] * 2 / 3)
    if not hit or not active["nodes"]:
        teams = []
    elif standby:
        teams = [(active["nodes"][0], main_share), (standby["nodes"][0], c["signal_teams"] - main_share)]
    else:
        teams = [(active["nodes"][0], c["signal_teams"])]
    return {"hit": hit, "nets": nets, "affected": affected, "active": active, "actions": actions, "teams": teams, "detected": t0, "report_due": t0 + rule["report_min"]}


# ---------- summaries ----------


def digest(s: Scenario, rules: Rules) -> str:
    """The scenario in a few lines of text, given to the model when Corpus is on in Chat."""
    m, f, helo = mobility_model(s, rules), force_model(s, rules), s["medical"]["helo"]
    w = m["weather"]
    routes = "; ".join(
        f"{r['name']} {r['km']} km, {'open' if r['open'] else 'blocked by ' + r['reason']}, {span(r['transit_h'])} at {m['speed']:.0f} km/h"
        + (f", {r['bridge']} limit {r['bridge_class_t']} t" if r["bridge"] else "")
        for r in m["routes"]
    )
    own = "; ".join(f"{u['name']} {u['fit']} of {u['establishment']} fit at {u['at']}{' (reserve)' if u['role'] == 'reserve' else ''}" for u in s["own"])
    logistics = ", ".join(f"{line['item']} {str(line['hours']) + ' h' if line.get('hours') else line['state'].lower()}" for line in s["logistics"])
    return "\n".join([
        f"Weather: night minimum {deg(s['weather']['temp_c'])}, wind {s['weather']['wind_kmh']} km/h, wind chill {deg(w['wind_chill'])}, "
        f"cold injury risk {w['cold']['risk']} ({w['cold']['frostbite'].lower()}), exposed duty limited to {span(w['cold']['exposure_min'] / 60)}. "
        f"Snowfall {s['weather']['snowfall_cm']} cm in 24 h. Flying window {w['window']['text'] if w['window'] else 'none'}.",
        f"Avalanche risk: {'; '.join(x['name'] + ' ' + x['risk'] for x in w['sectors'])}.",
        f"Routes: {routes}.",
        f"Vehicles: {', '.join(v['type'] + ' ' + str(v['weight_t']) + ' t' for v in s['vehicles'])}.",
        f"Own forces: {own}. {s['assets']['special_forces']['name']} at {s['assets']['special_forces']['at']}. {s['assets']['artillery']['name']} at {s['assets']['artillery']['position']}.",
        f"Enemy estimate: {s['enemy']['strength']} at {s['enemy']['at']}, forward elements on {s['enemy']['forward_on']}. Enemy to own ratio {f['ratio']:.2f} : 1.",
        f"Casualty evacuation: helicopter at {helo['base']}, {helo['stretchers']} stretchers, {helo['flight_min']} min flight; by road {s['medical']['road_hours']} h to the {s['medical']['road_post']}.",
        f"Communications in order of use: {', '.join(net['name'] for net in s['comms']['nets'])}.",
        f"Logistics: {logistics}.",
    ])


def situation(s: Scenario, rules: Rules) -> dict:
    """What the Situation panel shows: the scenario's reports and state, plus indicators calculated from it."""
    w, f = weather_model(s, rules), force_model(s, rules)
    return {
        "as_of": s["as_of"],
        "indicators": [
            {"label": "Enemy to own", "value": f"{f['ratio']:.2f} : 1", "tone": "bad" if f["deficit"] else "good"},
            {"label": "Wind chill", "value": f"{deg(w['wind_chill'])}, {w['cold']['risk'].lower()}", "tone": w["cold"]["tone"]},
            {"label": "Avalanche", "value": w["worst"]["risk"], "tone": w["worst"]["tone"]},
            {"label": "Flying window", "value": w["window"]["text"] if w["window"] else "None", "tone": "" if w["window"] else "bad"},
        ],
        "reports": s["reports"],
        "forces": [
            {
                "label": f"{unit['name']} (reserve)" if unit["role"] == "reserve" else unit["name"],
                "value": f"{rnd(100 * unit['fit'] / unit['establishment'])}%",
                "tone": eff_tone(100 * unit["fit"] / unit["establishment"], rules),
            }
            for unit in s["own"]
        ],
        "logistics": [
            {
                "label": line["item"],
                "value": f"{line['hours']} h" if line.get("hours") else line["state"],
                "tone": "warn" if line.get("state") == "Low" or line.get("hours", 999) <= 48 else "",
            }
            for line in s["logistics"]
        ],
    }
