"""The workflows: one per output of the scenario (A to G).

Each workflow's compute function is pure. It reads the scenario and returns
    steps     (label, result) pairs shown as the run trace
    cards     the supporting calculations (tiles, table, list, ...)
    facts     the lines handed to the model step
    query     what to look for in the corpus  (or refs, if it searched itself)
    standard  optional standard-format brief; defaults to the facts as bullets
The model only ever sees `facts` and the documents found, and only writes the brief.
"""

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

from .config import PromptsFile
from .corpus import Corpus, extracts, reference
from .scenario import (
    Rules,
    Scenario,
    and_list,
    casualty_model,
    clock,
    comms_model,
    deg,
    eff_tone,
    flies,
    fmt,
    force_model,
    mobility_model,
    risk_tone,
    rnd,
    span,
    to_min,
    weather_model,
)


def cell(text: str, tone: str) -> dict:
    return {"text": text, "tone": tone}


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


# ---------- A: mobility ----------


def mobility(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    m = mobility_model(s, rules)
    w, route, move = m["weather"], m["chosen"], s["movement"]
    table = {
        "head": ["Route", "Status", "Distance", "Transit", "Avalanche risk", "Bridge limit"],
        "rows": [
            [
                f"{r['name']} ({r['role'].lower()})",
                cell("Open", "good") if r["open"] else cell(f"Blocked, {r['reason']}", "bad"),
                f"{r['km']} km",
                span(r["transit_h"]),
                cell(r["avalanche"], risk_tone(r["avalanche"])),
                f"{r['bridge']}, {r['bridge_class_t']} t" if r["bridge"] else "None",
            ]
            for r in m["routes"]
        ],
    }
    speed_fact = f"Convoy speed is {m['speed']:.0f} km/h with {s['weather']['snowfall_cm']} cm of snow forecast in 24 h."
    blocked = [r for r in m["routes"] if not r["open"]]

    if not route:
        first = min(m["routes"], key=lambda r: r["clearance_h"])
        return {
            "steps": [("Assess routes", "No route is open"), ("Find the earliest clearance", f"{first['name']}, about {span(first['clearance_h'])}")],
            "cards": [{"title": "Route analysis", "tiles": [{"value": "No route open", "label": f"Earliest clearance: {first['name']}, about {span(first['clearance_h'])}", "tone": "bad"}], "table": table}],
            "facts": [
                "No route is open: " + "; ".join(f"{r['name']} is blocked by {r['reason']}" for r in blocked) + ".",
                f"The earliest clearance is {first['name']}, in about {span(first['clearance_h'])}.",
                speed_fact,
                "The move cannot start until a route is cleared.",
            ],
            "query": "convoy movement snow-bound axis avalanche debris cleared engineer",
        }

    t, fastest = m["timings"], m["fastest"]
    extra_h = route["transit_h"] - fastest["transit_h"]
    if blocked:
        why = "; ".join(f"{r['name']} is blocked by {r['reason']}" for r in blocked)
    elif route is not fastest:
        why = f"{fastest['name']} is faster but lies under a slope with High avalanche risk"
    else:
        why = "it is the fastest open route"
    refused = [v for v in m["fleet"] if not v["cleared"]]
    if not w["window"]:
        cover = "No flying window is forecast, so the move has no helicopter casevac cover."
    elif t["arrive"] <= w["window"]["to"]:
        cover = f"Arrival is {span((w['window']['to'] - t['arrive']) / 60)} before the flying window closes at {clock(w['window']['to'])}."
    else:
        cover = f"Arrival is after the flying window closes at {clock(w['window']['to'])}, so the last leg has no helicopter casevac cover."
    timeline = []
    if route["chokepoints"]:
        timeline += [["Snow clearing starts", clock(t["clearing_start"])], [f"{route['chokepoints']} chokepoints clear by", clock(t["cleared_by"])]]
    timeline += [["Probe team moves", clock(t["probe"])], ["Main body departs", clock(t["depart"])], ["Arrives", clock(t["arrive"])]]
    trips = plural(m["round_trips"], "round trip")
    longer = f", {span(extra_h)} longer than {fastest['name']}" if extra_h > 0.05 else ""
    if route["bridge"]:
        bridge_step = f"{route['bridge']} {route['bridge_class_t']} t, " + (f"{and_list([v['type'] for v in refused])} not cleared" if refused else "all vehicle types cleared")
        not_cleared = and_list([f"{v['type']} ({v['weight_t']} t)" for v in refused])
        bridge_fact = f"{route['bridge']} is limited to {route['bridge_class_t']} t. " + (f"Not cleared to cross: {not_cleared}." if refused else "All vehicle types are cleared to cross.")
    else:
        bridge_step = "No limiting bridge on this axis"
        bridge_fact = "There is no limiting bridge on this axis."

    route_tiles = [
        {"value": route["name"], "label": "Recommended route", "tone": "good"},
        {"value": span(route["transit_h"]), "label": f"Transit at {m['speed']:.0f} km/h"},
    ]
    if extra_h > 0.05:
        route_tiles.append({"value": f"+{span(extra_h)}", "label": f"Longer than {fastest['name']}", "tone": "warn"})
    route_tiles.append({"value": clock(t["arrive"]), "label": "Arrival"})
    return {
        "steps": [
            ("Assess routes", f"{route['name']} recommended, {span(route['transit_h'])} at {m['speed']:.0f} km/h"),
            ("Check bridge limits", bridge_step),
            ("Size and schedule the convoy", f"{m['carriers'] + m['snow_vehicles']} vehicles, depart {clock(t['depart'])}, arrive {clock(t['arrive'])}"),
            ("Check fuel", f"{trips} from {move['fuel_point']}"),
        ],
        "cards": [
            {"title": "Route analysis", "tiles": route_tiles, "table": table},
            {
                "title": "Vehicles",
                "tiles": [
                    {"value": str(m["carriers"]), "label": f"High-mobility vehicles for {move['troops']} troops"},
                    {"value": str(m["snow_vehicles"]), "label": f"Snow vehicles: {route['chokepoints']} chokepoints and 1 spare" if route["chokepoints"] else "Snow vehicle held as spare"},
                ],
                "table": {
                    "head": ["Vehicle type", "Weight", f"{route['bridge']}, {route['bridge_class_t']} t" if route["bridge"] else "Bridge limit"],
                    "rows": [[v["type"], f"{v['weight_t']} t", cell("Cleared", "good") if v["cleared"] else cell("Not cleared", "bad")] for v in m["fleet"]],
                },
            },
            {"title": "Timings", "table": {"head": ["Event", "Time"], "rows": timeline}, "note": cover},
            {
                "title": "Fuel",
                "tiles": [
                    {"value": f"{fmt(m['per_trip_l'])} L", "label": "Per round trip"},
                    {"value": f"{fmt(move['fuel_stock_l'])} L", "label": f"Stock at {move['fuel_point']}"},
                    {"value": str(m["round_trips"]), "label": "Round trips covered", "tone": "good" if m["round_trips"] >= 2 else "warn"},
                ],
            },
        ],
        "facts": [
            f"Recommended route: {route['name']}, {route['km']} km, because {why}.",
            f"{speed_fact} Transit time is {span(route['transit_h'])}{longer}.",
            f"Vehicles: {m['carriers']} high-mobility vehicles for {move['troops']} troops and {m['snow_vehicles']} snow vehicles.",
            bridge_fact,
            "Timings: " + ", ".join(f"{event.lower()} {time}" for event, time in timeline) + ".",
            f"Fuel: {fmt(m['per_trip_l'])} litres per round trip, {fmt(move['fuel_stock_l'])} litres at {move['fuel_point']}, enough for {trips}.",
            f"Risks: avalanche risk on the route is {route['avalanche']}; wind chill is {deg(w['wind_chill'])} with {w['cold']['risk']} cold injury risk, "
            f"so exposed halts are limited to {span(w['cold']['exposure_min'] / 60)}. {cover}",
        ],
        "query": f"convoy movement snow-bound axis probe team {route['bridge'] or 'bridge'} load classification fuel round trips",
    }


# ---------- B: operational plan ----------


def opplan(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    f, w, a = force_model(s, rules), weather_model(s, rules), s["assets"]
    limit = rules["force"]["reinforce_above_ratio"]
    ratio, with_reserve, trigger = f"{f['ratio']:.2f} : 1", f"{f['ratio_with_reserve']:.2f} : 1", f"{limit} : 1"
    leg, reaction = span(w["cold"]["exposure_min"] / 60), f"{rnd(f['reaction_min'])} min"
    reserve = f["reserve"]
    if f["deficit"]:
        verdict = f"The ratio is above the {trigger} trigger, so reinforcement should be requested. Committing {reserve['name']} brings it to {with_reserve}."
    else:
        verdict = f"The ratio is within the {trigger} trigger. No reinforcement is needed on numbers alone."
    matrix = [
        ["Area domination", and_list([u["name"] for u in f["forward"]]), "Ridge Axis Alpha and Ridge Axis Bravo", f"Platoon-strength patrols, legs of {leg} at most"],
        ["Surveillance grid", "Observation posts", and_list(a["observation_posts"]), f"Overlapping arcs on {a['uav']['sector']}, relief every {leg}"],
        ["Fallback positions", "Forward observation posts", and_list(a["fallback_positions"]), "Occupied if a forward post is compromised"],
        ["Reserve", reserve["name"], reserve["at"], f"{reaction} from any axis"],
        ["Artillery", a["artillery"]["name"], a["artillery"]["position"], f"Alternate at {a['artillery']['alternate']}, fire plans registered on the approach routes"],
        ["UAV watch", a["uav"]["name"], a["uav"]["sector"], f"Picture updated every {a['uav']['revisit_min']} min"],
    ]
    units = []
    for u in s["own"]:
        percent = 100 * u["fit"] / u["establishment"]
        units.append([u["name"], "Reserve" if u["role"] == "reserve" else "Forward", u["at"], f"{u['fit']} of {u['establishment']}", cell(f"{rnd(percent)}%", eff_tone(percent, rules))])
    forward_text = "; ".join(f"{u['name']} {u['fit']} of {u['establishment']} at {u['at']}" for u in f["forward"])
    return {
        "steps": [
            ("Count forward strength", f"{f['own_fit']} fit in {len(f['forward'])} companies"),
            ("Compare with the enemy estimate", f"{ratio} against, trigger {trigger}"),
            ("Place the reserve", f"{reserve['name']}, {reaction} from any axis"),
            ("Build the deployment matrix", f"{len(matrix)} tasks"),
        ],
        "cards": [
            {
                "title": "Force ratio",
                "tiles": [
                    {"value": ratio, "label": "Enemy to own, forward", "tone": "bad" if f["deficit"] else "good"},
                    {"value": str(f["own_fit"]), "label": "Own fit strength, forward"},
                    {"value": fmt(s["enemy"]["strength"]), "label": f"Enemy estimate at {s['enemy']['at']}"},
                    {"value": with_reserve, "label": f"With {reserve['name']} committed", "tone": "warn" if f["ratio_with_reserve"] > limit else ""},
                ],
                "note": verdict,
                "table": {"head": ["Unit", "Role", "Location", "Fit", "Effective"], "rows": units},
            },
            {"title": "Deployment matrix", "table": {"head": ["Task", "Force", "Where", "Detail"], "rows": matrix}},
        ],
        "facts": [
            f"Forward strength: {f['own_fit']} fit in {len(f['forward'])} companies ({forward_text}).",
            f"Enemy estimate: {s['enemy']['strength']} at {s['enemy']['at']}. The enemy to own ratio is {ratio}.",
            verdict,
            f"Reserve: {reserve['name']}, {reserve['fit']} fit, at {reserve['at']}, {reaction} from any axis.",
            *[f"{task}: {force}, {where}. {detail}." for task, force, where, detail in matrix],
        ],
        "query": "patrols observation posts fallback position reserve company reinforcement enemy to own strength ratio",
    }


# ---------- C: tactical recommendation ----------


def tactical(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    m, f, a = mobility_model(s, rules), force_model(s, rules), s["assets"]
    w, enemy, reserve = m["weather"], s["enemy"], f["reserve"]
    ratio, trigger = f"{f['ratio']:.2f} : 1", f"{rules['force']['reinforce_above_ratio']} : 1"
    leg = span(w["cold"]["exposure_min"] / 60)
    blocked = [r for r in m["routes"] if not r["open"]]
    sf, arty = a["special_forces"], a["artillery"]

    actions: list[list[str]] = []
    if sf["available"] == "yes":
        actions.append([f"Deploy the {sf['name']} along {enemy['forward_on']} for ISR and area denial", f"Forward elements are reported on {enemy['forward_on']}, and the detachment is available at {sf['at']}"])
    if f["deficit"]:
        actions.append([f"Reinforce {f['main']['name']} at {f['main']['at']} with one MMG detachment", f"The enemy to own ratio is {ratio}, above the {trigger} trigger"])
    if m["chosen"]:
        prefix = f"{and_list([r['name'] for r in blocked])} blocked; " if blocked else ""
        actions.append([f"Induct {m['carriers']} high-mobility and {m['snow_vehicles']} snow vehicles on {m['chosen']['name']}", f"{prefix}{m['chosen']['name']} is open and takes {span(m['chosen']['transit_h'])}"])
    else:
        actions.append(["Clear a route before committing vehicles", "No route is open"])
    actions.append([f"Hold the {arty['name'].lower()} at {arty['position']}, with {arty['alternate']} as alternate", f"It covers the approach routes in {a['uav']['sector']}"])
    if w["cold"]["tone"] != "good":
        actions.append([f"Limit exposed duty to {leg} and issue Scale B winter equipment", f"Wind chill is {deg(w['wind_chill'])}, with {w['cold']['risk']} cold injury risk"])
    if f["deficit"]:
        actions.append([f"Bring {reserve['name']} to immediate notice at {reserve['at']}", f"Committing it brings the ratio to {f['ratio_with_reserve']:.2f} : 1, and it is {rnd(f['reaction_min'])} min from any axis"])
    contingency = [
        "Deny the area with the pre-registered fire plans.",
        f"Withdraw exposed observation posts to {and_list(a['fallback_positions'])}.",
        f"Open the hotline within {rules['bpm']['hotline_within_min']} min of a confirmed crossing and ask for an emergency meeting.",
        f"Call aviation for casevac and resupply inside the flying window, {w['window']['text']}." if w["window"] else "No flying window is forecast: plan casevac and resupply by road.",
    ]
    numbered = "\n".join(f"{index}. **{action}.** {reason}." for index, (action, reason) in enumerate(actions, 1))
    return {
        "steps": [
            ("Read the latest reports", f"{sum(1 for r in s['reports'] if r.get('urgent'))} urgent of {len(s['reports'])}"),
            ("Run the mobility model", f"{m['chosen']['name']}, {span(m['chosen']['transit_h'])}" if m["chosen"] else "No route open"),
            ("Run the weather model", f"Wind chill {deg(w['wind_chill'])}, {w['cold']['risk'].lower()} risk"),
            ("Run the force model", f"{ratio} against{', above the trigger' if f['deficit'] else ''}"),
            ("Rank the actions", f"{len(actions)} actions, 1 contingency"),
        ],
        "cards": [
            {"title": "Recommended actions, in priority order", "ranked": actions},
            {"title": f"Contingency: forward elements reach {enemy['tripwire']}", "list": contingency},
        ],
        "facts": [
            "Recommended actions in priority order:",
            *[f"{index}. {action}. Reason: {reason}." for index, (action, reason) in enumerate(actions, 1)],
            f"Contingency if forward elements reach {enemy['tripwire']}: {' '.join(contingency)}",
        ],
        "standard": f"{numbered}\n\n**If forward elements reach {enemy['tripwire']}:** {' '.join(contingency)}",
        "query": "reinforcement enemy to own ratio reserve observation posts fallback hotline emergency meeting cold injury",
    }


# ---------- D: weather ----------

AVALANCHE_ACTION = {"High": "Out of bounds until assessed", "Considerable": "Probe teams lead, vehicles spaced out", "Moderate": "Cross with caution", "Low": "No restriction"}


def weather(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    w = weather_model(s, rules)
    cold, forecast = w["cold"], s["weather"]
    leg = span(cold["exposure_min"] / 60)
    kit_low = any(line["item"] == "Winter equipment" and line.get("state") == "Low" for line in s["logistics"])
    flyable = (w["window"]["to"] - w["window"]["from"]) // 60 if w["window"] else 0
    advice = [f"Limit exposed duty to **{leg}** at a time and relieve sentries and observation posts on that cycle."]
    if cold["tone"] != "good":
        advice += [
            "Issue Scale B winter equipment to forward troops before first exposure." + (" Stock is low, so forward companies come first." if kit_low else ""),
            "Open heated shelter points with a medical orderly at each company base.",
        ]
    limits = rules["aviation"]
    return {
        "steps": [
            ("Compute wind chill", f"{deg(w['wind_chill'])} from {deg(forecast['temp_c'])} and {forecast['wind_kmh']} km/h wind"),
            ("Rate cold injury risk", f"{cold['risk']}, exposed duty limited to {leg}"),
            ("Rate avalanche risk by slope", f"{w['worst']['risk']} on {w['worst']['name']}"),
            ("Find the flying window", w["window"]["text"] if w["window"] else "No flyable hours"),
        ],
        "cards": [
            {
                "title": "Cold injury",
                "tiles": [
                    {"value": deg(w["wind_chill"]), "label": "Wind chill", "tone": cold["tone"]},
                    {"value": cold["risk"], "label": "Cold injury risk", "tone": cold["tone"]},
                    {"value": leg, "label": "Longest spell on exposed duty"},
                ],
                "note": f"{cold['frostbite']}.",
                "list": advice,
            },
            {
                "title": "Avalanche risk by slope",
                "table": {
                    "head": ["Slope", "Angle", "New snow, 24 h", "Risk", "Action"],
                    "rows": [[x["name"], f"{x['slope_deg']}°", f"{x['new_snow_cm']} cm", cell(x["risk"], x["tone"]), AVALANCHE_ACTION[x["risk"]]] for x in w["sectors"]],
                },
            },
            {
                "title": "Flying window",
                "tiles": [
                    {"value": w["window"]["text"] if w["window"] else "None", "label": "Helicopter operations", "tone": "good" if w["window"] else "bad"},
                    {"value": f"{flyable} h", "label": "Flyable hours"},
                ],
                "strip": [{"label": f"{slot['hour']:02d}", "good": flies(slot, rules), "title": f"{fmt(slot['vis_m'])} m visibility, {slot['wind_kmh']} km/h wind"} for slot in forecast["hourly"]],
                "note": f"Each cell is one hour from {clock(forecast['hourly'][0]['hour'] * 60)}. Flying needs at least {fmt(limits['min_vis_m'])} m visibility and wind of {limits['max_wind_kmh']} km/h or less.",
            },
        ],
        "facts": [
            f"Forecast: night minimum {deg(forecast['temp_c'])}, wind {forecast['wind_kmh']} km/h, {forecast['snowfall_cm']} cm of snow in 24 h.",
            f"Wind chill is {deg(w['wind_chill'])}. Cold injury risk is {cold['risk']}: {cold['frostbite'].lower()}.",
            f"Exposed duty is limited to {leg} at a time.{' Winter equipment stock is low.' if kit_low else ''}",
            *[f"Avalanche risk on {x['name']} ({x['slope_deg']} degree slope, {x['new_snow_cm']} cm new snow): {x['risk']}. {AVALANCHE_ACTION[x['risk']]}." for x in w["sectors"]],
            f"The flying window is {w['window']['text']}, {flyable} flyable hours. No night flying." if w["window"] else "No flyable hours are forecast today.",
        ],
        "query": "cold injury wind chill frostbite avalanche slope new snow helicopter visibility flying window",
    }


# ---------- E: BPM and diplomacy ----------


def bpm(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    b = s["bpm"]
    observed = clock(to_min(b["observed"]))
    hits = corpus.search(f"{b['grid']} patrol overlap forward movement disengagement procedure hotline emergency meeting notice", limit=4, types=["BPM record", "Protocol"])
    records = [doc for doc in hits if doc["type"] == "BPM record"]
    protocol = next((doc for doc in hits if doc["type"] == "Protocol"), None)
    lead = next((doc for doc in records if b["grid"] in doc["text"]), records[0] if records else None)
    level = rules["bpm"]["level"][b["escalation"]]
    deadline = clock(to_min(b["observed"]) + rules["bpm"]["hotline_within_min"])
    meta = lead.get("meta", {}) if lead else {}
    reliance = f"{lead['id']}{', ' + meta['clause'] if meta.get('clause') else ''}" if lead else ""
    reference_line = "; ".join(part for part in [reliance, protocol["title"] if protocol else ""] if part) or "none on record"
    paragraphs = [f"Forward movement by PLA elements was observed at {b['grid']} at {observed} today."]
    if lead:
        paragraphs.append(f"We refer to the border personnel meeting of {meta.get('date', 'record')} ({reliance}){' and to markers ' + meta['markers'] if meta.get('markers') else ''}.")
    paragraphs.append(f"We request an emergency border personnel meeting{' under the emergency clause, which needs no 72-hour notice' if protocol else ''}.")
    paragraphs.append("We ask that patrols halt and withdraw to the agreed positions until the meeting is held.")
    message = "\n".join([
        "**To:** Counterpart formation, by hotline",
        f"**From:** Own formation headquarters, at the level of {level.lower()}",
        f"**Reference:** {reference_line}",
        "",
        *[f"{index}. {text}" for index, text in enumerate(paragraphs, 1)],
        "",
        "This message is sent to maintain peace and tranquillity along the border.",
    ])
    total_records = sum(1 for doc in corpus.documents if doc["type"] == "BPM record")
    if lead:
        approach = [f"Rely on **{reliance}**: {meta.get('outcome', '').lower()}."]
        if meta.get("markers"):
            approach.append(f"Quote markers **{meta['markers']}** as the agreed positions.")
        approach.append("Attach the record and the agreed map to the message.")
        lead_fact = f"Record to rely on: {reliance}, meeting of {meta.get('date', '')}. Issue then: {meta.get('issue', '')}. Outcome: {meta.get('outcome', '')}." + (f" Agreed markers: {meta['markers']}." if meta.get("markers") else "")
    else:
        approach = ["No earlier record mentions this location. Request the meeting on the protocol alone."]
        lead_fact = "No earlier meeting record mentions this location."
    return {
        "refs": hits,
        "steps": [
            ("Read the incident", f"{b['grid']}, observed {observed}, escalation {b['escalation'].lower()}"),
            ("Search the meeting records", f"{len(records)} of {total_records} records apply", "Corpus search"),
            ("Check the notice rule", "Emergency clause: no 72-hour notice" if protocol else "No protocol found"),
            ("Set the channel and level", f"Hotline by {deadline}, {level.lower()}"),
        ],
        "cards": [
            {
                "title": "Recommended approach",
                "tiles": [
                    {"value": deadline, "label": f"Open the hotline by ({rules['bpm']['hotline_within_min']} min rule)", "tone": "warn"},
                    {"value": "No notice" if protocol else "Check notice", "label": "Emergency clause, Annex F" if protocol else "Protocol not found in the corpus"},
                    {"value": level, "label": f"Interlocutor for {b['escalation'].lower()} escalation"},
                ],
                "list": approach,
            },
            {
                "title": "Earlier meeting records",
                "table": {"head": ["Reference", "Date", "Issue", "Outcome"], "rows": [[doc["id"], doc.get("meta", {}).get("date", ""), doc.get("meta", {}).get("issue", ""), doc.get("meta", {}).get("outcome", "")] for doc in records]},
            },
        ],
        "facts": [
            f"Observed: forward movement by PLA elements at {b['grid']} at {observed} today. Escalation level: {b['escalation']}.",
            lead_fact,
            "Emergency meeting protocol, Annex F: an emergency meeting may be requested by hotline at any time and needs no 72-hour notice." if protocol else "No protocol on notice was found.",
            f"The hotline is to be opened by {deadline}. Our interlocutor: {level}.",
            "The request: an emergency border personnel meeting, and that patrols halt and withdraw to the agreed positions until it is held.",
            "Closing sentence to use: This message is sent to maintain peace and tranquillity along the border.",
        ],
        "standard": message,
    }


# ---------- F: casualty planning ----------


def casualty(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    k, c, med = casualty_model(s, rules), s["casualties"], s["medical"]
    unit, trigger = k["unit"], rules["force"]["reinforce_below_pct"]
    f = force_model(s, rules)
    reserve = f["reserve"]
    sorties = plural(k["sorties"], "sortie")
    plan = []
    if c["cat_a"]:
        if k["by_helo"]:
            plan.append(["Cat A", str(c["cat_a"]), f"Helicopter from {med['helo']['base']}, {sorties}", med["hospital"], f"Lift {clock(k['lift_at'])}, last arrival {clock(k['last_at_hospital'])}"])
        else:
            plan.append(["Cat A", str(c["cat_a"]), "Road, with a medical escort", med["road_post"], f"Arrive {clock(k['road_arrives'])}"])
    if c["cat_b"]:
        plan.append(["Cat B", str(c["cat_b"]), "Road", med["road_post"], f"Arrive {clock(k['road_arrives'])}"])
    if c["cat_c"]:
        plan.append(["Cat C", str(c["cat_c"]), "Treated in place", med["aid_post"], "Review after 48 h"])
    if not c["cat_a"]:
        means = "No Cat A casualties"
    elif k["by_helo"]:
        means = f"{k['sorties']} helicopter {'sortie' if k['sorties'] == 1 else 'sorties'}, first lift {clock(k['lift_at'])}"
    else:
        means = f"By road, arriving {clock(k['road_arrives'])}"
    if not k["window"]:
        why_road = "No flying window is forecast."
    elif k["in_window"]:
        why_road = "The lifts cannot all reach hospital before the flying window closes."
    else:
        why_road = f"The incident is outside the flying window ({k['window']['text']}), and the road is quicker than waiting for it."
    when = "now" if k["hours_to_trigger"] == 0 else f"in about {rnd(k['hours_to_trigger'])} h"
    away = f"{reserve['name']} is {rnd(f['reaction_min'])} min away at {reserve['at']}."
    if k["hours_to_trigger"] == 0:
        outlook = f"{unit['name']} is already at or below the {trigger}% trigger. Reinforce now: {away}"
    else:
        outlook = f"Without rotation {unit['name']} reaches the {trigger}% trigger {when}. {away}"

    def row(values: list[float]) -> list[dict]:
        return [cell(f"{rnd(value)}%", eff_tone(value, rules)) for value in values]

    def tile_label(u: dict) -> str:
        if u["name"] == unit["name"] and k["lost"]:
            return f"{u['name']}, was {rnd(u['before'])}%"
        return f"{u['name']} (reserve)" if u["role"] == "reserve" else u["name"]

    evacuation: dict[str, Any] = {"title": "Evacuation plan", "table": {"head": ["Category", "Number", "Means", "To", "Timing"], "rows": plan or [["None reported", "0", "", "", ""]]}}
    by_road = bool(c["cat_a"]) and not k["by_helo"]
    if by_road:
        evacuation["note"] = why_road
    return {
        "steps": [
            ("Update unit strength", f"{unit['name']} {unit['fit']} to {unit['fit_after']} fit, {rnd(unit['after'])}% effective"),
            ("Check the flying window", f"Incident {clock(k['at'])}, {'inside' if k['in_window'] else 'outside'} {k['window']['text']}" if k["window"] else "No flying window"),
            ("Plan the evacuation", means),
            ("Fill the 9-line request", f"Ready to send on {med['net']}"),
            ("Project strength over 48 h", f"{trigger}% trigger {when} without rotation"),
        ],
        "cards": [
            {"title": "Effect on strength", "tiles": [{"value": f"{rnd(u['after'])}%", "label": tile_label(u), "tone": eff_tone(u["after"], rules)} for u in k["units"]]},
            evacuation,
            {"title": "9-line MEDEVAC request, filled in", "pre": k["nine_line"]},
            {
                "title": f"{unit['name']}: strength over 48 hours",
                "table": {"head": ["", "Now", "+12 h", "+24 h", "+48 h"], "rows": [["With rotation", *row(k["rotated"])], ["Without rotation", *row(k["no_rotation"])]]},
                "note": outlook,
            },
        ],
        "facts": [
            f"Incident: {c['cat_a']} Cat A, {c['cat_b']} Cat B and {c['cat_c']} Cat C casualties in {unit['name']} at {unit['at']}, at {clock(k['at'])}.",
            *[f"{category} ({number}): {how}, to the {to}. {timing}." for category, number, how, to, timing in plan],
            *([why_road] if by_road else []),
            f"{unit['name']} falls from {unit['fit']} to {unit['fit_after']} fit, {rnd(unit['after'])}% effective.",
            f"Cold attrition at {k['weather']['cold']['risk']} risk: with rotation {rnd(k['rotated'][2])}% effective after 24 h, without rotation {rnd(k['no_rotation'][2])}%.",
            outlook,
            f"The 9-line MEDEVAC request is filled in and goes on {med['net']}.",
        ],
        "query": "casualty evacuation Cat A helicopter flying window road forward medical post 9-line reinforcement 80 per cent",
    }


# ---------- G: cyber and communications ----------


def cyber(s: Scenario, rules: Rules, corpus: Corpus) -> dict:
    k, c = comms_model(s, rules), s["comms"]
    active = k["active"]

    def status(net: dict) -> dict | str:
        if net["compromised"]:
            return cell("Do not use", "bad")
        return cell("In use", "good") if net is active else "Available"

    net_table = {"head": ["Net", "Order", "Runs through", "Status", "Note"], "rows": [[net["name"], net["order"], and_list(net["nodes"]) if net["nodes"] else "No node", status(net), net["note"]] for net in k["nets"]]}
    if not k["hit"]:
        return {
            "steps": [("Map nodes to nets", "No node reported compromised"), ("Select the net", f"{active['name']} stays in use")],
            "cards": [{"title": "Nets", "tiles": [{"value": "No compromise", "label": "All nets available", "tone": "good"}], "table": net_table}],
            "facts": ["No node is reported compromised.", f"{active['name']} stays in use as the primary net."],
            "query": "communications compromise net node primary alternate contingency emergency",
        }
    nodes, lost = and_list(k["hit"]), and_list([net["name"] for net in k["affected"]])
    one_node, one_net = len(k["hit"]) == 1, len(k["affected"]) == 1
    team_text = and_list([f"{count} to {node}" for node, count in k["teams"]]) if k["teams"] else "No relay to man"
    timeline = [[clock(at), action] for at, action in k["actions"]]
    note = f" ({active['note']})" if active["note"] else ""
    restore = ", ".join(c["restore_order"])
    report = "\n".join([
        "**Incident report: communications compromise**",
        f"**Detected:** {clock(k['detected'])}. **Report due:** {clock(k['report_due'])}.",
        "",
        f"1. {'Node ' + nodes + ' is' if one_node else 'Nodes ' + nodes + ' are'} reported compromised. " + (f"{lost} {'is' if one_net else 'are'} not to be used." if lost else "No net runs through them."),
        f"2. The net in use is {active['name']}{note}.",
        "3. Actions: " + "; ".join(f"by {time}, {action[0].lower()}{action[1:]}" for time, action in timeline) + ".",
        f"4. Signal teams: {team_text}. Restoration order: {restore}.",
    ])
    return {
        "steps": [
            ("Map nodes to nets", f"{plural(len(k['hit']), 'node')} compromised, {plural(len(k['affected']), 'net')} lost"),
            ("Select the next net", f"{active['name']} ({active['order'].lower()})"),
            ("Schedule containment", f"{len(k['actions'])} actions, report due {clock(k['report_due'])}"),
            ("Reassign signal teams", team_text),
        ],
        "cards": [
            {
                "title": "Nets",
                "tiles": [
                    {"value": active["name"], "label": f"Move to ({active['order'].lower()})", "tone": "good"},
                    {"value": lost or "None", "label": "Do not use", "tone": "bad" if lost else ""},
                    {"value": clock(k["report_due"]), "label": "Incident report due"},
                ],
                "table": net_table,
            },
            {"title": "Containment timeline", "table": {"head": ["By", "Action"], "rows": timeline}},
            {
                "title": "Restoration and signal teams",
                "list": [f"Restore in this order: **{restore}**.", f"Signal teams: {team_text}.", "Treat all earlier traffic on the lost nets as read by the adversary."],
            },
        ],
        "facts": [
            f"Detected at {clock(k['detected'])}: {'node ' + nodes if one_node else 'nodes ' + nodes} reported compromised.",
            f"Nets not to be used: {lost}." if lost else "No net runs through the compromised nodes.",
            f"Net now in use: {active['name']}, the {active['order'].lower()} net{', ' + active['note'] if active['note'] else ''}.",
            *[f"By {time}: {action}." for time, action in timeline],
            f"Signal teams: {team_text}.",
            f"Restoration order: {restore}.",
        ],
        "standard": report,
        "query": "communications compromise net node COMSEC authentication codes incident report restore order signal detachments",
    }


# ---------- registry ----------


def number(path: str, label: str, low: float, high: float, step: float = 1, unit: str = "") -> dict:
    return {"path": path, "label": label, "type": "number", "min": low, "max": high, "step": step, "unit": unit}


def choice(path: str, label: str, options: list[str] | str) -> dict:
    return {"path": path, "label": label, "type": "select", "options": options}


@dataclass(frozen=True)
class Workflow:
    id: str
    letter: str
    label: str
    summary: str
    inputs: list[dict]
    compute: Callable[[Scenario, Rules, Corpus], dict]


# Options starting with "@" are filled from the scenario (see _options).
WORKFLOWS = [
    Workflow("mobility", "A", "Mobility plan", "Chooses the route, sizes the convoy and sets the timings from route, bridge and weather data.", [
        choice("routes.0.status", "Axis Alpha", ["open", "blocked"]),
        number("weather.snowfall_cm", "Snowfall, next 24 h", 0, 200, 5, "cm"),
        number("routes.1.bridge_class_t", "Bridge 52B limit", 5, 70, 1, "t"),
        number("movement.troops", "Troops to move", 10, 600, 10),
        {"path": "movement.departure", "label": "Main body departs", "type": "time"},
    ], mobility),
    Workflow("opplan", "B", "Operational plan", "Compares own and enemy strength, places the reserve and lays out the deployment matrix.", [
        number("enemy.strength", "Enemy strength estimate", 50, 3000, 10),
        number("own.0.fit", "Alpha Coy fit", 0, 120),
        number("own.1.fit", "Bravo Coy fit", 0, 120),
        number("own.2.fit", "Charlie Coy fit", 0, 120),
    ], opplan),
    Workflow("tactical", "C", "Tactical recommendation", "Runs the mobility, weather and force models together and ranks the actions that follow from them.", [
        number("enemy.strength", "Enemy strength estimate", 50, 3000, 10),
        choice("assets.special_forces.available", "Para SF detachment available", ["yes", "no"]),
        choice("routes.0.status", "Axis Alpha", ["open", "blocked"]),
        number("weather.temp_c", "Night minimum", -60, 10, 1, "°C"),
    ], tactical),
    Workflow("weather", "D", "Weather analysis", "Turns the forecast into cold injury limits, avalanche risk by slope and the flying window.", [
        number("weather.temp_c", "Night minimum", -60, 10, 1, "°C"),
        number("weather.wind_kmh", "Wind", 0, 120, 5, "km/h"),
        number("weather.snowfall_cm", "Snowfall, next 24 h", 0, 200, 5, "cm"),
    ], weather),
    Workflow("bpm", "E", "BPM and diplomacy", "Finds the earlier meeting records that apply, sets the channel and level, and drafts the hotline message.", [
        {"path": "bpm.grid", "label": "Location of the incident", "type": "text"},
        {"path": "bpm.observed", "label": "Observed at", "type": "time"},
        choice("bpm.escalation", "Escalation", ["Low", "Elevated", "High"]),
    ], bpm),
    Workflow("casualty", "F", "Casualty planning", "Plans the evacuation, fills the 9-line request and projects the unit's strength over 48 hours.", [
        choice("casualties.unit", "Unit", "@forward_units"),
        number("casualties.cat_a", "Cat A, immediate", 0, 40),
        number("casualties.cat_b", "Cat B, urgent", 0, 40),
        number("casualties.cat_c", "Cat C, routine", 0, 40),
        {"path": "casualties.time", "label": "Time of incident", "type": "time"},
    ], casualty),
    Workflow("cyber", "G", "Cyber and communications", "Works out which nets are lost when nodes are compromised, the net to move to, and the containment timeline.", [
        {"path": "comms.compromised", "label": "Nodes reported compromised", "type": "chips", "options": "@comms_nodes"},
        {"path": "comms.detected", "label": "Detected at", "type": "time"},
    ], cyber),
]
BY_ID = {workflow.id: workflow for workflow in WORKFLOWS}


def _get(holder: Any, path: str) -> Any:
    for key in path.split("."):
        holder = holder[int(key)] if isinstance(holder, list) else holder[key]
    return holder


def _set(holder: Any, path: str, value: Any) -> None:
    *keys, last = path.split(".")
    for key in keys:
        holder = holder[int(key)] if isinstance(holder, list) else holder[key]
    if isinstance(holder, list):
        holder[int(last)] = value
    else:
        holder[last] = value


def _options(field: dict, base: Scenario) -> list[str]:
    options = field.get("options", [])
    if options == "@forward_units":
        return [unit["name"] for unit in base["own"] if unit["role"] == "forward"]
    if options == "@comms_nodes":
        return list(base["comms"]["nodes"])
    return options


def fields(base: Scenario) -> dict[str, dict]:
    """Every input any workflow offers, by path. Only these parts of the scenario can be changed from the UI."""
    return {field["path"]: {**field, "options": _options(field, base)} for workflow in WORKFLOWS for field in workflow.inputs}


def describe(workflow: Workflow, base: Scenario, prompts: PromptsFile) -> dict:
    """A workflow as the UI needs it: its inputs with their starting values, and whether it has a model step."""
    brief = prompts.briefs[workflow.id]
    return {
        "id": workflow.id, "letter": workflow.letter, "label": workflow.label, "summary": workflow.summary,
        "inputs": [{**field, "options": _options(field, base), "value": _get(base, field["path"])} for field in workflow.inputs],
        "brief": {"enabled": brief.enabled, "title": brief.title},
    }


def apply_inputs(base: Scenario, inputs: dict[str, Any]) -> Scenario:
    """A copy of the scenario with the UI's input values applied. Raises ValueError for anything not offered."""
    scenario, allowed = copy.deepcopy(base), fields(base)
    for path, value in inputs.items():
        field = allowed.get(path)
        if field is None:
            raise ValueError(f"'{path}' is not an input of any workflow.")
        kind = field["type"]
        if kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field['label']} must be a number.")
            value = min(field["max"], max(field["min"], value))
            value = int(value) if float(value).is_integer() else float(value)
        elif kind == "select":
            if value not in field["options"]:
                raise ValueError(f"{field['label']} must be one of: {', '.join(field['options'])}.")
        elif kind == "time":
            if not isinstance(value, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
                raise ValueError(f"{field['label']} must be a time such as 05:30.")
        elif kind == "chips":
            if not isinstance(value, list) or any(item not in field["options"] for item in value):
                raise ValueError(f"{field['label']} must be chosen from: {', '.join(field['options'])}.")
            value = [option for option in field["options"] if option in value]
        else:
            if not isinstance(value, str) or not value.strip() or len(value) > 40:
                raise ValueError(f"{field['label']} must be 1 to 40 characters of text.")
            value = value.strip()
        _set(scenario, path, value)
    return scenario


@dataclass
class Computed:
    result: dict  # what the UI shows
    facts: list[str]  # what the model step is given
    documents: list[dict]  # the references found, with their text


def run(workflow: Workflow, scenario: Scenario, rules: Rules, corpus: Corpus) -> Computed:
    raw = workflow.compute(scenario, rules, corpus)
    documents = raw.get("refs")
    steps = [{"label": step[0], "detail": step[1], "kind": step[2] if len(step) > 2 else "Calculation"} for step in raw["steps"]]
    if documents is None:
        documents = corpus.search(raw["query"], limit=3)
        found = f"{len(documents)} of {len(corpus.documents)} sample documents" if documents else "No matching document"
        steps.append({"label": "Find references", "detail": found, "kind": "Corpus search"})
    result = {
        "steps": steps,
        "cards": raw["cards"],
        "refs": [reference(doc) for doc in documents],
        "standard": raw.get("standard") or "\n".join(f"- {fact}" for fact in raw["facts"]),
    }
    return Computed(result=result, facts=raw["facts"], documents=documents)


def draft_messages(prompts: PromptsFile, workflow: Workflow, computed: Computed, model_label: str, language: str) -> tuple[str, str]:
    """The system and user messages for the model step: the task, the calculated facts and the references found."""
    facts = "\n".join(f"- {fact}" for fact in computed.facts)
    user = f"TASK\n{prompts.briefs[workflow.id].task} {prompts.languages[language]}\n\nFACTS\n{facts}\n\nREFERENCES\n{extracts(computed.documents) or 'None.'}"
    return prompts.system.replace("{model}", model_label), user
