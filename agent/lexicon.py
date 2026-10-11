"""Shared vocabulary for the rule-based NLU and the synthetic data generator."""

from __future__ import annotations

import re

# canonical feature key (matches structured.safercar.FEATURES) -> phrases
FEATURE_SYNONYMS: dict[str, list[str]] = {
    "abs": ["abs", "anti-lock brakes", "antilock brakes", "anti lock brakes", "anti-lock braking"],
    "electronic_stability_control": ["electronic stability control", "stability control", "esc"],
    "forward_collision_warning": ["forward collision warning", "collision warning", "fcw"],
    "automatic_emergency_braking": [
        "automatic emergency braking", "emergency braking", "aeb", "crash imminent braking",
        "automatic braking", "auto braking", "auto brake",
    ],
    "dynamic_brake_support": ["dynamic brake support", "brake assist"],
    "lane_departure_warning": ["lane departure warning", "lane departure", "lane keeping", "lane assist", "lane keep assist", "ldw"],
    "blind_spot_detection": ["blind spot detection", "blind spot monitoring", "blind spot monitor", "blind spot", "blind-spot", "bsm"],
    "adaptive_cruise_control": ["adaptive cruise control", "adaptive cruise", "radar cruise"],
    "backup_camera": ["backup camera", "back-up camera", "rearview camera", "rear view camera", "reverse camera", "rear camera"],
    "daytime_running_lights": ["daytime running lights", "daytime running lamps", "drl"],
    "head_side_airbags": ["curtain airbags", "curtain airbag", "head airbags", "side curtain"],
    "torso_side_airbags": ["side airbags", "side airbag", "torso airbags"],
    "pelvis_side_airbags": ["pelvis airbags", "pelvic airbag"],
    "knee_airbags": ["knee airbags", "knee airbag", "knee bolster"],
    "seat_belt_pretensioners": ["seat belt pretensioners", "pretensioners", "pretensioner"],
    "seat_belt_load_limiters": ["load limiters", "load limiter"],
    "automatic_crash_notification": ["automatic crash notification", "crash notification"],
}

RATING_WORDS = re.compile(
    r"\b(rating|ratings|rated|stars?|crash[- ]test\w*|safety score|how safe|safe|safest|"
    r"rollover|frontal|side[- ]impact|side pole|5[- ]star|five[- ]star|ncap)\b",
    re.I,
)
RECALL_WORDS = re.compile(r"\b(?:(recall|recalls|recalled|safety campaign|open campaign))\b", re.I)
POSSESSIVE_VEHICLE = re.compile(
    r"\b(my|our|his|her|their|this|that)\s+(?:\w+\s+){0,2}?(car|truck|suv|vehicle|van|minivan|pickup|sedan|coupe|ride|wagon|hatchback)\b",
    re.I,
)
OUT_OF_SCOPE = re.compile(
    r"\b(price|prices|msrp|cost|costs|insurance|insure|financ\w*|"
    r"loan|lease|leasing|0-60|resale|trade[- ]in|dealer\w*|"
    r"warranty|lawyer|lawsuit|stock price|recipe|movie|song|homework)\b",
    re.I,
)
# Questions about the bot itself ("can I ask you anything?", "what can you do?").
FUEL_ECONOMY_WORDS = re.compile(r"\b(mpg|mpge|fuel economy|gas mileage|fuel efficiency|fuel consumption|emissions)\b", re.I)
VEHICLE_INFO_WORDS = re.compile(
    r"\b(mpg|mpge|fuel economy|gas mileage|fuel efficiency|fuel consumption|emissions|engine|engines|"
    r"horsepower|hp|torque|transmission|drivetrain|fuel type|specifications?|specs|history|overview)\b", re.I,
)

# Answered with a description of what it covers, unless the message also
# names a vehicle, a feature or another safety topic.
CAPABILITY = re.compile(
    r"^\s*(?:(?:can|could|may) i ask\b|what (?:can|do|should) (?:you|i) (?:do|ask|know|answer|help)"
    r"|what (?:kind|sort|type)s? of (?:questions|things)|how (?:can|do) you help|what are you\b"
    r"|who (?:are|made) you|are you (?:a |an )?(?:bot|ai|robot|human|real))",
    re.I,
)
# "What data were you trained on?", "where does your information come from?"
SOURCES_QUESTION = re.compile(
    r"\b(?:what|which)\s+(?:data|sources?|information|info)\b.{0,30}\b(?:you|your)\b"
    r"|\b(?:trained on|train(?:ed)? you|your (?:data|sources?|information|knowledge))\b"
    r"|where (?:does|do) (?:your|you get|the) (?:data|information|answers?|info)",
    re.I,
)
# Two vehicles in one question: "Camry or Accord, which is safer?", "X vs Y".
COMPARE_CUE = re.compile(r"\b(vs\.?|versus|compare\w*|safer|safest|better|difference between)\b", re.I)
COMPARE_SPLIT = re.compile(r"\s+(?:vs\.?|versus|or|compared (?:to|with)|than|and|with)\s+", re.I)
# A question about how something works, not about the vehicle in context.
CONCEPT_CUE = re.compile(
    r"\b(how (?:does|do|is|are|can|should)|what (?:is|are) (?:a|an)\b|what does .{1,40} mean|explain|why\b|"
    r"difference between)",
    re.I,
)
# "what about ...", "and the side rating?": continues the previous topic.
FOLLOW_UP_CUE = re.compile(r"^\s*(?:and\b|also\b|what about\b|how about\b|what'?s the\b|what is the\b|plus\b)", re.I)
# "Which car has the highest safety rating?", "safest 2024 SUV": a ranking
# across all vehicles, which the ratings data can't settle (many share 5 stars).
_VEHICLE_NOUN = r"(?:car|vehicle|suv|truck|pickup|sedan|minivan|van|crossover|ev|model|brand|make)s?"
RANKING = re.compile(
    r"\b(?:which|what)\s+(?:\w+\s+){0,2}?" + _VEHICLE_NOUN + r"\b.{0,40}\b(?:highest|best|safest|top|most)\b"
    r"|\b(?:safest|top[- ]rated|highest[- ]rated|best[- ]rated|most safe)\s+(?:\d{4}\s+)?"
    r"(?:(?:new|used|family|small|midsize|compact|electric|luxury)\s+)?" + _VEHICLE_NOUN + r"\b",
    re.I,
)
SMALLTALK = re.compile(
    r"^\s*(?:(?:hi|hello|hey|yo|thanks|thank you|thx|good (?:morning|afternoon|evening)|who are you|"
    r"what can you do|help|bye|goodbye|ok|okay|cool|great)\b[\s!.?]*$"
    r"|(?:thanks|thank you|thx|great|perfect|awesome)\b[^?]{0,30}$)",
    re.I,
)
# Everyday words that collide with model names by prefix ("Is" -> Lexus IS,
# "Golf", "Edge" are handled by COMMON_WORD_MODELS). Without an explicit make
# these are never read as a model.
FUNCTION_WORDS = frozenset("""
a an the is it its as at on in up go do so my me we us or if by to of for and not no yes
one all any new can will may are was be been how what when which who why does did has have
had get got his her our your their this that these those there here with from about
""".split())

PREDICTION = re.compile(
    r"\b(will (my|the|it)\b.*\b(fail|break|explode|deploy|catch fire)|when will\b|how long (will|until)\b|"
    r"likely to (fail|crash)|predict"
    # forecasts: "which car will be safest in 2030?", "next year's safest SUV"
    r"|will\b(?:\s+\w+){0,4}?\s+(?:be|become|get)\s+(?:the\s+)?(?:safest|safer|most|least|best|worst)"
    r"|(?:safest|safer|best|worst)\b.{0,40}\b(?:in|by)\s+(?:20[3-9]\d|next year|the future)"
    r"|next year'?s\s+(?:safest|best)|in the future)\b",
    re.I,
)
DOMAIN_WORDS = re.compile(
    r"\b(car|cars|vehicle|vehicles|truck|suv|van|drive|driving|driver|brake|brakes|braking|airbag|airbags|"
    r"seat|belt|seatbelt|crash|crashes|safety|safe|nhtsa|recall|recalls|tire|tires|collision|lane|camera|"
    r"rating|ratings|stability|cruise|blind|steering|child seat|car seat|headlight|headlights|"
    r"warning|assist|ncap|rollover|defect|complaint|vin|motor|automotive|traction|esc|abs|aeb|"
    # crash-test vocabulary from the IIHS protocols and rating guidelines (ADR-009)
    r"iihs|overlap|small overlap|moderate overlap|side impact|frontal|crash test|crash tests|crashworthiness|"
    r"dummy|dummies|barrier|mdb|intrusion|injury|injuries|kinematics|submarining|restraint|restraints|"
    r"excursion|occupant|occupants|head protection|pelvis|torso|thorax|femur|tibia|hybrid iii|thor|worldsid|"
    r"sid-iis|load cell|test protocol|rating guidelines|demerits?|marginal|top safety pick)\b",
    re.I,
)

DRIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:(awd|all[- ]wheel[- ]drive|all[- ]wheel))\b", re.I), "AWD"),
    (re.compile(r"\b(?:(fwd|front[- ]wheel[- ]drive|front[- ]wheel))\b", re.I), "FWD"),
    (re.compile(r"\b(?:(rwd|rear[- ]wheel[- ]drive|rear[- ]wheel))\b", re.I), "RWD"),
    (re.compile(r"\b(?:(4wd|4x4|four[- ]wheel[- ]drive|4 wheel drive))\b", re.I), "4WD"),
    (re.compile(r"\b(?:(2wd|4x2|two[- ]wheel[- ]drive))\b", re.I), "2WD"),
]

BODY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:crew ?cab|super ?crew|double ?cab|quad ?cab)\b", re.I), "CREWCAB"),
    (re.compile(r"\b(?:extended ?cab|super ?cab|access ?cab|king ?cab)\b", re.I), "EXTENDEDCAB"),
    (re.compile(r"\b(?:regular ?cab|single ?cab)\b", re.I), "REGULARCAB"),
    (re.compile(r"\b(?:sedan|4[- ]door)\b", re.I), "SEDAN"),
    (re.compile(r"\b(?:coupe|2[- ]door)\b", re.I), "COUPE"),
    (re.compile(r"\b(?:convertible|cabriolet|roadster)\b", re.I), "CONVERTIBLE"),
    (re.compile(r"\b(?:hatchback|hatch)\b", re.I), "HATCHBACK"),
    (re.compile(r"\b(?:wagon)\b", re.I), "WAGON"),
    (re.compile(r"\b(?:minivan)\b", re.I), "MINIVAN"),
]


def find_features(text: str) -> list[str]:
    lowered = f" {text.lower()} "
    found: list[str] = []
    for key, phrases in FEATURE_SYNONYMS.items():
        for phrase in phrases:
            if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", lowered):
                found.append(key)
                break
    return found
