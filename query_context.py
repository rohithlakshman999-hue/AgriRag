# ============================================================
# KrishiJal AI - Automatic Farmer Context Extraction
# ============================================================
#
# Detects:
#   - Crop
#   - Growth stage
#   - Irrigation method
#   - Water availability
#   - Farmer intent
#
# Also combines detected context with sidebar context
# to create a context-aware retrieval query.
# ============================================================


# ============================================================
# CROP DETECTION
# ============================================================

CROPS = {

    "Tomato": [
        "tomato",
        "tomatoes"
    ],

    "Rice": [
        "rice",
        "paddy",
        "paddy crop"
    ],

    "Wheat": [
        "wheat",
        "wheat crop"
    ],

    "Maize": [
        "maize",
        "corn",
        "maize crop"
    ],

    "Sugarcane": [
        "sugarcane",
        "sugar cane",
        "sugarcane crop"
    ],

    "Cotton": [
        "cotton",
        "cotton crop"
    ],

    "Groundnut": [
        "groundnut",
        "ground nuts",
        "peanut",
        "peanuts"
    ],

    "Vegetables": [
        "vegetable",
        "vegetables",
        "vegetable crop"
    ]
}


# ============================================================
# GROWTH STAGE DETECTION
# ============================================================

GROWTH_STAGES = {

    "Seedling": [
        "seedling",
        "seedlings",
        "seedling stage",
        "early seedling"
    ],

    "Vegetative": [
        "vegetative",
        "vegetative stage",
        "vegetative growth",
        "vegetative period"
    ],

    "Flowering": [
        "flowering",
        "flowering stage",
        "flower stage",
        "during flowering",
        "at flowering",
        "when flowering",
        "flowering period"
    ],

    "Fruit-set": [
        "fruit set",
        "fruit-set",
        "fruit setting",
        "fruit setting stage",
        "fruit set stage",
        "fruiting"
    ],

    "Grain filling": [
        "grain filling",
        "grain filling stage",
        "grain-fill",
        "grain fill",
        "during grain filling",
        "grain filling period"
    ],

    "Maturity": [
        "maturity",
        "mature stage",
        "maturity stage",
        "at maturity",
        "during maturity",
        "ripening",
        "ripening stage"
    ]
}


# ============================================================
# IRRIGATION METHOD DETECTION
# ============================================================

IRRIGATION_METHODS = {

    "Flood irrigation": [
        "flood irrigation",
        "flooded irrigation",
        "flooding",
        "flood method",
        "flood system"
    ],

    "Drip irrigation": [
        "drip irrigation",
        "drip system",
        "drip method",
        "drip"
    ],

    "Sprinkler irrigation": [
        "sprinkler irrigation",
        "sprinkler system",
        "sprinkler method",
        "sprinkler"
    ],

    "Furrow irrigation": [
        "furrow irrigation",
        "furrow method",
        "furrow system",
        "furrow"
    ]
}


# ============================================================
# WATER AVAILABILITY DETECTION
# ============================================================

WATER_LEVELS = {

    "Severely limited": [
        "severely limited",
        "severely limited water",
        "water is severely limited",
        "severe water shortage",
        "severe water scarcity",
        "extreme water shortage",
        "extremely low water",
        "very little water",
        "almost no water"
    ],

    "Low": [
        "low water",
        "low water availability",
        "water is low",
        "water availability is low",
        "water availability low",
        "limited water",
        "limited water availability",
        "water is limited",
        "water availability is limited",
        "water availability limited",
        "water is scarce",
        "water scarcity",
        "water shortage",
        "low availability of water",
        "limited availability of water",
        "less water",
        "not much water",
        "water is not enough",
        "not enough water"
    ],

    "Moderate": [
        "moderate water",
        "moderate water availability",
        "water availability is moderate",
        "water availability moderate",
        "moderate amount of water"
    ],

    "Abundant": [
        "abundant water",
        "abundant water availability",
        "water is abundant",
        "water availability is abundant",
        "plenty of water",
        "sufficient water",
        "enough water",
        "adequate water",
        "water is sufficient"
    ]
}


# ============================================================
# INTENT DETECTION
# ============================================================

INTENTS = {

    # --------------------------------------------------------
    # Irrigation scheduling
    # --------------------------------------------------------

    "Irrigation Scheduling": [

        "irrigation interval",
        "irrigation intervals",

        "watering interval",
        "watering intervals",

        "irrigation frequency",
        "watering frequency",

        "irrigation schedule",
        "watering schedule",
        "schedule irrigation",

        "when should i irrigate",
        "when should i water",

        "when to irrigate",
        "when to water",

        "how often should i irrigate",
        "how often should i water",

        "how frequently should i irrigate",
        "how frequently should i water",

        "how many days",
        "interval in days",

        "days between irrigation",
        "days between watering",

        "days between irrigations",
        "watering frequency",

        "irrigation timing",
        "watering timing",

        "irrigation duration",
        "how long should i irrigate"
    ],


    # --------------------------------------------------------
    # Irrigation method
    # --------------------------------------------------------

    "Irrigation Method": [

        "irrigation method",
        "which irrigation method",
        "what irrigation method",

        "which method should i use",
        "what method should i use",

        "which irrigation system",
        "what irrigation system",

        "best irrigation method",
        "best irrigation system",

        "should i use drip",
        "should i use sprinkler",
        "should i use flood",

        "drip or sprinkler",
        "drip vs sprinkler",

        "flood or drip",
        "flood vs drip",

        "which method"
    ],


    # --------------------------------------------------------
    # Water saving
    # --------------------------------------------------------

    "Water Saving": [

        "save water",
        "saving water",
        "water saving",
        "water-saving",

        "save irrigation water",
        "reduce water use",
        "reduce water usage",

        "conserve water",
        "water conservation",

        "water efficiency",
        "water use efficiency",

        "improve water use efficiency",
        "improve irrigation efficiency",

        "reduce irrigation water",

        "use less water",
        "waste less water"
    ],


    # --------------------------------------------------------
    # Crop water management
    # --------------------------------------------------------

    "Crop Water Management": [

        "crop water management",
        "water management",

        "manage water",
        "manage irrigation",

        "irrigation management",

        "water requirement",
        "crop water requirement",

        "water needs",
        "crop water needs",

        "how much water",
        "water requirement of",

        "water needs of",
        "irrigation needs"
    ],


    # --------------------------------------------------------
    # Irrigation precautions
    # --------------------------------------------------------

    "Irrigation Precautions": [

        "what should i avoid",
        "what should i not do",

        "what to avoid",
        "things to avoid",

        "precautions",
        "what precautions",

        "avoid overwatering",
        "overwatering",
        "over watering",

        "avoid flooding",
        "avoid overhead watering",

        "disease risk",
        "evaporation"
    ],


    # --------------------------------------------------------
    # Micro irrigation
    # --------------------------------------------------------

    "Micro Irrigation": [

        "micro irrigation",
        "micro-irrigation",

        "micro irrigation system",
        "micro-irrigation system",

        "drip and sprinkler",
        "micro irrigation technology"
    ]
}


# ============================================================
# GENERIC FIND MATCH
# ============================================================

def find_match(
    text,
    dictionary
):
    """
    Find the best matching category.

    Longer keywords are checked first so that
    specific phrases receive priority.
    """

    text = text.lower().strip()


    # Longest phrases first
    all_items = []


    for category, keywords in dictionary.items():

        for keyword in keywords:

            all_items.append(
                (
                    keyword,
                    category
                )
            )


    all_items.sort(
        key=lambda item: len(item[0]),
        reverse=True
    )


    for keyword, category in all_items:

        if keyword in text:

            return category


    return None


# ============================================================
# DETECT CONTEXT
# ============================================================

def detect_context(
    question
):
    """
    Automatically extract structured context
    from the farmer's question.
    """

    question = question.strip()


    context = {

        "crop": find_match(
            question,
            CROPS
        ),

        "growth_stage": find_match(
            question,
            GROWTH_STAGES
        ),

        "irrigation_method": find_match(
            question,
            IRRIGATION_METHODS
        ),

        "water_availability": find_match(
            question,
            WATER_LEVELS
        ),

        "intent": find_match(
            question,
            INTENTS
        )
    }


    return context


# ============================================================
# MERGE SIDEBAR + QUESTION CONTEXT
# ============================================================

def build_context(
    question,
    sidebar_context=None
):
    """
    Combine question-detected context with
    sidebar context.

    Question information takes priority.
    """

    if sidebar_context is None:

        sidebar_context = {}


    detected = detect_context(
        question
    )


    final_context = {}


    # ========================================================
    # CROP
    # ========================================================

    if detected.get("crop"):

        final_context["crop"] = detected[
            "crop"
        ]

    elif (
        sidebar_context.get("crop")
        and sidebar_context.get("crop")
        != "Not specified"
    ):

        final_context["crop"] = (
            sidebar_context["crop"]
        )

    else:

        final_context["crop"] = None


    # ========================================================
    # GROWTH STAGE
    # ========================================================

    if detected.get("growth_stage"):

        final_context["growth_stage"] = (
            detected["growth_stage"]
        )

    elif (
        sidebar_context.get("growth_stage")
        and sidebar_context.get("growth_stage")
        != "Not specified"
    ):

        final_context["growth_stage"] = (
            sidebar_context["growth_stage"]
        )

    else:

        final_context["growth_stage"] = None


    # ========================================================
    # IRRIGATION METHOD
    # ========================================================

    if detected.get("irrigation_method"):

        final_context["irrigation_method"] = (
            detected["irrigation_method"]
        )

    elif (
        sidebar_context.get(
            "irrigation_method"
        )
        and sidebar_context.get(
            "irrigation_method"
        ) != "Not specified"
    ):

        final_context["irrigation_method"] = (
            sidebar_context[
                "irrigation_method"
            ]
        )

    else:

        final_context["irrigation_method"] = None


    # ========================================================
    # WATER AVAILABILITY
    # ========================================================

    if detected.get("water_availability"):

        final_context[
            "water_availability"
        ] = detected[
            "water_availability"
        ]

    elif (
        sidebar_context.get(
            "water_availability"
        )
        and sidebar_context.get(
            "water_availability"
        ) != "Not specified"
    ):

        final_context[
            "water_availability"
        ] = sidebar_context[
            "water_availability"
        ]

    else:

        final_context[
            "water_availability"
        ] = None


    # ========================================================
    # INTENT
    # ========================================================

    final_context["intent"] = detected.get(
        "intent"
    )


    return final_context


# ============================================================
# BUILD CONTEXT-AWARE RETRIEVAL QUERY
# ============================================================

def build_context_query(
    question,
    sidebar_context=None
):
    """
    Create the final query used by the retrieval system.
    """

    final_context = build_context(
        question,
        sidebar_context
    )


    query_parts = []


    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    if final_context.get("crop"):

        query_parts.append(
            f"Crop: "
            f"{final_context['crop']}"
        )


    # --------------------------------------------------------
    # Growth stage
    # --------------------------------------------------------

    if final_context.get(
        "growth_stage"
    ):

        query_parts.append(
            f"Growth Stage: "
            f"{final_context['growth_stage']}"
        )


    # --------------------------------------------------------
    # Irrigation method
    # --------------------------------------------------------

    if final_context.get(
        "irrigation_method"
    ):

        query_parts.append(
            f"Irrigation Method: "
            f"{final_context['irrigation_method']}"
        )


    # --------------------------------------------------------
    # Water availability
    # --------------------------------------------------------

    if final_context.get(
        "water_availability"
    ):

        query_parts.append(
            f"Water Availability: "
            f"{final_context['water_availability']}"
        )


    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    if final_context.get(
        "intent"
    ):

        query_parts.append(
            f"Intent: "
            f"{final_context['intent']}"
        )


    # --------------------------------------------------------
    # Original farmer question
    # --------------------------------------------------------

    query_parts.append(
        f"Farmer Question: "
        f"{question}"
    )


    retrieval_query = "\n".join(
        query_parts
    )


    return (
        final_context,
        retrieval_query
    )


# ============================================================
# PRETTY PRINT
# ============================================================

def print_context(context):

    print(
        "\nDETECTED CONTEXT"
    )

    print(
        "-" * 50
    )

    print(
        f"Crop               : "
        f"{context.get('crop') or 'Not specified'}"
    )

    print(
        f"Growth Stage       : "
        f"{context.get('growth_stage') or 'Not specified'}"
    )

    print(
        f"Irrigation Method  : "
        f"{context.get('irrigation_method') or 'Not specified'}"
    )

    print(
        f"Water Availability : "
        f"{context.get('water_availability') or 'Not specified'}"
    )

    print(
        f"Intent             : "
        f"{context.get('intent') or 'General'}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [

        (
            "What is the exact irrigation interval "
            "in days for tomato during flowering?"
        ),

        (
            "My tomato crop is flowering and water "
            "availability is low. What irrigation "
            "method should I use?"
        ),

        (
            "How can I save water in my rice field?"
        ),

        (
            "Should I use drip or sprinkler for vegetables?"
        ),

        (
            "My rice is in grain filling and water "
            "is limited. What should I do?"
        ),

        (
            "Tomato is flowering and water availability "
            "is severely limited."
        ),

        (
            "My rice crop is in the vegetative stage "
            "and I have moderate water availability."
        ),

        (
            "My wheat crop is at maturity and water "
            "is abundant."
        ),

        (
            "Should I avoid overhead watering for my "
            "tomato crop during flowering?"
        )
    ]


    for question in test_questions:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "QUESTION:"
        )

        print(
            question
        )


        context = detect_context(
            question
        )


        print_context(
            context
        )


        _, retrieval_query = (
            build_context_query(
                question
            )
        )


        print(
            "\nCONTEXT-AWARE RETRIEVAL QUERY:"
        )

        print(
            retrieval_query
        )