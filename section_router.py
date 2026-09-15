# =========================================================
# SECTION ROUTER
# =========================================================

SECTION_KEYWORDS = {

    "3. Coverage of Farmers": [
        "who is covered",
        "who can apply",
        "eligible farmer",
        "eligible farmers",
        "farmer eligibility",
        "loanee farmer",
        "non-loanee farmer",
        "tenant farmer",
        "tenant farmers",
        "sharecropper",
        "sharecroppers",
        "coverage of farmers"
    ],

    "4. Coverage of Crops": [
        "what crops",
        "which crops",
        "crops covered",
        "crop covered",
        "eligible crops",
        "coverage of crops",
        "food crops",
        "oilseeds",
        "horticultural crops"
    ],

    "5. Coverage of Risks and Exclusions": [
        "what risks",
        "which risks",
        "risks covered",
        "risk covered",
        "crop loss risks",
        "perils",
        "prevented sowing",
        "standing crop",
        "post harvest",
        "post-harvest",
        "localized calamity",
        "localized calamities",
        "wild animals",
        "exclusions",
        "risks and exclusions"
    ]
}


def route_query(query):

    query = query.lower().strip()

    section_scores = {}

    for section, keywords in SECTION_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in query:
                score += 1

        section_scores[section] = score

    best_section = None
    best_score = 0

    for section, score in section_scores.items():

        if score > best_score:

            best_section = section
            best_score = score

    return best_section, best_score


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_questions = [
        "Who is covered under PMFBY?",
        "What crops are covered?",
        "What risks are covered under PMFBY?",
        "What are the exclusions?"
    ]

    for question in test_questions:

        section, score = route_query(question)

        print("\nQuestion :", question)
        print("Section  :", section)
        print("Score    :", score)