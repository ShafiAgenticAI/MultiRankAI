import re


PRODUCT_SECTIONS = {
    "home_loan": "SECTION 1: HOME LOAN PRODUCTS",
    "fixed_deposit": "SECTION 2: FIXED DEPOSIT PRODUCTS",
    "credit_card": "SECTION 3: CREDIT CARD PRODUCTS",
    "personal_loan": "SECTION 4: PERSONAL LOAN PRODUCTS",
}


PRODUCT_KEYWORDS = {
    "home_loan": [
        "home loan",
        "home loans",
        "housing loan",
        "housing loans",
        "mortgage",
    ],
    "fixed_deposit": [
        "fixed deposit",
        "fixed deposits",
        "fd",
    ],
    "credit_card": [
        "credit card",
        "credit cards",
    ],
    "personal_loan": [
        "personal loan",
        "personal loans",
    ],
}


def detect_product(query: str) -> str | None:
    """
    Detect the banking product mentioned in the query.

    Returns:
        home_loan
        fixed_deposit
        credit_card
        personal_loan
        None
    """

    if not query or not query.strip():
        return None

    normalized = re.sub(
        r"\s+",
        " ",
        query.lower().strip(),
    )

    for product, keywords in PRODUCT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in normalized:
                return product

    return None


def get_product_section(query: str) -> str | None:
    """
    Return the document section associated with
    the product mentioned in the query.
    """

    product = detect_product(query)

    if product is None:
        return None

    return PRODUCT_SECTIONS.get(product)