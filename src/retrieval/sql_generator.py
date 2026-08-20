from __future__ import annotations

import re


# ============================================================
# SQL GENERATOR
# ============================================================

def generate_sql(
    question: str,
) -> str:
    """
    Generate a read-only SQL query for the Smart Banking
    PostgreSQL database.

    The generated SQL uses only columns that actually exist
    in the application database schema.

    Supported areas:

    - Accounts
    - Transactions
    - Loans
    - Fixed deposits
    - Credit cards
    - Credit card transactions
    """

    if not question or not question.strip():
        return "UNSUPPORTED"

    query = question.strip()

    normalized = re.sub(
        r"\s+",
        " ",
        query.lower(),
    ).strip()

    # ========================================================
    # TRANSACTIONS
    # ========================================================

    if (
        "transaction" in normalized
        or "transactions" in normalized
    ):

        # ----------------------------------------------------
        # Amount filter
        # ----------------------------------------------------

        amount_match = re.search(
            r"(?:above|over|greater than|more than)\s*"
            r"(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)",
            normalized,
        )

        if amount_match:

            amount = (
                amount_match.group(1)
                .replace(",", "")
            )

            return f"""
SELECT
    account_id,
    txn_date,
    txn_type,
    amount,
    merchant_name,
    category
FROM transactions
WHERE amount > {amount}
ORDER BY txn_date DESC
LIMIT 100
""".strip()

        # ----------------------------------------------------
        # Default transaction query
        # ----------------------------------------------------

        return """
SELECT
    account_id,
    txn_date,
    txn_type,
    amount,
    merchant_name,
    category
FROM transactions
ORDER BY txn_date DESC
LIMIT 100
""".strip()

    # ========================================================
    # CREDIT CARD TRANSACTIONS
    # ========================================================

    if (
        "credit card transaction" in normalized
        or "credit card transactions" in normalized
        or "card transaction" in normalized
        or "card transactions" in normalized
    ):

        return """
SELECT
    card_id,
    txn_date,
    txn_type,
    amount,
    merchant_name,
    category,
    is_international,
    currency
FROM card_transactions
ORDER BY txn_date DESC
LIMIT 100
""".strip()

    # ========================================================
    # CREDIT CARD OUTSTANDING
    # ========================================================

    if (
        "credit card" in normalized
        and (
            "outstanding" in normalized
            or "balance" in normalized
            or "due" in normalized
        )
    ):

        return """
SELECT
    card_id,
    account_id,
    card_variant,
    outstanding_amt
FROM credit_cards
LIMIT 100
""".strip()

    # ========================================================
    # CREDIT CARDS
    # ========================================================

    if (
        "credit card" in normalized
        or "credit cards" in normalized
    ):

        return """
SELECT
    card_id,
    account_id,
    card_variant,
    credit_limit,
    available_limit,
    outstanding_amt,
    due_date,
    min_due
FROM credit_cards
LIMIT 100
""".strip()

    # ========================================================
    # LOANS
    # ========================================================

    if (
        "loan" in normalized
        or "emi" in normalized
        or "outstanding loan" in normalized
    ):

        # ----------------------------------------------------
        # Customer-specific loan query
        # ----------------------------------------------------

        if (
            "my " in normalized
            or "my loan" in normalized
            or "my loans" in normalized
        ):

            return """
SELECT
    loan_id,
    loan_type,
    principal,
    outstanding,
    emi_amount,
    next_emi_date,
    interest_rate,
    tenure_months,
    emi_paid,
    status
FROM loan_accounts
WHERE account_id = :account_id
LIMIT 100
""".strip()

        # ----------------------------------------------------
        # General loan query
        # ----------------------------------------------------

        return """
SELECT
    loan_id,
    account_id,
    loan_type,
    principal,
    outstanding,
    emi_amount,
    next_emi_date,
    interest_rate,
    tenure_months,
    emi_paid,
    status
FROM loan_accounts
LIMIT 100
""".strip()

    # ========================================================
    # FIXED DEPOSITS
    # ========================================================

    if (
        "fixed deposit" in normalized
        or "fixed deposits" in normalized
        or re.search(r"\bfd\b", normalized)
    ):

        # ----------------------------------------------------
        # Customer-specific active FD
        # ----------------------------------------------------

        if (
            "my " in normalized
            or "my fixed deposit" in normalized
            or "my fixed deposits" in normalized
        ):

            if "active" in normalized:

                return """
SELECT
    fd_id,
    principal,
    interest_rate,
    tenure_days,
    start_date,
    maturity_date,
    maturity_amount,
    interest_payout,
    status
FROM fixed_deposits
WHERE account_id = :account_id
  AND status = 'active'
LIMIT 100
""".strip()

            return """
SELECT
    fd_id,
    principal,
    interest_rate,
    tenure_days,
    start_date,
    maturity_date,
    maturity_amount,
    interest_payout,
    status
FROM fixed_deposits
WHERE account_id = :account_id
LIMIT 100
""".strip()

        # ----------------------------------------------------
        # General active FD
        # ----------------------------------------------------

        if "active" in normalized:

            return """
SELECT
    fd_id,
    account_id,
    principal,
    interest_rate,
    tenure_days,
    start_date,
    maturity_date,
    maturity_amount,
    interest_payout,
    status
FROM fixed_deposits
WHERE status = 'active'
LIMIT 100
""".strip()

        # ----------------------------------------------------
        # General FD query
        # ----------------------------------------------------

        return """
SELECT
    fd_id,
    account_id,
    principal,
    interest_rate,
    tenure_days,
    start_date,
    maturity_date,
    maturity_amount,
    interest_payout,
    status
FROM fixed_deposits
LIMIT 100
""".strip()

    # ========================================================
    # ACCOUNTS
    # ========================================================

    if (
        "account" in normalized
        or "accounts" in normalized
        or "customer details" in normalized
        or "customer information" in normalized
    ):

        return """
SELECT
    account_id,
    customer_name,
    account_type,
    branch_code,
    ifsc_code,
    mobile,
    email,
    kyc_status,
    created_at
FROM accounts
LIMIT 100
""".strip()

    # ========================================================
    # UNSUPPORTED
    # ========================================================

    return "UNSUPPORTED"