import logging
from typing import Tuple

logger = logging.getLogger(__name__)

DOMAIN_PATTERNS = [

    (".edu", "academic"),
    (".ac.uk", "academic"),
    (".ac.jp", "academic"),
    (".ac.kr", "academic"),
    (".ac.nz", "academic"),
    (".ac.za", "academic"),
    (".edu.au", "academic"),
    (".edu.cn", "academic"),
    (".edu.sg", "academic"),
    (".uni-", "academic"),
    ("university", "academic"),
    (".edu.", "academic"),

    (".gov", "government"),
    (".gov.", "government"),
    (".gob.", "government"),
    (".gouv.", "government"),
    (".mil", "government"),

    (".org", "nonprofit"),

    (".com", "commercial"),
    (".io", "commercial"),
    (".co", "commercial"),
    (".ai", "commercial"),
    (".tech", "commercial"),
    (".app", "commercial"),
    (".dev", "commercial"),
    (".inc", "commercial"),
]

KNOWN_DOMAINS = {

    "stanford.edu": "academic",
    "mit.edu": "academic",
    "harvard.edu": "academic",
    "ox.ac.uk": "academic",
    "cam.ac.uk": "academic",
    "ethz.ch": "academic",
    "epfl.ch": "academic",
    "nih.gov": "government",
    "cdc.gov": "government",

    "broad.org": "academic",
    "sanger.ac.uk": "academic",
    "ebi.ac.uk": "academic",

    "wellcome.org": "nonprofit",
    "gatesfoundation.org": "nonprofit",

    "mozilla.org": "nonprofit",
}

def infer_requester_type(
    email: str,
    declared_type: str | None = None
) -> Tuple[str, str]:
    if not email or "@" not in email:
        return ("unknown", "invalid_email")

    domain = email.split("@")[-1].lower().strip()

    if domain in KNOWN_DOMAINS:
        inferred = KNOWN_DOMAINS[domain]
        source = f"known_domain:{domain}"
    else:

        inferred = "unknown"
        source = f"email_heuristic:{domain}"

        for pattern, req_type in DOMAIN_PATTERNS:
            if pattern in domain:
                inferred = req_type
                break

    if declared_type:
        declared_lower = declared_type.lower()
        if declared_lower != inferred and inferred != "unknown":
            logger.warning(
                "Requester type mismatch",
                extra={
                    "email": email,
                    "declared": declared_lower,
                    "inferred": inferred,
                    "domain": domain
                }
            )
        return (declared_lower, f"declared:{declared_lower}")

    return (inferred, source)

def get_country_from_email_tld(email: str) -> str | None:
    if not email or "@" not in email:
        return None

    domain = email.split("@")[-1].lower()

    COUNTRY_TLDS = {
        ".uk": "GB",
        ".de": "DE",
        ".fr": "FR",
        ".it": "IT",
        ".es": "ES",
        ".nl": "NL",
        ".be": "BE",
        ".ch": "CH",
        ".at": "AT",
        ".au": "AU",
        ".nz": "NZ",
        ".ca": "CA",
        ".jp": "JP",
        ".kr": "KR",
        ".cn": "CN",
        ".sg": "SG",
        ".in": "IN",
        ".br": "BR",
        ".mx": "MX",
        ".za": "ZA",
        ".il": "IL",
        ".se": "SE",
        ".no": "NO",
        ".dk": "DK",
        ".fi": "FI",
        ".pl": "PL",
        ".cz": "CZ",
        ".ie": "IE",
        ".pt": "PT",
    }

    for tld, country in COUNTRY_TLDS.items():
        if domain.endswith(tld):
            return country

    if domain.endswith(".edu") or domain.endswith(".gov"):
        return "US"

    return None

def is_email_from_allowed_country(email: str, allowed_countries: list[str]) -> bool:
    country = get_country_from_email_tld(email)

    if country is None:

        return True

    return country in allowed_countries
