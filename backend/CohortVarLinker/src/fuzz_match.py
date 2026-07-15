
from typing import list, set, tuple, Optional
import re
from .config import settings
from .utils import extract_visit_period, is_interval_period, is_determinate_period

_nlp = None
class FuzzyMatcher:



    @staticmethod
    def check_visit_string(visit_str_1: str, visit_str_2: str) -> bool:
        """Return True only when temporal contexts are comparable."""
        s_low = extract_visit_period(visit_str_1)
        t_low = extract_visit_period(visit_str_2)

        # Reject interval-vs-point comparisons.
        # Example: "between baseline and visit 1" vs "baseline"
        if is_interval_period(s_low) != is_interval_period(t_low):
            return False

        # If both are intervals, accept only the same interval.
        # Example: "baseline to visit 1" vs "baseline to visit 1"
        if is_interval_period(s_low) and is_interval_period(t_low):
            return s_low == t_low

        for hint in settings.DATE_HINTS:
            if hint in s_low and hint in t_low:
                return s_low == t_low
            elif hint in s_low:
                return True
            elif hint in t_low:
                return True

        return s_low == t_low

    @staticmethod
    def visits_compatible(visit_str_1: str, visit_str_2: str) -> bool:
        """Retrieval-stage visit gate.

        A concept-matched candidate is rejected on temporal grounds only when
        both visit labels resolve to a determinate, discrete period and those
        periods disagree. When either side does not resolve to a discrete
        timepoint (undetermined, a date field, or a study-period axis such as
        'study days'), the visit supplies no evidence of disagreement and does
        not block retrieval. The determinate branch reproduces the
        interval-versus-point and same-period logic of check_visit_string, so
        genuine repeated-measures mismatches (e.g. baseline vs month 6) are
        still rejected.
        """
        s = extract_visit_period(visit_str_1)
        t = extract_visit_period(visit_str_2)

        if is_determinate_period(visit_str_1) and is_determinate_period(visit_str_2):
            # Both sides denote a discrete or interval timepoint: apply the
            # strict comparison.
            if is_interval_period(s) != is_interval_period(t):
                return False
            return s == t

        # At least one side is non-discrete: not a temporal constraint.
        return True
    @staticmethod
    def tokenize(text):
            # Split on non-alphanumeric chars (keep only words)
            return set(re.split(r'[^a-zA-Z0-9]+', text.lower()))
    
    @staticmethod
    def _has_token_overlap(label1: str, label2: str) -> bool:
       return (FuzzyMatcher.tokenize(label2).issubset(FuzzyMatcher.tokenize(label1)) or FuzzyMatcher.tokenize(label1).issubset(FuzzyMatcher.tokenize(label2)))
   
    @staticmethod
    def _is_negation_pair(cat1: str, cat2: str) -> tuple[bool, Optional[str], Optional[str]]:
        """Check if two categories form a positive/negative pair."""
        
        neg1 = FuzzyMatcher.has_negation(cat1)
        neg2 = FuzzyMatcher.has_negation(cat2)
        
        if neg1 and not neg2:
            return True, cat1, cat2
        elif neg2 and not neg1:
            return True, cat2, cat1
        
        return False, None, None
    @staticmethod
    def _get_nlp():
            global _nlp
            if _nlp is None:
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            return _nlp
    @staticmethod
    def has_negation(text: str) -> bool:
            """Check if text expresses negation."""
            text_lower = text.lower().strip()
            if text_lower in {'no', 'none', 'absent', 'negative', 'false', '0'} or text_lower.startswith(('no', 'not ', 'non-')):
                return True
            doc = FuzzyMatcher._get_nlp()(text_lower)
            return any(token.dep_ == 'neg' for token in doc)
    
    @staticmethod
    def get_lemmas(text: str) -> set[str]:
        """Extract meaningful lemmas using spaCy's built-in filtering."""
        nlp = FuzzyMatcher._get_nlp()
        doc = nlp(text.lower())
        return {
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        and len(token.text) > 1
    }

    @staticmethod
    def _is_redundant(candidate: str, existing: list[str], threshold: float = 0.8) -> bool:
        cand_lemmas = FuzzyMatcher.get_lemmas(candidate)
        if not cand_lemmas:
            return True
        existing_lemmas = set()
        for part in existing:
            existing_lemmas.update(FuzzyMatcher.get_lemmas(part))
        if not existing_lemmas:
            return False
        
        overlap = cand_lemmas & existing_lemmas
        score = len(overlap) / len(cand_lemmas)
       

        return score >= threshold
        
    @staticmethod
    def _deduplicate_parts(parts: list[str], threshold: float = 0.8) -> list[str]:
        """Remove redundant parts, keeping first occurrence."""
        result = []
        for part in parts:
            if not part or not part.strip():
                continue
            if not result or not FuzzyMatcher._is_redundant(part.strip(), result, threshold):
                result.append(part.strip())
        return result
        