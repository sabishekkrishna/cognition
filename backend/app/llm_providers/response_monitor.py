import re
import time
from typing import List, Dict, Any, Tuple, Optional
from ..storage.database import get_db

UNCERTAINTY_PHRASES = [
    r"i am not (certain|sure)",
    r"(unclear|blurry|fuzzy|distorted)",
    r"cannot (clearly )?determine",
    r"insufficient (detail|information|quality)",
    r"difficult to (see|hear|tell)",
    r"appears to be .* but (might|could)",
    r"low confidence",
    r"quality is too (low|poor)",
    r"not enough context"
]

class ResponseMonitor:
    """
    Monitors LLM outputs and contextual input-output reliability.
    Distinguishes input-quality hazards from output-uncertainty signals.
    """

    @classmethod
    def evaluate_response(cls,
                          prompt: str,
                          response_text: str,
                          input_risk_score: float = 0.0,
                          input_subsystem: Optional[str] = None) -> Tuple[float, List[str], bool, bool]:
        """
        Evaluates an LLM response.
        Returns: (response_risk_score, warnings, is_uncertain, is_empty)
        """
        warnings: List[str] = []
        is_empty = False
        is_uncertain = False
        risk_score = 0.0

        # 1. Empty or trivial response check
        cleaned = response_text.strip()
        if len(cleaned) == 0:
            is_empty = True
            risk_score += 90.0
            warnings.append("Output warning: Model generated an empty response or terminated prematurely.")
        elif len(cleaned) < 15 and not any(char.isalnum() for char in cleaned):
            is_empty = True
            risk_score += 80.0
            warnings.append("Output warning: Model generated degenerate non-alphanumeric text.")

        # 2. Input quality coupling check
        if input_risk_score > 40.0:
            risk_score += (input_risk_score * 0.4)
            warnings.append(
                f"Input hazard warning: Upstream {input_subsystem or 'sensor'} input exhibited high risk ({input_risk_score:.1f}/100). "
                "Downstream response may be based on degraded or corrupted perceptions."
            )

        # 3. Explicit uncertainty phrase detection in generated text
        lower_resp = response_text.lower()
        matched_phrases = []
        for pattern in UNCERTAINTY_PHRASES:
            match = re.search(pattern, lower_resp)
            if match:
                matched_phrases.append(match.group(0))

        if matched_phrases:
            is_uncertain = True
            risk_score += 35.0
            warnings.append(
                f"Uncertainty detected: Model explicitly expressed doubt ('{matched_phrases[0]}'). "
                "The AI indicates it lacks adequate fidelity or information to answer definitively."
            )

        # 4. Truncation check
        if cleaned and not cleaned[-1] in [".", "!", "?", '"', "'", "}", "]", ">", "`"]:
            if len(cleaned) > 200:
                warnings.append("Output notice: Response appears abruptly truncated without sentence completion.")
                risk_score += 15.0

        clamped_risk = max(0.0, min(100.0, risk_score))
        return float(round(clamped_risk, 1)), warnings, is_uncertain, is_empty

    @classmethod
    def record_interaction(cls,
                           provider: str,
                           model: str,
                           prompt: str,
                           response: str,
                           input_subsystem: Optional[str],
                           input_risk_score: float,
                           response_risk_score: float,
                           warnings: List[str]) -> int:
        """Saves interaction record to SQLite database."""
        conn = get_db()
        cursor = conn.cursor()
        import json
        cursor.execute("""
        INSERT INTO llm_interactions (
            timestamp, provider, model, prompt, response,
            input_subsystem, input_risk_score, response_risk_score,
            warnings_json, user_rating, user_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """, (
            time.time(),
            provider,
            model,
            prompt,
            response,
            input_subsystem,
            input_risk_score,
            response_risk_score,
            json.dumps(warnings)
        ))
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return interaction_id

    @classmethod
    def record_feedback(cls, interaction_id: int, rating: str, notes: Optional[str] = None):
        """Records human feedback (correct, incorrect, uncertain) for local evaluation."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE llm_interactions
        SET user_rating = ?, user_notes = ?
        WHERE id = ?
        """, (rating, notes, interaction_id))
        conn.commit()
        conn.close()
