import re
import time
import json
from typing import List, Dict, Any, Tuple, Optional
from ..storage.database import get_db, add_event

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

COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "what",
    "which", "who", "whom", "how", "when", "where", "why", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "please", "tell", "say", "show", "give"
}

class ResponseMonitor:
    """
    Monitors LLM outputs, contextual input-output reliability, relevance, and factual consistency.
    Distinguishes input-quality hazards from output-uncertainty signals.
    """

    @classmethod
    def calculate_relevance(cls, prompt: str, response_text: str) -> Tuple[float, List[str]]:
        """
        Calculates lexical and topical relevance between prompt and generated response.
        Returns (relevance_score, warnings).
        Relevance score is between 0.0 and 100.0.
        """
        warnings: List[str] = []
        p_clean = prompt.strip().lower()
        r_clean = response_text.strip().lower()

        if not p_clean or not r_clean:
            return 0.0, ["Relevance alert: Empty prompt or response text."]

        # Check for exact target directives (e.g., 'Say OK', 'Respond with READY')
        m_say = re.search(r"(say|respond with|reply with|output)\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", p_clean)
        if m_say:
            target_token = m_say.group(2).lower()
            if target_token in r_clean:
                return 100.0, []

        # Check greetings
        if p_clean in ["hi", "hello", "hey", "greetings"] and any(w in r_clean for w in ["hi", "hello", "hey", "assist", "help"]):
            return 100.0, []

        # Tokenize content words
        p_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", p_clean)) - COMMON_STOPWORDS
        r_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", r_clean)) - COMMON_STOPWORDS

        if not p_words:
            # Short generic prompt without stopwords
            return 85.0, []

        # Overlap fraction
        overlap = p_words.intersection(r_words)
        coverage = len(overlap) / len(p_words)
        relevance_score = round(min(100.0, max(10.0, coverage * 100.0)), 1)

        # If very low overlap on substantial prompt
        if relevance_score < 20.0 and len(p_words) >= 3:
            warnings.append(
                f"Low relevance ({relevance_score:.0f}%): Output shares few keywords with the prompt. "
                "The response may be off-topic or tangential to the user's intent."
            )

        return relevance_score, warnings

    @classmethod
    def check_factual_consistency(cls, prompt: str, response_text: str) -> Tuple[Optional[float], str, List[str]]:
        """
        Checks factual consistency against available local real-world evidence sources.
        When real-world evidence is unavailable, explicitly reports that limitation without fabricating results.
        Returns: (factual_score, evidence_status, warnings)
        """
        p_lower = prompt.lower()
        r_lower = response_text.lower()
        warnings: List[str] = []

        # Check for queries about local baselines or datasets
        is_about_baselines = any(term in p_lower for term in ["baseline", "calibration", "threshold"])
        is_about_datasets = any(term in p_lower for term in ["dataset", "training data", "samples"])

        if is_about_baselines or is_about_datasets:
            # Query local ground-truth evidence
            try:
                from ..datasets.manager import DatasetManager
                datasets = DatasetManager.list_datasets()
                actual_count = len(datasets)
                evidence_found = True
            except Exception:
                actual_count = 0
                evidence_found = False

            if evidence_found and is_about_datasets:
                # Compare claims
                if str(actual_count) in r_lower or f"{actual_count} dataset" in r_lower:
                    return 100.0, "verified: Corroborated by local dataset catalog.", []
                else:
                    warnings.append(
                        f"Factual check: Response does not mention the actual registered dataset count ({actual_count})."
                    )
                    return 50.0, "unconfirmed: Partial local dataset evidence available.", warnings

        # Open-domain / General knowledge where no local ground-truth files exist
        return (
            None,
            "unavailable: Real-world ground truth evidence is not present in local datasets for independent factual verification.",
            []
        )

    @classmethod
    def detect_hallucinations(cls, prompt: str, response_text: str, input_risk_score: float) -> Tuple[bool, List[str]]:
        """
        Detects potential hallucination signals or unsupported assertions.
        """
        warnings: List[str] = []
        is_hallucination = False
        r_lower = response_text.lower()

        # Overconfidence signal when upstream sensor input was degraded
        if input_risk_score > 40.0:
            overconfident_phrases = ["definitely", "absolutely certain", "100% sure", "there is no doubt", "without question"]
            if any(p in r_lower for p in overconfident_phrases):
                is_hallucination = True
                warnings.append(
                    "Potential hallucination / overconfidence: Model expresses absolute certainty despite degraded upstream input quality."
                )

        # Repetitive loop detection (repeating 3+ word n-gram multiple times)
        tokens = r_lower.split()
        if len(tokens) >= 12:
            trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]
            for tri, count in collections_counter(trigrams).items():
                if count >= 3 and len(tri) > 8:
                    is_hallucination = True
                    warnings.append(f"Degenerative loop: Model repeated phrase '{tri}' {count} times.")
                    break

        return is_hallucination, warnings

    @classmethod
    def evaluate_response_detailed(cls,
                                   prompt: str,
                                   response_text: str,
                                   input_risk_score: float = 0.0,
                                   input_subsystem: Optional[str] = None,
                                   latency_ms: Optional[float] = None,
                                   model: Optional[str] = None) -> Dict[str, Any]:
        """
        Full multidimensional evaluation of an LLM request/response pair:
        - Output risk score (0-100)
        - Input/Output relevance
        - Explicit uncertainty
        - Factual consistency against local evidence
        - Potential hallucinations / overconfidence
        - Latency and operational metrics
        """
        all_warnings: List[str] = []
        risk_score = 0.0

        # 1. Empty or trivial response check
        cleaned = (response_text or "").strip()
        is_empty = False
        if len(cleaned) == 0:
            is_empty = True
            risk_score += 90.0
            all_warnings.append("Output warning: Model generated an empty response or terminated prematurely.")
        elif len(cleaned) < 15 and not any(char.isalnum() for char in cleaned):
            is_empty = True
            risk_score += 80.0
            all_warnings.append("Output warning: Model generated degenerate non-alphanumeric text.")

        # 2. Input hazard coupling check
        if input_risk_score > 40.0:
            risk_score += (input_risk_score * 0.4)
            all_warnings.append(
                f"Input hazard warning: Upstream {input_subsystem or 'sensor'} input exhibited high risk ({input_risk_score:.1f}/100). "
                "Downstream response may be based on degraded or corrupted perceptions."
            )

        # 3. Explicit uncertainty phrase detection
        is_uncertain = False
        lower_resp = cleaned.lower()
        matched_phrases = []
        for pattern in UNCERTAINTY_PHRASES:
            match = re.search(pattern, lower_resp)
            if match:
                matched_phrases.append(match.group(0))

        if matched_phrases:
            is_uncertain = True
            risk_score += 35.0
            all_warnings.append(
                f"Uncertainty detected: Model explicitly expressed doubt ('{matched_phrases[0]}'). "
                "The AI indicates it lacks adequate fidelity or information to answer definitively."
            )

        # 4. Truncation check
        is_truncated = False
        if cleaned and not cleaned[-1] in [".", "!", "?", '"', "'", "}", "]", ">", "`"]:
            if len(cleaned) > 200:
                is_truncated = True
                all_warnings.append("Output notice: Response appears abruptly truncated without sentence completion.")
                risk_score += 15.0

        # 5. Input and Output Relevance
        relevance_score, rel_warnings = cls.calculate_relevance(prompt, response_text)
        all_warnings.extend(rel_warnings)
        if relevance_score < 25.0 and not is_empty:
            risk_score += 15.0

        # 6. Factual consistency against local evidence
        factual_score, evidence_status, fact_warnings = cls.check_factual_consistency(prompt, response_text)
        all_warnings.extend(fact_warnings)
        if factual_score is not None and factual_score < 50.0:
            risk_score += 30.0

        # 7. Potential Hallucinations
        is_hallucinating, hal_warnings = cls.detect_hallucinations(prompt, response_text, input_risk_score)
        all_warnings.extend(hal_warnings)
        if is_hallucinating:
            risk_score += 25.0

        clamped_risk = float(round(max(0.0, min(100.0, risk_score)), 1))
        status_label = "warning" if clamped_risk > 50 else ("caution" if clamped_risk > 30 else "normal")

        return {
            "risk_score": clamped_risk,
            "response_risk_score": clamped_risk,
            "status": status_label,
            "warnings": all_warnings,
            "is_uncertain": is_uncertain,
            "is_empty": is_empty,
            "is_truncated": is_truncated,
            "relevance_score": relevance_score,
            "factual_consistency_score": factual_score,
            "evidence_status": evidence_status,
            "latency_ms": latency_ms or 0.0,
            "model": model or "unknown",
            "summary": {
                "relevance": relevance_score,
                "factual_evidence": evidence_status,
                "uncertainty_detected": is_uncertain,
                "hallucination_risk": is_hallucinating,
                "latency_ms": latency_ms or 0.0
            }
        }

    @classmethod
    def evaluate_response(cls,
                          prompt: str,
                          response_text: str,
                          input_risk_score: float = 0.0,
                          input_subsystem: Optional[str] = None) -> Tuple[float, List[str], bool, bool]:
        """
        Backwards-compatible interface returning:
        (response_risk_score, warnings, is_uncertain, is_empty)
        """
        detailed = cls.evaluate_response_detailed(
            prompt=prompt,
            response_text=response_text,
            input_risk_score=input_risk_score,
            input_subsystem=input_subsystem
        )
        return (
            detailed["risk_score"],
            detailed["warnings"],
            detailed["is_uncertain"],
            detailed["is_empty"]
        )

    @classmethod
    def record_interaction(cls,
                           provider: str,
                           model: str,
                           prompt: str,
                           response: str,
                           input_subsystem: Optional[str] = None,
                           input_risk_score: float = 0.0,
                           response_risk_score: float = 0.0,
                           warnings: Optional[List[str]] = None,
                           latency_ms: Optional[float] = None,
                           relevance_score: Optional[float] = None,
                           evidence_status: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> int:
        """Saves interaction record to SQLite database and updates live dashboard status."""
        conn = get_db()
        cursor = conn.cursor()
        warnings_list = warnings or []

        meta_dict = metadata or {}
        if latency_ms is not None:
            meta_dict["latency_ms"] = latency_ms
        if relevance_score is not None:
            meta_dict["relevance_score"] = relevance_score
        if evidence_status is not None:
            meta_dict["evidence_status"] = evidence_status

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
            json.dumps(warnings_list)
        ))
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Update Unified Monitoring Engine live status
        try:
            from ..monitoring.engine import get_monitoring_engine
            engine = get_monitoring_engine()
            state = "warning" if response_risk_score > 50 else ("caution" if response_risk_score > 30 else "normal")
            engine.set_llm_status(state, round(100.0 - response_risk_score, 1))

            if response_risk_score > 30 and warnings_list:
                add_event(
                    subsystem="llm",
                    severity=state,
                    title=f"LLM Reliability Risk ({response_risk_score:.0f}/100)",
                    message=warnings_list[0],
                    metric_name="response_risk",
                    measured_value=response_risk_score,
                    recommendation="Review generated response before utilizing in production decisions.",
                    metadata=meta_dict
                )
        except Exception:
            pass

        return interaction_id

    @classmethod
    def list_interactions(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent interactions from database for audit and dashboard monitoring."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, timestamp, provider, model, prompt, response,
               input_subsystem, input_risk_score, response_risk_score,
               warnings_json, user_rating, user_notes
        FROM llm_interactions
        ORDER BY timestamp DESC
        LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            w = []
            try:
                w = json.loads(r["warnings_json"]) if r["warnings_json"] else []
            except Exception:
                pass
            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "provider": r["provider"],
                "model": r["model"],
                "prompt": r["prompt"],
                "response": r["response"],
                "input_subsystem": r["input_subsystem"],
                "input_risk_score": r["input_risk_score"],
                "response_risk_score": r["response_risk_score"],
                "warnings": w,
                "user_rating": r["user_rating"],
                "user_notes": r["user_notes"]
            })
        return results

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

def collections_counter(iterable):
    counts = {}
    for item in iterable:
        counts[item] = counts.get(item, 0) + 1
    return counts
