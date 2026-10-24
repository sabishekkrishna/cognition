import time
import collections
from typing import List, Dict, Any, Optional
from ..schemas.schemas import (
    ImageQualityMetrics, AudioQualityMetrics, ModelPredictions,
    MetricWarning, SystemOverallStatus
)
from ..storage.database import add_event, get_all_settings
from ..baselines.manager import BaselineManager
from .scoring import ReliabilityScoringEngine

class RollingWindowSmoother:
    """
    Maintains a rolling buffer of metric values, computing moving averages
    to eliminate single-frame glitches and transient spikes.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffers: Dict[str, collections.deque] = {}

    def update(self, key: str, value: float) -> float:
        if key not in self.buffers:
            self.buffers[key] = collections.deque(maxlen=self.window_size)
        self.buffers[key].append(value)
        return float(sum(self.buffers[key]) / len(self.buffers[key]))

    def get_smoothed_map(self, raw_dict: Dict[str, float]) -> Dict[str, float]:
        return {k: self.update(k, v) for k, v in raw_dict.items()}

    def clear(self):
        self.buffers.clear()

class SustainedDeviationDetector:
    """
    Hysteresis mechanism:
    Requires N consecutive abnormal samples before escalating to Caution/Warning.
    Requires M consecutive normal samples before de-escalating to Normal.
    Prevents alert flickering in live video/audio streams.
    """
    def __init__(self, escalate_threshold: int = 3, deescalate_threshold: int = 3):
        self.escalate_threshold = escalate_threshold
        self.deescalate_threshold = deescalate_threshold
        self.consecutive_abnormal = 0
        self.consecutive_normal = 0
        self.current_state = "normal" # 'normal', 'caution', 'warning'

    def update(self, sample_state: str) -> str:
        if sample_state in ["warning", "caution"]:
            self.consecutive_abnormal += 1
            self.consecutive_normal = 0
            if self.consecutive_abnormal >= self.escalate_threshold:
                self.current_state = sample_state
        else: # normal
            self.consecutive_normal += 1
            self.consecutive_abnormal = 0
            if self.consecutive_normal >= self.deescalate_threshold:
                self.current_state = "normal"

        return self.current_state

class UnifiedMonitoringEngine:
    """
    Central orchestration engine for real-time and on-demand reliability monitoring.
    Aggregates image, audio, and LLM subsystem signals.
    """
    def __init__(self):
        self.image_smoother = RollingWindowSmoother(window_size=5)
        self.audio_smoother = RollingWindowSmoother(window_size=5)
        self.image_hysteresis = SustainedDeviationDetector(escalate_threshold=3, deescalate_threshold=3)
        self.audio_hysteresis = SustainedDeviationDetector(escalate_threshold=3, deescalate_threshold=3)

        # Current live state cache
        self.latest_image_metrics: Optional[ImageQualityMetrics] = None
        self.latest_audio_metrics: Optional[AudioQualityMetrics] = None
        self.latest_image_score: float = 100.0
        self.latest_audio_score: float = 100.0
        self.latest_llm_score: float = 100.0
        self.image_state: str = "normal"
        self.audio_state: str = "normal"
        self.llm_state: str = "normal"
        self.active_warnings: List[MetricWarning] = []
        self._last_event_time = 0.0

    def process_image_frame(self,
                            raw_metrics: ImageQualityMetrics,
                            predictions: ModelPredictions) -> Dict[str, Any]:
        """Processes a live or uploaded image frame."""
        self.latest_image_metrics = raw_metrics
        baseline = BaselineManager.get_or_create_default_image_baseline()
        baseline_stats = baseline.get("stats", {}) if baseline else {}

        # 1. Evaluate image reliability & warnings
        rel_score, risk_score, raw_status, warnings, deviations = ReliabilityScoringEngine.evaluate_image(
            raw_metrics, predictions, baseline_stats
        )

        # 2. Smooth score and apply hysteresis for sustained deviations
        smoothed_score = self.image_smoother.update("image_score", rel_score)
        sustained_status = self.image_hysteresis.update(raw_status)

        self.latest_image_score = round(smoothed_score, 1)
        self.image_state = sustained_status

        # 3. If sustained warning occurs, emit and record event to DB (rate-limited to 1 event per 3s)
        now = time.time()
        if sustained_status in ["warning", "caution"] and (now - self._last_event_time > 3.0):
            primary_warning = warnings[0] if warnings else None
            if primary_warning:
                add_event(
                    subsystem="image",
                    severity=sustained_status,
                    title=f"Image {primary_warning.metric.capitalize()} Sustained Degradation",
                    message=primary_warning.explanation,
                    metric_name=primary_warning.metric,
                    measured_value=primary_warning.measured_value,
                    expected_range=primary_warning.expected_range,
                    recommendation=primary_warning.recommendation,
                    metadata={"reliability_score": self.latest_image_score, "raw_status": raw_status}
                )
                self._last_event_time = now

        # Update active warnings list
        self._refresh_active_warnings(image_warnings=warnings)

        return {
            "metrics": raw_metrics,
            "predictions": predictions,
            "reliability_score": self.latest_image_score,
            "risk_score": round(100.0 - self.latest_image_score, 1),
            "status": sustained_status,
            "raw_status": raw_status,
            "warnings": warnings,
            "deviations": deviations
        }

    def process_audio_chunk(self,
                            raw_metrics: AudioQualityMetrics,
                            classification: Any) -> Dict[str, Any]:
        """Processes a live or uploaded audio stream chunk."""
        self.latest_audio_metrics = raw_metrics
        baseline = BaselineManager.get_or_create_default_audio_baseline()
        baseline_stats = baseline.get("stats", {}) if baseline else {}

        # 1. Evaluate audio reliability & warnings
        rel_score, risk_score, raw_status, warnings, deviations = ReliabilityScoringEngine.evaluate_audio(
            raw_metrics, classification, baseline_stats
        )

        # 2. Smooth score and apply hysteresis
        smoothed_score = self.audio_smoother.update("audio_score", rel_score)
        sustained_status = self.audio_hysteresis.update(raw_status)

        self.latest_audio_score = round(smoothed_score, 1)
        self.audio_state = sustained_status

        # 3. Record event if sustained
        now = time.time()
        if sustained_status in ["warning", "caution"] and (now - self._last_event_time > 3.0):
            primary_warning = warnings[0] if warnings else None
            if primary_warning:
                add_event(
                    subsystem="audio",
                    severity=sustained_status,
                    title=f"Audio {primary_warning.metric.capitalize()} Degradation",
                    message=primary_warning.explanation,
                    metric_name=primary_warning.metric,
                    measured_value=primary_warning.measured_value,
                    expected_range=primary_warning.expected_range,
                    recommendation=primary_warning.recommendation,
                    metadata={"reliability_score": self.latest_audio_score}
                )
                self._last_event_time = now

        self._refresh_active_warnings(audio_warnings=warnings)

        return {
            "metrics": raw_metrics,
            "classification": classification,
            "reliability_score": self.latest_audio_score,
            "risk_score": round(100.0 - self.latest_audio_score, 1),
            "status": sustained_status,
            "warnings": warnings,
            "deviations": deviations
        }

    def set_llm_status(self, state: str, score: float = 100.0):
        self.llm_state = state
        self.latest_llm_score = score

    def _refresh_active_warnings(self,
                                 image_warnings: Optional[List[MetricWarning]] = None,
                                 audio_warnings: Optional[List[MetricWarning]] = None):
        combined = []
        if image_warnings is not None:
            combined.extend(image_warnings)
        if audio_warnings is not None:
            combined.extend(audio_warnings)
        self.active_warnings = combined

    def get_system_overall_status(self) -> SystemOverallStatus:
        """
        Determines overall system reliability state.
        Hierarchical rule:
        - If any subsystem has a severe WARNING, the overall state MUST be WARNING.
          (An acceptable audio score must not cancel a severe image warning).
        - If any subsystem has CAUTION and none WARNING -> CAUTION.
        - If all active subsystems are NORMAL -> NORMAL.
        """
        subsystem_states = [self.image_state, self.audio_state, self.llm_state]

        if "warning" in subsystem_states:
            overall_state = "warning"
        elif "caution" in subsystem_states:
            overall_state = "caution"
        elif "unavailable" in subsystem_states:
            overall_state = "unavailable"
        else:
            overall_state = "normal"

        # Overall reliability score is dominated by the weakest subsystem (bottleneck principle)
        overall_reliability = min(self.latest_image_score, self.latest_audio_score, self.latest_llm_score)

        return SystemOverallStatus(
            overall_state=overall_state,
            overall_reliability_score=round(overall_reliability, 1),
            image_state=self.image_state,
            image_reliability_score=round(self.latest_image_score, 1),
            audio_state=self.audio_state,
            audio_reliability_score=round(self.latest_audio_score, 1),
            llm_state=self.llm_state,
            active_warnings=self.active_warnings,
            recent_events_count=len(self.active_warnings),
            timestamp=time.time()
        )

# Global engine singleton
_engine_instance = None

def get_monitoring_engine() -> UnifiedMonitoringEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = UnifiedMonitoringEngine()
    return _engine_instance
