from dataclasses import dataclass


class CleaningSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class CleaningDecision:
    allowed: bool
    reason: str


def cleaning_decision(condition: str, confidence: float = 100,
                      threshold: float = 0, affected_area: float | None = None,
                      dust_threshold: float = 0) -> CleaningDecision:
    if not 0 <= confidence <= 100 or not 0 <= threshold <= 100:
        return CleaningDecision(False, "Confidence values must be between 0 and 100.")
    if not 0 <= dust_threshold <= 100:
        return CleaningDecision(False, "Dust threshold must be between 0 and 100.")
    if affected_area is not None and not 0 <= affected_area <= 100:
        return CleaningDecision(False, "Estimated affected area must be between 0 and 100.")
    normalized = condition.strip().lower()
    if normalized == "dust":
        if confidence < threshold:
            return CleaningDecision(False, "Dust confidence is below the configured threshold.")
        if affected_area is None:
            return CleaningDecision(False, "No inspection area estimate is available.")
        if affected_area < dust_threshold:
            return CleaningDecision(False, "Estimated dust coverage is below the configured threshold.")
        return CleaningDecision(True, "Dust detected; cleaning is permitted.")
    reasons = {
        "normal": "Cleaning is not required for a normal panel.",
        "crack": "Cleaning is prohibited when panel damage is detected.",
        "partial shading": "Cleaning is prohibited for partial shading.",
    }
    return CleaningDecision(False, reasons.get(normalized, "Unknown condition; cleaning is prohibited."))


def require_cleaning_allowed(condition: str, confidence: float = 100,
                             threshold: float = 0, affected_area: float | None = None,
                             dust_threshold: float = 0) -> None:
    decision = cleaning_decision(condition, confidence, threshold, affected_area, dust_threshold)
    if not decision.allowed:
        raise CleaningSafetyError(decision.reason)


def validate_servo_limits(min_angle: int, max_angle: int) -> None:
    if not 0 <= min_angle < max_angle <= 180:
        raise ValueError("Servo limits must satisfy 0 <= minimum < maximum <= 180 degrees.")
