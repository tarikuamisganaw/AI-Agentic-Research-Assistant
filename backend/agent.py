import re
from typing import Dict, Any

def verify_answer(answer: str, context: str) -> Dict[str, Any]:
    """Heuristic self-verification: checks if key claims in answer exist in context."""
    if not answer or not context:
        return {"confidence": 0.0, "status": "no_context", "missing": "No context provided"}
    
    answer_lower = answer.lower()
    context_lower = context.lower()
    stopwords = {"the", "and", "for", "with", "this", "that", "have", "been", "are", "was", "is", "it", "in", "of", "to", "a"}
    key_phrases = [w for w in re.findall(r'\b\w{4,}\b', answer_lower) if w not in stopwords]
    
    if not key_phrases:
        return {"confidence": 50.0, "status": "generic", "missing": "Answer too generic to verify"}
    
    matches = sum(1 for p in key_phrases if p in context_lower)
    confidence = round((matches / len(key_phrases)) * 100, 1)
    
    if confidence >= 80: status, missing = "fully_supported", "None"
    elif confidence >= 50: status, missing = "partially_supported", f"{len(key_phrases) - matches} claims not in context"
    else: status, missing = "weakly_supported", f"Most claims ({len(key_phrases)-matches}/{len(key_phrases)}) lack context"
    
    return {"confidence": confidence, "status": status, "missing": missing}

def needs_clarification(verification: Dict[str, Any]) -> bool:
    """Triggers fallback if confidence is too low."""
    return verification["confidence"] < 45 or verification["status"] == "weakly_supported"