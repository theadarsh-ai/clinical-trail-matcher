"""
Grading functions for Clinical Trial Matcher
Implements F1 score (easy/medium) and NDCG (hard) metrics
"""

import math
from typing import List, Set, Dict
from models import Patient, Trial


# ============================================================================
# ELIGIBILITY CHECKER (Ground Truth)
# ============================================================================

def check_basic_eligibility(patient: Patient, trial: Trial) -> bool:
    """Check basic eligibility criteria (Task 1: Easy)"""
    # Age
    if not (trial.min_age <= patient.age <= trial.max_age):
        return False
    
    # Gender
    if patient.gender not in trial.allowed_genders:
        return False
    
    # Condition
    if patient.condition != trial.required_condition:
        return False
    
    # City
    if patient.city not in trial.allowed_cities:
        return False
    
    return True


def check_medical_eligibility(patient: Patient, trial: Trial) -> bool:
    """Check medical criteria (Task 2: Medium) - includes basic checks"""
    # First check basic eligibility
    if not check_basic_eligibility(patient, trial):
        return False
    
    # Required biomarkers
    for biomarker, required_value in trial.required_biomarkers.items():
        if biomarker not in patient.biomarkers:
            return False
        if patient.biomarkers[biomarker] != required_value:
            return False
    
    # Excluded biomarkers
    for biomarker, excluded_value in trial.excluded_biomarkers.items():
        if biomarker in patient.biomarkers:
            if patient.biomarkers[biomarker] == excluded_value:
                return False
    
    # Required prior treatments
    for treatment in trial.required_prior_treatments:
        if treatment not in patient.prior_treatments:
            return False
    
    # Excluded medications
    for medication in trial.excluded_medications:
        if medication in patient.medications:
            return False
    
    # Max comorbidities
    if trial.max_comorbidities is not None:
        if len(patient.comorbidities) > trial.max_comorbidities:
            return False
    
    return True


def get_ground_truth_eligible(
    patient: Patient,
    trials: List[Trial],
    task: str
) -> List[str]:
    """Get ground truth list of eligible trial IDs"""
    eligible = []
    
    for trial in trials:
        if task == "easy":
            is_eligible = check_basic_eligibility(patient, trial)
        else:  # medium or hard
            is_eligible = check_medical_eligibility(patient, trial)
        
        if is_eligible:
            eligible.append(trial.id)
    
    return eligible


# ============================================================================
# TRIAL SCORING (For Hard Task Ranking)
# ============================================================================

def score_trial(patient: Patient, trial: Trial) -> float:
    """
    Score a trial for a patient (0.0 to 100.0)
    Higher score = better match
    Used for Task 3: Hard (ranking)
    """
    # Only score eligible trials
    if not check_medical_eligibility(patient, trial):
        return 0.0
    
    score = 50.0  # Base score for being eligible
    
    # Phase preference (more mature = better)
    phase_scores = {"phase_1": 0, "phase_2": 10, "phase_3": 20}
    score += phase_scores.get(trial.phase, 0)
    
    # Distance penalty
    if trial.distance_km > patient.max_travel_km:
        score -= min(20, (trial.distance_km - patient.max_travel_km) / 10)
    else:
        score += 5  # Bonus for being within travel range
    
    # Slots availability
    if trial.has_slots:
        score += 10
    else:
        score -= 10  # Penalty for waitlist
    
    # Visit frequency penalty (fewer visits = better)
    if trial.visit_frequency_per_month > 4:
        score -= 5
    elif trial.visit_frequency_per_month <= 2:
        score += 5
    
    # Oral preference
    if patient.prefers_oral and trial.is_oral:
        score += 10
    elif patient.prefers_oral and not trial.is_oral:
        score -= 5
    
    return max(0.0, score)


def get_ground_truth_ranking(patient: Patient, trials: List[Trial]) -> List[str]:
    """Get ground truth ranking of trials (best to worst)"""
    # Score all trials
    scored_trials = []
    for trial in trials:
        score = score_trial(patient, trial)
        if score > 0:  # Only rank eligible trials
            scored_trials.append((trial.id, score))
    
    # Sort by score (descending)
    scored_trials.sort(key=lambda x: x[1], reverse=True)
    
    return [trial_id for trial_id, _ in scored_trials]


# ============================================================================
# GRADING METRICS
# ============================================================================

def calculate_f1_score(
    predicted: List[str],
    ground_truth: List[str]
) -> float:
    """
    Calculate F1 score for eligibility classification
    Used for Tasks 1 & 2
    """
    if len(ground_truth) == 0 and len(predicted) == 0:
        return 1.0  # Perfect if both empty
    
    if len(ground_truth) == 0 or len(predicted) == 0:
        return 0.0  # No match if one is empty
    
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    
    true_positives = len(predicted_set & ground_truth_set)
    false_positives = len(predicted_set - ground_truth_set)
    false_negatives = len(ground_truth_set - predicted_set)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_ndcg(
    predicted_ranking: List[str],
    ideal_ranking: List[str],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k)
    Used for Task 3 (Hard)
    
    Args:
        predicted_ranking: Agent's proposed ranking
        ideal_ranking: Ground truth ideal ranking
        k: Cutoff rank (default 10)
    
    Returns:
        NDCG score between 0.0 and 1.0
    """
    if len(ideal_ranking) == 0:
        return 1.0 if len(predicted_ranking) == 0 else 0.0
    
    # Create relevance mapping (position in ideal ranking)
    relevance_map = {}
    for i, trial_id in enumerate(ideal_ranking):
        # Higher relevance for better ranks (inverted index)
        relevance_map[trial_id] = len(ideal_ranking) - i
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, trial_id in enumerate(predicted_ranking[:k]):
        if trial_id in relevance_map:
            relevance = relevance_map[trial_id]
            dcg += relevance / math.log2(i + 2)  # i+2 because positions start at 1
    
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    for i, trial_id in enumerate(ideal_ranking[:k]):
        relevance = relevance_map[trial_id]
        idcg += relevance / math.log2(i + 2)
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


# ============================================================================
# MAIN GRADER FUNCTION
# ============================================================================

def grade_action(
    patient: Patient,
    trials: List[Trial],
    task: str,
    proposed_trial_ids: List[str]
) -> Dict:
    """
    Grade an agent's action
    
    Returns:
        {
            "reward": float (0.0 to 1.0),
            "ground_truth": list or dict,
            "metric_name": str,
            "details": dict
        }
    """
    if task == "easy":
        # F1 score for basic eligibility
        ground_truth = get_ground_truth_eligible(patient, trials, "easy")
        f1 = calculate_f1_score(proposed_trial_ids, ground_truth)
        
        return {
            "reward": f1,
            "ground_truth": ground_truth,
            "metric_name": "f1_score",
            "details": {
                "predicted_count": len(proposed_trial_ids),
                "ground_truth_count": len(ground_truth),
                "predicted": proposed_trial_ids,
                "ground_truth": ground_truth
            }
        }
    
    elif task == "medium":
        # F1 score for medical eligibility
        ground_truth = get_ground_truth_eligible(patient, trials, "medium")
        f1 = calculate_f1_score(proposed_trial_ids, ground_truth)
        
        return {
            "reward": f1,
            "ground_truth": ground_truth,
            "metric_name": "f1_score",
            "details": {
                "predicted_count": len(proposed_trial_ids),
                "ground_truth_count": len(ground_truth),
                "predicted": proposed_trial_ids,
                "ground_truth": ground_truth
            }
        }
    
    elif task == "hard":
        # NDCG for ranking
        ideal_ranking = get_ground_truth_ranking(patient, trials)
        ndcg = calculate_ndcg(proposed_trial_ids, ideal_ranking, k=10)
        
        return {
            "reward": ndcg,
            "ground_truth": ideal_ranking,
            "metric_name": "ndcg@10",
            "details": {
                "predicted_ranking": proposed_trial_ids,
                "ideal_ranking": ideal_ranking,
                "eligible_count": len(ideal_ranking)
            }
        }
    
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test grading functions
    from data_generator import generate_episode
    
    patient, trials = generate_episode("test", "medium", seed=42)
    
    # Get ground truth
    eligible_easy = get_ground_truth_eligible(patient, trials, "easy")
    eligible_medium = get_ground_truth_eligible(patient, trials, "medium")
    ranking = get_ground_truth_ranking(patient, trials)
    
    print(f"Easy eligible: {len(eligible_easy)} trials")
    print(f"Medium eligible: {len(eligible_medium)} trials")
    print(f"Hard ranking: {len(ranking)} trials")
    print(f"Top 3 ranked: {ranking[:3]}")
    
    # Test F1 score
    test_predicted = eligible_easy[:2]  # Predict subset
    f1 = calculate_f1_score(test_predicted, eligible_easy)
    print(f"\nTest F1 score: {f1:.3f}")
    
    # Test NDCG
    test_ranking = ranking[::-1]  # Reverse ranking (worst case)
    ndcg = calculate_ndcg(test_ranking, ranking)
    print(f"Test NDCG (reversed): {ndcg:.3f}")
