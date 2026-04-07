"""
Synthetic data generator for Clinical Trial Matcher
Generates realistic patient profiles and trial specifications
"""

import random
from typing import List, Tuple
from models import Patient, Trial


# ============================================================================
# CONFIGURATION
# ============================================================================

CITIES = ["Boston", "New York", "Chicago", "Los Angeles", "Houston", "Miami"]

CONDITIONS = {
    "type_2_diabetes": {
        "biomarkers": ["HbA1c", "fasting_glucose"],
        "treatments": ["metformin", "insulin", "glp1_agonist"],
        "comorbidities": ["hypertension", "obesity", "hyperlipidemia"],
        "medications": ["metformin", "lisinopril", "atorvastatin"]
    },
    "her2_breast_cancer": {
        "biomarkers": ["HER2", "ER", "PR"],
        "treatments": ["trastuzumab", "pertuzumab", "chemotherapy"],
        "comorbidities": ["hypertension", "diabetes"],
        "medications": ["trastuzumab", "metformin", "lisinopril"]
    }
}


# ============================================================================
# PATIENT GENERATOR
# ============================================================================

def generate_patient(patient_id: str, condition: str, seed: int = None) -> Patient:
    """Generate a synthetic patient with realistic medical profile"""
    if seed is not None:
        random.seed(seed)
    
    config = CONDITIONS.get(condition, CONDITIONS["type_2_diabetes"])
    
    # Basic demographics
    age = random.randint(25, 75)
    gender = random.choice(["male", "female"])
    city = random.choice(CITIES)
    
    # Biomarkers (random subset with values)
    biomarkers = {}
    if condition == "type_2_diabetes":
        biomarkers["HbA1c"] = random.choice(["7.5", "8.0", "9.0", "10.0"])
        if random.random() > 0.5:
            biomarkers["fasting_glucose"] = random.choice(["140", "180", "200"])
    elif condition == "her2_breast_cancer":
        biomarkers["HER2"] = random.choice(["positive", "negative"])
        biomarkers["ER"] = random.choice(["positive", "negative"])
        biomarkers["PR"] = random.choice(["positive", "negative"])
    
    # Prior treatments (0-3 treatments)
    num_treatments = random.randint(0, 3)
    prior_treatments = random.sample(config["treatments"], min(num_treatments, len(config["treatments"])))
    
    # Comorbidities (0-3)
    num_comorbidities = random.randint(0, 3)
    comorbidities = random.sample(config["comorbidities"], min(num_comorbidities, len(config["comorbidities"])))
    
    # Current medications (related to comorbidities)
    medications = []
    if "hypertension" in comorbidities:
        medications.append("lisinopril")
    if "diabetes" in comorbidities or condition == "type_2_diabetes":
        if random.random() > 0.5:
            medications.append("metformin")
    
    # Logistics
    max_travel_km = random.choice([30, 50, 100, 200])
    prefers_oral = random.choice([True, False])
    
    return Patient(
        id=patient_id,
        age=age,
        gender=gender,
        condition=condition,
        city=city,
        biomarkers=biomarkers,
        prior_treatments=prior_treatments,
        comorbidities=comorbidities,
        medications=medications,
        max_travel_km=max_travel_km,
        prefers_oral=prefers_oral
    )


# ============================================================================
# TRIAL GENERATOR
# ============================================================================

def generate_trial(
    trial_id: str,
    condition: str,
    patient_city: str,
    difficulty: str = "easy",
    seed: int = None
) -> Trial:
    """Generate a synthetic clinical trial with eligibility criteria"""
    if seed is not None:
        random.seed(seed)
    
    config = CONDITIONS.get(condition, CONDITIONS["type_2_diabetes"])
    
    # Basic eligibility (always present)
    min_age = random.choice([18, 21, 30, 40])
    max_age = random.choice([65, 70, 75, 80])
    allowed_genders = random.choice([
        ["male", "female"],
        ["female"],
        ["male"]
    ])
    
    # Allowed cities (include patient city + others)
    num_cities = random.randint(2, 4)
    other_cities = [c for c in CITIES if c != patient_city]
    allowed_cities = [patient_city] + random.sample(other_cities, min(num_cities - 1, len(other_cities)))
    
    # Medical criteria (for medium/hard)
    required_biomarkers = {}
    excluded_biomarkers = {}
    required_prior_treatments = []
    excluded_medications = []
    max_comorbidities = None
    
    if difficulty in ["medium", "hard"]:
        # Add biomarker requirements
        if condition == "her2_breast_cancer":
            if random.random() > 0.5:
                required_biomarkers["HER2"] = "positive"
            if random.random() > 0.7:
                excluded_biomarkers["ER"] = "positive"
        
        # Add treatment requirements
        if random.random() > 0.6 and len(config["treatments"]) > 0:
            required_prior_treatments = [random.choice(config["treatments"])]
        
        # Add medication exclusions
        if random.random() > 0.7 and len(config["medications"]) > 0:
            excluded_medications = [random.choice(config["medications"])]
        
        # Add comorbidity limit
        if random.random() > 0.5:
            max_comorbidities = random.randint(2, 4)
    
    # Logistics
    phase = random.choice(["phase_1", "phase_2", "phase_3"])
    distance_km = random.uniform(10, 150)
    has_slots = random.choice([True, True, False])  # 2/3 chance of slots
    visit_frequency_per_month = random.choice([1, 2, 4, 8])
    is_oral = random.choice([True, False])
    
    return Trial(
        id=trial_id,
        title=f"Trial for {condition.replace('_', ' ').title()}",
        min_age=min_age,
        max_age=max_age,
        allowed_genders=allowed_genders,
        required_condition=condition,
        allowed_cities=allowed_cities,
        required_biomarkers=required_biomarkers,
        excluded_biomarkers=excluded_biomarkers,
        required_prior_treatments=required_prior_treatments,
        excluded_medications=excluded_medications,
        max_comorbidities=max_comorbidities,
        phase=phase,
        distance_km=distance_km,
        has_slots=has_slots,
        visit_frequency_per_month=visit_frequency_per_month,
        is_oral=is_oral
    )


# ============================================================================
# EPISODE GENERATOR
# ============================================================================

def generate_episode(
    episode_id: str,
    task: str,
    num_trials: int = 8,
    seed: int = None
) -> Tuple[Patient, List[Trial]]:
    """
    Generate a complete episode: one patient + multiple trials
    
    Args:
        episode_id: Unique episode identifier
        task: "easy", "medium", or "hard"
        num_trials: Number of trials to generate
        seed: Random seed for reproducibility
    
    Returns:
        (patient, trials) tuple
    """
    if seed is not None:
        random.seed(seed)
    
    # Choose condition
    condition = random.choice(list(CONDITIONS.keys()))
    
    # Generate patient
    patient = generate_patient(f"patient_{episode_id}", condition, seed=seed)
    
    # Generate trials
    trials = []
    for i in range(num_trials):
        trial_seed = (seed + i) if seed is not None else None
        trial = generate_trial(
            trial_id=f"trial_{episode_id}_{i:03d}",
            condition=condition,
            patient_city=patient.city,
            difficulty=task,
            seed=trial_seed
        )
        trials.append(trial)
    
    return patient, trials


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test generation
    patient, trials = generate_episode("test_001", task="medium", seed=42)
    
    print(f"Patient: {patient.id}, Age: {patient.age}, Condition: {patient.condition}")
    print(f"Biomarkers: {patient.biomarkers}")
    print(f"Prior treatments: {patient.prior_treatments}")
    print(f"\nGenerated {len(trials)} trials")
    for trial in trials[:3]:
        print(f"  {trial.id}: Age {trial.min_age}-{trial.max_age}, Cities: {trial.allowed_cities[:2]}")
