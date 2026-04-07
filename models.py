"""
Pydantic models for Clinical Trial Matcher Environment
Defines Action, Observation, and State types for OpenEnv compatibility
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


# ============================================================================
# DOMAIN MODELS (Patient, Trial)
# ============================================================================

class Patient(BaseModel):
    """Patient profile with medical and logistical information"""
    id: str
    age: int
    gender: Literal["male", "female", "other"]
    condition: str  # e.g., "type_2_diabetes", "her2_breast_cancer"
    city: str
    biomarkers: Dict[str, str] = Field(default_factory=dict)  # e.g., {"HER2": "positive"}
    prior_treatments: List[str] = Field(default_factory=list)  # e.g., ["trastuzumab"]
    comorbidities: List[str] = Field(default_factory=list)  # e.g., ["hypertension"]
    medications: List[str] = Field(default_factory=list)  # e.g., ["metformin"]
    max_travel_km: float = 50.0
    prefers_oral: bool = False


class Trial(BaseModel):
    """Clinical trial with eligibility criteria and logistics"""
    id: str
    title: str
    
    # Basic eligibility
    min_age: int
    max_age: int
    allowed_genders: List[str]
    required_condition: str
    allowed_cities: List[str]
    
    # Medical criteria
    required_biomarkers: Dict[str, str] = Field(default_factory=dict)
    excluded_biomarkers: Dict[str, str] = Field(default_factory=dict)
    required_prior_treatments: List[str] = Field(default_factory=list)
    excluded_medications: List[str] = Field(default_factory=list)
    max_comorbidities: Optional[int] = None
    
    # Logistics
    phase: Literal["phase_1", "phase_2", "phase_3"]
    distance_km: float
    has_slots: bool
    visit_frequency_per_month: float
    is_oral: bool


# ============================================================================
# OPENENV ACTION
# ============================================================================

class CTMatchAction(BaseModel):
    """
    Action for Clinical Trial Matcher
    
    Agent can either:
    - list_eligible: Return all eligible trial IDs (for easy/medium tasks)
    - rank_trials: Return ranked trial IDs (for hard task)
    """
    action_type: Literal["list_eligible", "rank_trials"]
    task: Literal["easy", "medium", "hard"]
    proposed_trial_ids: List[str] = Field(
        default_factory=list,
        description="Trial IDs (unordered for list_eligible, ranked for rank_trials)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "action_type": "list_eligible",
                "task": "easy",
                "proposed_trial_ids": ["trial_001", "trial_003"]
            }
        }


# ============================================================================
# OPENENV OBSERVATION
# ============================================================================

class CTMatchObservation(BaseModel):
    """
    Observation returned by the environment
    Contains patient data, trials, and feedback
    """
    # Episode context
    patient: Patient
    trials: List[Trial]
    task: Literal["easy", "medium", "hard"]
    
    # Feedback
    message: str = "Episode in progress"
    done: bool = False
    reward: float = 0.0
    
    # Ground truth (only provided after step, in info)
    info: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "patient": {
                    "id": "patient_001",
                    "age": 45,
                    "gender": "female",
                    "condition": "type_2_diabetes",
                    "city": "Boston"
                },
                "trials": [],
                "task": "easy",
                "message": "Episode started",
                "done": False,
                "reward": 0.0,
                "info": {}
            }
        }


# ============================================================================
# OPENENV STATE
# ============================================================================

class CTMatchState(BaseModel):
    """
    Internal state tracking for the environment
    """
    episode_id: str
    step_count: int = 0
    task: Literal["easy", "medium", "hard"]
    
    # Ground truth for grading
    ground_truth_eligible: List[str] = Field(default_factory=list)
    ground_truth_ranking: List[str] = Field(default_factory=list)
    
    # Episode data
    patient: Optional[Patient] = None
    trials: List[Trial] = Field(default_factory=list)
