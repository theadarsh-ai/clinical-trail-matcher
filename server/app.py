"""
FastAPI server for Clinical Trial Matcher Environment
Exposes OpenEnv-compatible endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from fastapi.staticfiles import StaticFiles
from models import CTMatchAction, CTMatchObservation
from server.environment import ClinicalTrialMatcherEnv

import os

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Clinical Trial Matcher Environment",
    description="OpenEnv environment for matching patients to clinical trials",
    version="1.0.0"
)

# Serve static files for the Dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Redirect root to our index.html dashboard
# Redirect /dashboard to index.html dashboard
@app.get("/dashboard", include_in_schema=False)
async def get_dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))

# Root endpoint provides programmatic metadata (consolidated)
@app.get("/")
async def root():
    """Root endpoint with API information for automated scanners"""
    return {
        "name": "Clinical Trial Matcher Environment",
        "description": "OpenEnv environment for matching patients to clinical trials",
        "version": "1.0.0",
        "endpoints": {
            "core": ["/reset", "/step", "/state"],
            "hackathon": ["/baseline", "/grader", "/tasks"],
            "info": ["/health", "/docs", "/dashboard"]
        },
        "tasks": ["easy", "medium", "hard"],
        "github": "https://github.com/theadarsh-ai/clinical-trail-matcher"
    }

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
# In production, you'd want session management for concurrent users
env = ClinicalTrialMatcherEnv()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: CTMatchAction


class BaselineResponse(BaseModel):
    task: str
    scores: Dict[str, float]
    details: Dict[str, Any]


class GraderRequest(BaseModel):
    episode_data: Dict[str, Any]


class GraderResponse(BaseModel):
    reward: float
    metric_name: str
    details: Dict[str, Any]


# ============================================================================
# OPENENV CORE ENDPOINTS
# ============================================================================

@app.post("/reset", response_model=CTMatchObservation)
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode
    
    Args:
        task: "easy", "medium", or "hard"
        seed: Optional random seed for reproducibility
    
    Returns:
        Initial observation
    """
    try:
        observation = env.reset(task=request.task, seed=request.seed)
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=CTMatchObservation)
async def step(request: StepRequest):
    """
    Execute an action in the environment
    
    Args:
        action: Agent's proposed trial matches/ranking
    
    Returns:
        Observation with reward and feedback
    """
    try:
        observation = env.step(request.action)
        return observation
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state():
    """
    Get current environment state
    
    Returns:
        Current state information
    """
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count,
        "task": env.state.task
    }


# ============================================================================
# HACKATHON REQUIRED ENDPOINTS
# ============================================================================

@app.post("/baseline", response_model=BaselineResponse)
async def run_baseline():
    """
    Run baseline inference for all tasks and return scores
    Required for hackathon validation
    
    Returns:
        Baseline scores for easy, medium, and hard tasks
    """
    results = {
        "task": "all",
        "scores": {},
        "details": {}
    }
    
    try:
        # Run baseline for each task
        for task in ["easy", "medium", "hard"]:
            # Reset with fixed seed for reproducibility
            obs = env.reset(task=task, seed=12345)
            
            # Get baseline action (ground truth)
            baseline_action = env.get_baseline_action()
            
            # Execute action
            obs = env.step(baseline_action)
            
            # Store results
            results["scores"][task] = obs.reward
            results["details"][task] = {
                "metric_name": obs.info.get("metric_name", "unknown"),
                "proposed_count": len(baseline_action.proposed_trial_ids),
                "ground_truth_count": len(obs.info.get("ground_truth", []))
            }
        
        return BaselineResponse(**results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")


@app.post("/grader", response_model=GraderResponse)
async def run_grader(request: GraderRequest):
    """
    Grade an episode based on stored episode data
    Required for hackathon validation
    
    Args:
        episode_data: Contains task, action, patient, trials
    
    Returns:
        Grading score and details
    """
    try:
        from models import Patient, Trial
        from server.graders import grade_action
        
        # Extract data
        task = request.episode_data.get("task") or request.episode_data.get("task_id")
        action_data = request.episode_data.get("action")
        patient_data = request.episode_data.get("patient")
        trials_data = request.episode_data.get("trials")
        
        if task is None or action_data is None or patient_data is None or trials_data is None:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Parse models
        patient = Patient(**patient_data)
        trials = [Trial(**t) for t in trials_data]
        action = CTMatchAction(**action_data)
        
        # Grade
        result = grade_action(patient, trials, task, action.proposed_trial_ids)
        
        return GraderResponse(
            reward=result["reward"],
            metric_name=result["metric_name"],
            details=result["details"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grader failed: {str(e)}")


@app.get("/tasks")
async def get_tasks():
    """
    Get list of tasks and action schema
    Required for hackathon validation
    
    Returns:
        Task definitions and action schema
    """
    return env.get_tasks_info()


# ============================================================================
# DEMO ENDPOINT - Real per-trial eligibility breakdown for the UI
# ============================================================================

@app.post("/match_demo")
async def match_demo():
    """
    Run real eligibility matching on the current episode.
    Returns per-trial breakdown with REAL pass/fail reasons.
    Used by the visual dashboard only.
    """
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    from server.graders import (
        check_basic_eligibility, check_medical_eligibility,
        get_ground_truth_eligible, get_ground_truth_ranking, score_trial
    )

    patient = env.state.patient
    trials  = env.state.trials
    task    = env.state.task

    trial_results = []
    for trial in trials:
        basic_ok   = check_basic_eligibility(patient, trial)
        medical_ok = check_medical_eligibility(patient, trial) if task in ["medium", "hard"] else basic_ok

        # Build a reason string so the UI can display WHY it passed/failed
        reasons = []
        if patient.age < trial.min_age or patient.age > trial.max_age:
            reasons.append(f"Age {patient.age} outside {trial.min_age}-{trial.max_age}")
        if patient.gender not in trial.allowed_genders:
            reasons.append(f"Gender '{patient.gender}' not in {trial.allowed_genders}")
        if patient.city not in trial.allowed_cities:
            reasons.append(f"City '{patient.city}' not in {trial.allowed_cities[:2]}")
        if task in ["medium", "hard"]:
            for bm, val in trial.required_biomarkers.items():
                if bm not in patient.biomarkers or patient.biomarkers[bm] != val:
                    reasons.append(f"Missing biomarker {bm}={val}")
            for med in trial.excluded_medications:
                if med in patient.medications:
                    reasons.append(f"Excluded med: {med}")
            for tx in trial.required_prior_treatments:
                if tx not in patient.prior_treatments:
                    reasons.append(f"Missing treatment: {tx}")

        eligible = medical_ok if task in ["medium", "hard"] else basic_ok
        trial_score = score_trial(patient, trial) if task == "hard" else (1.0 if eligible else 0.0)

        trial_results.append({
            "id":          trial.id,
            "title":       trial.title,
            "phase":       trial.phase,
            "distance_km": round(trial.distance_km, 1),
            "min_age":     trial.min_age,
            "max_age":     trial.max_age,
            "allowed_cities": trial.allowed_cities,
            "allowed_genders": trial.allowed_genders,
            "has_slots":   trial.has_slots,
            "eligible":    eligible,
            "score":       round(trial_score, 2),
            "fail_reasons": reasons if not eligible else [],
        })

    # Ground truth
    gt_eligible = get_ground_truth_eligible(patient, trials, task)
    gt_ranking  = get_ground_truth_ranking(patient, trials) if task == "hard" else gt_eligible

    # Score the "perfect" action
    from server.graders import grade_action
    result = grade_action(patient, trials, task, gt_eligible if task != "hard" else gt_ranking)

    return {
        "task":          task,
        "patient":       {
            "condition":        patient.condition,
            "age":              patient.age,
            "gender":           patient.gender,
            "city":             patient.city,
            "biomarkers":       patient.biomarkers,
            "prior_treatments": patient.prior_treatments,
            "medications":      patient.medications,
            "max_travel_km":    patient.max_travel_km,
            "prefers_oral":     patient.prefers_oral,
        },
        "trials":            trial_results,
        "eligible_count":    len(gt_eligible),
        "total_trials":      len(trials),
        "baseline_reward":   result["reward"],
        "metric_name":       result["metric_name"],
        "ground_truth_ids":  gt_eligible if task != "hard" else gt_ranking,
    }



# ============================================================================
# PYTORCH TRAINING ENDPOINT
# ============================================================================

from pydantic import BaseModel
class TrainRequest(BaseModel):
    task: str = "medium"
    episodes: int = 50

@app.post("/train_real")
async def train_real(req: TrainRequest):
    """
    Executes real mathematical PyTorch Reinforcement Learning Training natively.
    """
    try:
        from server.pytorch_agent import run_real_pytorch_training
        metrics = run_real_pytorch_training(task=req.task, max_episodes=req.episodes)
        return {"data": metrics}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": "clinical_trial_matcher",
        "version": "1.0.0"
    }





# ============================================================================
# RUN SERVER
# ============================================================================

def main():
    """Main entry point for the server"""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
