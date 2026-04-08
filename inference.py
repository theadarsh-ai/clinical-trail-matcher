"""
Inference Script for Clinical Trial Matcher Environment
========================================================
MANDATORY REQUIREMENTS:
- Uses OpenAI Client for LLM calls
- Emits structured stdout logs: [START], [STEP], [END]
- Returns scores in [0, 1] range
- Named inference.py in root directory

Environment Variables (set these before running):
    API_BASE_URL    The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME      The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        Your Hugging Face API key (required)
    LOCAL_IMAGE_NAME The Docker image name (if using from_docker_image)
    TASK_NAME       Task to run: easy, medium, or hard (default: easy)
"""

import asyncio
import os
import sys
import json
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# Import our environment - adjust import based on your deployment
try:
    from client import ClinicalTrialMatcherClient
    from models import CTMatchAction
    USE_CLIENT = True
except ImportError:
    # Fallback for Docker deployment
    from server.environment import ClinicalTrialMatcherEnv
    from models import CTMatchAction
    USE_CLIENT = False


# ============================================================================
# CONFIGURATION
# ============================================================================

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "clinical-trial-matcher")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "1"))  # Set >1 to run multiple patients
BENCHMARK = "clinical_trial_matcher"

MAX_STEPS = 1  # Clinical trial matching is single-step
TEMPERATURE = 0.7
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.7  # 70% score to pass


# ============================================================================
# LOGGING FUNCTIONS (Required Format)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log episode start"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log each step"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# ============================================================================
# LLM AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an expert clinical trial matching agent.
Your task is to match patients to suitable clinical trials based on eligibility criteria.

You will be given:
- A patient profile (age, gender, condition, biomarkers, medical history, location, preferences)
- A list of clinical trials with eligibility criteria
- The task difficulty level (easy, medium, or hard)

For EASY and MEDIUM tasks:
- Return a list of trial IDs that the patient is eligible for
- Format: ["trial_id_1", "trial_id_2", ...]

For HARD tasks:
- Return a ranked list of trial IDs (best match first)
- Consider: trial phase, distance, availability, visit frequency, patient preferences
- Format: ["best_trial", "second_best", ...]

Respond with ONLY a valid JSON array of trial IDs, nothing else.
Example: ["trial_001", "trial_003", "trial_005"]
"""


def build_user_prompt(patient: dict, trials: List[dict], task: str) -> str:
    """Build prompt for LLM with patient and trials data"""
    patient_summary = f"""
Patient Profile:
- ID: {patient['id']}
- Age: {patient['age']}
- Gender: {patient['gender']}
- Condition: {patient['condition']}
- City: {patient['city']}
- Biomarkers: {patient.get('biomarkers', {})}
- Prior Treatments: {patient.get('prior_treatments', [])}
- Comorbidities: {patient.get('comorbidities', [])}
- Current Medications: {patient.get('medications', [])}
- Max Travel Distance: {patient.get('max_travel_km', 50)} km
- Prefers Oral: {patient.get('prefers_oral', False)}
"""
    
    trials_summary = "\nAvailable Trials:\n"
    for i, trial in enumerate(trials, 1):
        trials_summary += f"""
Trial {i}: {trial['id']}
- Title: {trial['title']}
- Age Range: {trial['min_age']}-{trial['max_age']}
- Genders: {trial['allowed_genders']}
- Condition: {trial['required_condition']}
- Cities: {trial['allowed_cities']}
- Phase: {trial['phase']}
- Distance: {trial['distance_km']:.1f} km
- Has Slots: {trial['has_slots']}
"""
        # Add medical criteria for medium/hard
        if task in ["medium", "hard"]:
            if trial.get('required_biomarkers'):
                trials_summary += f"- Required Biomarkers: {trial['required_biomarkers']}\n"
            if trial.get('excluded_biomarkers'):
                trials_summary += f"- Excluded Biomarkers: {trial['excluded_biomarkers']}\n"
            if trial.get('required_prior_treatments'):
                trials_summary += f"- Required Treatments: {trial['required_prior_treatments']}\n"
            if trial.get('excluded_medications'):
                trials_summary += f"- Excluded Medications: {trial['excluded_medications']}\n"
    
    task_instruction = ""
    if task in ["easy", "medium"]:
        task_instruction = f"\nTask ({task.upper()}): Return ALL trial IDs where the patient is eligible."
    else:  # hard
        task_instruction = f"\nTask ({task.upper()}): Return trial IDs RANKED from best to worst match."
    
    return patient_summary + trials_summary + task_instruction + "\n\nRespond with ONLY a JSON array of trial IDs:"


def get_model_prediction(
    client: OpenAI,
    patient: dict,
    trials: List[dict],
    task: str
) -> List[str]:
    """Get LLM prediction for trial matching"""
    user_prompt = build_user_prompt(patient, trials, task)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        
        response_text = (completion.choices[0].message.content or "").strip()
        
        # Parse JSON response
        # Remove markdown code blocks if present
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        trial_ids = json.loads(response_text)
        
        if isinstance(trial_ids, list):
            return [str(tid) for tid in trial_ids]
        else:
            print(f"[DEBUG] Invalid response format: {response_text}", flush=True)
            return []
    
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        print(f"[DEBUG] Response was: {response_text[:200]}", flush=True)
        return []
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return []


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

async def run_inference_local():
    """Run inference using local environment (no Docker)"""
    from server.environment import ClinicalTrialMatcherEnv
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ClinicalTrialMatcherEnv()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment with no fixed seed for variety
        obs = env.reset(task=TASK_NAME)
        
        # Extract patient and trials data  (Pydantic v2: use model_dump())
        patient_dict = obs.patient.model_dump() if hasattr(obs.patient, 'model_dump') else obs.patient.dict()
        trials_list = [
            (trial.model_dump() if hasattr(trial, 'model_dump') else trial.dict())
            for trial in obs.trials
        ]
        
        # Get LLM prediction
        predicted_trial_ids = get_model_prediction(client, patient_dict, trials_list, TASK_NAME)
        
        # Create action
        action_type = "list_eligible" if TASK_NAME in ["easy", "medium"] else "rank_trials"
        action = CTMatchAction(
            action_type=action_type,
            task=TASK_NAME,
            proposed_trial_ids=predicted_trial_ids
        )
        
        # Execute step
        obs = env.step(action)
        
        reward = obs.reward
        done = obs.done
        error = None
        
        rewards.append(reward)
        steps_taken = 1
        
        # Format action for logging (truncate if too long)
        action_str = str(predicted_trial_ids)
        if len(action_str) > 100:
            action_str = action_str[:97] + "..."
        
        log_step(step=1, action=action_str, reward=reward, done=done, error=error)
        
        score = reward  # Score is already in [0, 1] range
        success = score >= SUCCESS_SCORE_THRESHOLD
    
    except Exception as e:
        print(f"[DEBUG] Inference error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_inference_http():
    """Run inference using HTTP client (for deployed environment)"""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = ClinicalTrialMatcherClient("http://localhost:8000")
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment
        obs = env_client.reset(task=TASK_NAME, seed=42)
        
        # Extract patient and trials data  (Pydantic v2: use model_dump())
        patient_dict = obs.patient.model_dump() if hasattr(obs.patient, 'model_dump') else obs.patient.dict()
        trials_list = [
            (trial.model_dump() if hasattr(trial, 'model_dump') else trial.dict())
            for trial in obs.trials
        ]
        
        # ── VERBOSE: Show LLM prediction ────────────────────────────────
        print(f"[LLM] Model response: {predicted_trial_ids}", flush=True)
        print(f"[LLM] Proposed {len(predicted_trial_ids)} trial(s) out of {len(trials_list)}", flush=True)
        print(f"{'─'*60}\n", flush=True)
        # ────────────────────────────────────────────────────────────────
        
        # Create action
        action_type = "list_eligible" if TASK_NAME in ["easy", "medium"] else "rank_trials"
        action = CTMatchAction(
            action_type=action_type,
            task=TASK_NAME,
            proposed_trial_ids=predicted_trial_ids
        )
        
        # Execute step
        obs = env_client.step(action)
        
        reward = obs.reward
        done = obs.done
        error = None
        
        rewards.append(reward)
        steps_taken = 1
        
        # Format action for logging
        action_str = str(predicted_trial_ids)
        if len(action_str) > 100:
            action_str = action_str[:97] + "..."
        
        log_step(step=1, action=action_str, reward=reward, done=done, error=error)
        
        score = reward
        success = score >= SUCCESS_SCORE_THRESHOLD
    
    except Exception as e:
        print(f"[DEBUG] Inference error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ============================================================================
# ENTRY POINT
# ============================================================================

async def run_single_task(task: str, client, env):
    """Run inference for a single task and emit required logs."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task=task)

        patient_dict = obs.patient.model_dump() if hasattr(obs.patient, 'model_dump') else obs.patient.dict()
        trials_list = [
            (t.model_dump() if hasattr(t, 'model_dump') else t.dict())
            for t in obs.trials
        ]

        predicted_trial_ids = get_model_prediction(client, patient_dict, trials_list, task)

        action_type = "list_eligible" if task in ["easy", "medium"] else "rank_trials"
        action = CTMatchAction(
            action_type=action_type,
            task=task,
            proposed_trial_ids=predicted_trial_ids
        )

        obs = env.step(action)
        reward = obs.reward
        done = obs.done

        rewards.append(reward)
        steps_taken = 1

        action_str = str(predicted_trial_ids)
        if len(action_str) > 100:
            action_str = action_str[:97] + "..."

        log_step(step=1, action=action_str, reward=reward, done=done, error=None)

        score = reward
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Inference error for task={task}: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    """Main entry point - runs ALL 3 tasks so validator sees every grader."""
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable not set!", flush=True)
        sys.exit(1)

    from server.environment import ClinicalTrialMatcherEnv
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ClinicalTrialMatcherEnv()

    # Always run all 3 tasks so the validator counts 3 graded tasks
    all_tasks = ["easy", "medium", "hard"]
    scores = {}
    for task in all_tasks:
        scores[task] = await run_single_task(task, client, env)

    # Final summary
    avg = sum(scores.values()) / len(scores)
    print(f"[SUMMARY] easy={scores['easy']:.3f} medium={scores['medium']:.3f} hard={scores['hard']:.3f} avg={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

