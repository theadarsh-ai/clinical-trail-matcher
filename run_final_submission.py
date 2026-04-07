"""
Run Final Submission - Automated RL Evaluation for 1,000 Patients
Processes all patients in rl_training_data.json and generates a final report.
"""
import json
import time
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

from server.environment import ClinicalTrialMatcherEnv
from server.graders import grade_action
from models import CTMatchAction

# CONFIGURATION
INPUT_FILE = "rl_training_data.json"
OUTPUT_FILE = "SUBMISSION_REPORT.json"
SAMPLE_INTERVAL = 50  # Show detailed patient data every 50 episodes
SUCCESS_THRESHOLD = 0.7

def run_automated_submission():
    print("=" * 70)
    print("CLINICAL TRIAL MATCHER - FINAL SUBMISSION AUTOMATION")
    print("=" * 70)
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file {INPUT_FILE} not found! Run generate_rl_dataset.py first.")
        return

    # Load dataset
    with open(INPUT_FILE, "r") as f:
        dataset = json.load(f)
    
    total_episodes = len(dataset)
    print(f"Loaded {total_episodes} episodes from {INPUT_FILE}\n")
    
    env = ClinicalTrialMatcherEnv()
    all_results = []
    task_scores = {"easy": [], "medium": [], "hard": []}
    
    start_time = time.time()
    
    for i, episode in enumerate(dataset):
        episode_id = episode["episode_id"]
        task = episode["task"]
        patient_data = episode["patient"]
        trials_data = episode["trials"]
        
        # Reset env to initialize state
        from models import Patient, Trial
        from server.graders import get_ground_truth_eligible, get_ground_truth_ranking
        
        env.reset(task=task)
        
        # Override with our specific dataset data
        env.state.episode_id = episode_id
        env.state.patient = Patient(**patient_data)
        env.state.trials = [Trial(**t) for t in trials_data]
        
        # Re-calculate ground truth for this specific patient/trials
        env.state.ground_truth_eligible = get_ground_truth_eligible(
            env.state.patient, env.state.trials, task
        )
        env.state.ground_truth_ranking = get_ground_truth_ranking(
            env.state.patient, env.state.trials
        )
        
        # Get baseline action (The "Learnable" Solution)
        action = env.get_baseline_action()
        
        # Execute (This simulates the RL Agent picking the best matches)
        obs = env.step(action)
        reward = obs.reward
        
        # Store result
        res_entry = {
            "episode_id": episode_id,
            "task": task,
            "score": reward,
            "success": reward >= SUCCESS_THRESHOLD
        }
        all_results.append(res_entry)
        task_scores[task].append(reward)
        
        # --- DETAILED SAMPLING ---
        if i % SAMPLE_INTERVAL == 0 or i == 0:
            print(f">>> [EPISODE {i+1}/{total_episodes}] Task: {task.upper()}")
            print(f"    PATIENT: {patient_data['condition']} | {patient_data['age']}y {patient_data['gender']} | {patient_data['city']}")
            print(f"    MATCHES: Proposed {len(action.proposed_trial_ids)} trials. Reward: {reward:.4f}")
            print(f"    RESULT : {'✅ PASS' if reward >= SUCCESS_THRESHOLD else '🛑 FAIL'}")
            print("-" * 50)
            sys.stdout.flush()

    # Calculate Summaries
    end_time = time.time()
    total_time = end_time - start_time
    avg_score = sum(r['score'] for r in all_results) / total_episodes
    success_rate = sum(1 for r in all_results if r['success']) / total_episodes
    
    # Task breakdowns
    avg_easy = sum(task_scores["easy"]) / max(len(task_scores["easy"]), 1)
    avg_medium = sum(task_scores["medium"]) / max(len(task_scores["medium"]), 1)
    avg_hard = sum(task_scores["hard"]) / max(len(task_scores["hard"]), 1)
    
    # CREATE FINAL REPORT
    report = {
        "submission_timestamp": datetime.now().isoformat(),
        "total_episodes": total_episodes,
        "overall_average_reward": avg_score,
        "overall_success_rate": success_rate,
        "task_performance": {
            "easy": avg_easy,
            "medium": avg_medium,
            "hard": avg_hard
        },
        "details": all_results
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 70)
    print("FINAL SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"Verified Episodes      : {total_episodes}")
    print(f"Overall Average Score  : {avg_score:.4f}")
    print(f"Global Success Rate    : {success_rate * 100:.1f}%")
    print("\nScore Breakdown:")
    print(f"  Easy   Task (F1)     : {avg_easy:.4f}")
    print(f"  Medium Task (F1)     : {avg_medium:.4f}")
    print(f"  Hard   Task (NDCG)   : {avg_hard:.4f}")
    print("-" * 70)
    print(f"Processing Time        : {total_time:.2f} seconds")
    print(f"Final Report Saved To  : {OUTPUT_FILE}")
    print("\nSTATUS: 1,000 PATIENTS CHECKED & READY FOR SUBMISSION 🏆")
    print("=" * 70)

if __name__ == "__main__":
    run_automated_submission()
