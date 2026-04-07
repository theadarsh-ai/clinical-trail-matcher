"""
Baseline Inference Script for Clinical Trial Matcher
Demonstrates how to use the environment and achieves baseline scores
"""

import sys
import json
from server.environment import ClinicalTrialMatcherEnv
from models import CTMatchAction


def run_baseline_inference(seed: int = 42):
    """
    Run baseline inference for all tasks
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of scores for each task
    """
    env = ClinicalTrialMatcherEnv()
    results = {}
    
    print("="*60)
    print("Clinical Trial Matcher - Baseline Inference")
    print("="*60)
    
    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print(f"{'='*60}")
        
        # Reset environment
        obs = env.reset(task=task, seed=seed)
        print(f"\nEpisode ID: {env.state.episode_id}")
        print(f"Patient: {obs.patient.id}")
        print(f"  Age: {obs.patient.age}")
        print(f"  Gender: {obs.patient.gender}")
        print(f"  Condition: {obs.patient.condition}")
        print(f"  City: {obs.patient.city}")
        print(f"  Biomarkers: {obs.patient.biomarkers}")
        print(f"  Prior treatments: {obs.patient.prior_treatments}")
        print(f"Number of trials: {len(obs.trials)}")
        
        # Get baseline action (ground truth)
        baseline_action = env.get_baseline_action()
        
        print(f"\nBaseline Strategy:")
        print(f"  Action type: {baseline_action.action_type}")
        print(f"  Proposed trials: {len(baseline_action.proposed_trial_ids)}")
        if len(baseline_action.proposed_trial_ids) <= 5:
            print(f"  Trial IDs: {baseline_action.proposed_trial_ids}")
        else:
            print(f"  Trial IDs (first 5): {baseline_action.proposed_trial_ids[:5]}")
        
        # Execute action
        obs = env.step(baseline_action)
        
        print(f"\nResults:")
        print(f"  Reward: {obs.reward:.4f}")
        print(f"  Metric: {obs.info.get('metric_name', 'unknown')}")
        print(f"  Message: {obs.message}")
        
        # Store results
        results[task] = {
            "reward": obs.reward,
            "metric": obs.info.get("metric_name"),
            "details": obs.info.get("details", {})
        }
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task, result in results.items():
        print(f"{task.upper()}: {result['reward']:.4f} ({result['metric']})")
    
    print(f"\nAverage Score: {sum(r['reward'] for r in results.values()) / len(results):.4f}")
    
    return results


def test_custom_actions():
    """Test with custom (non-baseline) actions to show scoring"""
    env = ClinicalTrialMatcherEnv()
    
    print(f"\n{'='*60}")
    print("Testing Custom Actions")
    print(f"{'='*60}")
    
    # Test Task: Easy
    print("\n--- Easy Task: Predict half the trials ---")
    obs = env.reset(task="easy", seed=42)
    baseline = env.get_baseline_action()
    
    # Take only first half of correct answers
    partial_action = CTMatchAction(
        action_type="list_eligible",
        task="easy",
        proposed_trial_ids=baseline.proposed_trial_ids[:len(baseline.proposed_trial_ids)//2]
    )
    
    obs = env.step(partial_action)
    print(f"Partial prediction reward: {obs.reward:.4f}")
    print(f"Predicted {len(partial_action.proposed_trial_ids)} out of {len(baseline.proposed_trial_ids)} trials")
    
    # Test Task: Hard (reversed ranking)
    print("\n--- Hard Task: Reversed ranking ---")
    obs = env.reset(task="hard", seed=42)
    baseline = env.get_baseline_action()
    
    # Reverse the ranking
    reversed_action = CTMatchAction(
        action_type="rank_trials",
        task="hard",
        proposed_trial_ids=list(reversed(baseline.proposed_trial_ids))
    )
    
    obs = env.step(reversed_action)
    print(f"Reversed ranking reward: {obs.reward:.4f}")
    print(f"(Lower score because ranking is reversed)")


if __name__ == "__main__":
    # Parse command line arguments
    seed = 42
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print(f"Invalid seed: {sys.argv[1]}. Using default seed=42")
    
    # Run baseline inference
    results = run_baseline_inference(seed=seed)
    
    # Test custom actions
    test_custom_actions()
    
    # Output results as JSON (for automated evaluation)
    print(f"\n{'='*60}")
    print("JSON Output (for automated evaluation):")
    print(f"{'='*60}")
    print(json.dumps({
        "baseline_scores": {task: r["reward"] for task, r in results.items()},
        "average_score": sum(r["reward"] for r in results.values()) / len(results),
        "seed": seed
    }, indent=2))
