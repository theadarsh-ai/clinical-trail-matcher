"""
Clinical Trial Matcher Environment
Core logic implementing the OpenEnv interface
"""

import uuid
from typing import Optional
from models import CTMatchAction, CTMatchObservation, CTMatchState, Patient, Trial
from server.data_generator import generate_episode
from server.graders import grade_action, get_ground_truth_eligible, get_ground_truth_ranking


class ClinicalTrialMatcherEnv:
    """
    OpenEnv-compatible environment for clinical trial matching
    
    Supports three tasks:
    - Easy: Basic eligibility matching
    - Medium: Medical criteria matching
    - Hard: Optimal ranking with logistics
    """
    
    def __init__(self):
        """Initialize the environment"""
        self._state: Optional[CTMatchState] = None
        self._current_patient: Optional[Patient] = None
        self._current_trials: list[Trial] = []
        self._episode_done = False
    
    def reset(self, task: str = "easy", seed: Optional[int] = None) -> CTMatchObservation:
        """
        Start a new episode
        
        Args:
            task: "easy", "medium", or "hard"
            seed: Random seed for reproducibility
        
        Returns:
            Initial observation
        """
        # Validate task
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid task: {task}. Must be 'easy', 'medium', or 'hard'")
        
        # Generate new episode
        episode_id = str(uuid.uuid4())
        patient, trials = generate_episode(episode_id, task=task, num_trials=8, seed=seed)
        
        # Compute ground truth
        ground_truth_eligible = get_ground_truth_eligible(patient, trials, task)
        ground_truth_ranking = get_ground_truth_ranking(patient, trials)
        
        # Initialize state
        self._state = CTMatchState(
            episode_id=episode_id,
            step_count=0,
            task=task,
            ground_truth_eligible=ground_truth_eligible,
            ground_truth_ranking=ground_truth_ranking,
            patient=patient,
            trials=trials
        )
        
        self._current_patient = patient
        self._current_trials = trials
        self._episode_done = False
        
        # Return initial observation
        return CTMatchObservation(
            patient=patient,
            trials=trials,
            task=task,
            message=f"Episode {episode_id} started. Task: {task}",
            done=False,
            reward=0.0,
            info={}
        )
    
    def step(self, action: CTMatchAction) -> CTMatchObservation:
        """
        Execute an action and return result
        
        Args:
            action: Agent's proposed trial matches/ranking
        
        Returns:
            Observation with reward and feedback
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self._episode_done:
            raise RuntimeError("Episode already finished. Call reset() to start new episode.")
        
        # Validate action task matches episode task
        if action.task != self._state.task:
            return CTMatchObservation(
                patient=self._current_patient,
                trials=self._current_trials,
                task=self._state.task,
                message=f"Error: Action task '{action.task}' does not match episode task '{self._state.task}'",
                done=True,
                reward=0.0,
                info={"error": "task_mismatch"}
            )
        
        # Grade the action
        grading_result = grade_action(
            patient=self._current_patient,
            trials=self._current_trials,
            task=self._state.task,
            proposed_trial_ids=action.proposed_trial_ids
        )
        
        # Update state
        self._state.step_count += 1
        self._episode_done = True
        
        # Prepare observation
        observation = CTMatchObservation(
            patient=self._current_patient,
            trials=self._current_trials,
            task=self._state.task,
            message=f"Episode completed. {grading_result['metric_name']}: {grading_result['reward']:.3f}",
            done=True,
            reward=grading_result["reward"],
            info={
                "ground_truth": grading_result["ground_truth"],
                "metric_name": grading_result["metric_name"],
                "details": grading_result["details"]
            }
        )
        
        return observation
    
    @property
    def state(self) -> Optional[CTMatchState]:
        """
        Get current environment state
        
        Returns:
            Current state or None if not initialized
        """
        return self._state
    
    def get_baseline_action(self) -> CTMatchAction:
        """
        Get baseline (ground truth) action for current episode
        Useful for testing and baseline inference
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self._state.task in ["easy", "medium"]:
            # Return ground truth eligible trials
            return CTMatchAction(
                action_type="list_eligible",
                task=self._state.task,
                proposed_trial_ids=self._state.ground_truth_eligible
            )
        else:  # hard
            # Return ground truth ranking
            return CTMatchAction(
                action_type="rank_trials",
                task=self._state.task,
                proposed_trial_ids=self._state.ground_truth_ranking
            )
    
    def get_tasks_info(self) -> dict:
        """
        Return information about available tasks and action schema
        Required for hackathon /tasks endpoint
        """
        return {
            "tasks": [
                {
                    "task_id": "easy",
                    "name": "easy",
                    "description": "Basic eligibility matching (age, gender, condition, city)",
                    "metric": "f1_score",
                    "action_type": "list_eligible"
                },
                {
                    "task_id": "medium",
                    "name": "medium",
                    "description": "Medical criteria matching (biomarkers, treatments, comorbidities)",
                    "metric": "f1_score",
                    "action_type": "list_eligible"
                },
                {
                    "task_id": "hard",
                    "name": "hard",
                    "description": "Optimal ranking (phase, distance, slots, preferences)",
                    "metric": "ndcg@10",
                    "action_type": "rank_trials"
                }
            ],
            "action_schema": CTMatchAction.model_json_schema()
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Clinical Trial Matcher Environment\n")
    
    # Test all three tasks
    env = ClinicalTrialMatcherEnv()
    
    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"Testing Task: {task.upper()}")
        print(f"{'='*60}")
        
        # Reset
        obs = env.reset(task=task, seed=42)
        print(f"\nReset: {obs.message}")
        print(f"Patient: {obs.patient.id}, Age {obs.patient.age}, {obs.patient.condition}")
        print(f"Trials: {len(obs.trials)}")
        
        # Get baseline action
        baseline_action = env.get_baseline_action()
        print(f"\nBaseline action: {baseline_action.action_type}")
        print(f"Proposed trials: {baseline_action.proposed_trial_ids}")
        
        # Step with baseline (should get perfect score)
        obs = env.step(baseline_action)
        print(f"\nStep result: {obs.message}")
        print(f"Reward: {obs.reward:.3f}")
        print(f"Ground truth: {obs.info.get('ground_truth', [])[:5]}...")
        
        # Test with partial action (should get lower score)
        env.reset(task=task, seed=42)
        if task in ["easy", "medium"]:
            # Predict only first trial
            test_action = CTMatchAction(
                action_type="list_eligible",
                task=task,
                proposed_trial_ids=[obs.trials[0].id]
            )
        else:  # hard
            # Reverse ranking
            test_action = CTMatchAction(
                action_type="rank_trials",
                task=task,
                proposed_trial_ids=list(reversed(baseline_action.proposed_trial_ids))
            )
        
        obs = env.step(test_action)
        print(f"\nTest action reward: {obs.reward:.3f}")
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")
