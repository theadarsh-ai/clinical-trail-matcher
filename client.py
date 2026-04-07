"""
Client for Clinical Trial Matcher Environment
Provides easy interface for connecting to the environment server
"""

import requests
from typing import Optional
from models import CTMatchAction, CTMatchObservation


class ClinicalTrialMatcherClient:
    """
    Client for interacting with Clinical Trial Matcher environment
    
    Usage:
        client = ClinicalTrialMatcherClient("http://localhost:7860")
        obs = client.reset(task="easy")
        action = CTMatchAction(...)
        obs = client.step(action)
    """
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        """
        Initialize client
        
        Args:
            base_url: URL of the environment server
        """
        self.base_url = base_url.rstrip("/")
    
    def reset(self, task: str = "easy", seed: Optional[int] = None) -> CTMatchObservation:
        """
        Reset the environment
        
        Args:
            task: "easy", "medium", or "hard"
            seed: Optional random seed
        
        Returns:
            Initial observation
        """
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task": task, "seed": seed}
        )
        response.raise_for_status()
        return CTMatchObservation(**response.json())
    
    def step(self, action: CTMatchAction) -> CTMatchObservation:
        """
        Execute an action
        
        Args:
            action: Agent's action
        
        Returns:
            Observation with reward
        """
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.dict()}
        )
        response.raise_for_status()
        return CTMatchObservation(**response.json())
    
    def get_state(self) -> dict:
        """Get current state"""
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()
    
    def get_tasks(self) -> dict:
        """Get available tasks and action schema"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()
    
    def run_baseline(self) -> dict:
        """Run baseline inference"""
        response = requests.post(f"{self.base_url}/baseline")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Connect to local server
    client = ClinicalTrialMatcherClient("http://localhost:8000")
    
    # Health check
    print("Health check:", client.health_check())
    
    # Get tasks info
    print("\nAvailable tasks:", client.get_tasks())
    
    # Test easy task
    print("\n" + "="*60)
    print("Testing Easy Task")
    print("="*60)
    
    obs = client.reset(task="easy", seed=42)
    print(f"Episode started: {obs.message}")
    print(f"Patient: {obs.patient.id}, Age {obs.patient.age}")
    print(f"Trials: {len(obs.trials)}")
    
    # Create a simple action (predict first 3 trials)
    action = CTMatchAction(
        action_type="list_eligible",
        task="easy",
        proposed_trial_ids=[trial.id for trial in obs.trials[:3]]
    )
    
    obs = client.step(action)
    print(f"Result: {obs.message}")
    print(f"Reward: {obs.reward:.3f}")
    
    # Run baseline
    print("\n" + "="*60)
    print("Running Baseline")
    print("="*60)
    baseline_results = client.run_baseline()
    print(f"Baseline scores: {baseline_results['scores']}")
