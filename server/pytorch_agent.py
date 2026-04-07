import torch
import torch.nn as nn
import torch.optim as optim
from server.environment import ClinicalTrialMatcherEnv
from models import CTMatchAction
from server.graders import get_ground_truth_eligible

# 1. The Neural Network Model
class RealCTMAgent(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        # A simple Multi-Layer Perceptron (MLP) mapping trial features to "Eligibility Probability"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs probability between 0 and 1
        )

    def forward(self, x):
        return self.net(x)

def extract_features(patient, trial, task):
    """Convert the raw Medical Data into numerical Tensors for the Neural Net"""
    age_match = 1.0 if trial.min_age <= patient.age <= trial.max_age else -1.0
    city_match = 1.0 if patient.city in trial.allowed_cities else -1.0
    gender_match = 1.0 if patient.gender in trial.allowed_genders else -1.0
    
    # Biomarker match (proxy simplified for matrix math speed)
    bio_match = 1.0
    if task in ["medium", "hard"]:
        for k, v in trial.required_biomarkers.items():
            if patient.biomarkers.get(k) != v:
                bio_match = -1.0
                break
                
    return [age_match, city_match, gender_match, bio_match]

def run_real_pytorch_training(task="medium", max_episodes=300):
    """
    Executes a literal PyTorch REINFORCE (Policy Gradient) loop.
    Runs until it achieves a nearly perfect reward (convergence), or hits max limit.
    """
    env = ClinicalTrialMatcherEnv()
    agent = RealCTMAgent()
    
    # Using Adam Optimizer to update the neural network weights
    optimizer = optim.Adam(agent.parameters(), lr=0.1)
    
    metrics = []
    success_streak = 0
    
    for ep in range(1, max_episodes + 1):
        obs = env.reset(task=task)
        patient = obs.patient
        trials = obs.trials
        
        # Ensure we have 8 trials exactly for consistency
        trials = trials[:8] 
        
        # 1. Forward Pass (Create State Tensors)
        action_probs = []
        action_log_probs = []
        proposed_trials_with_prob = []
        
        for trial in trials:
            # Extract features for this specific Trial/Patient pairs
            features = torch.tensor(extract_features(patient, trial, task), dtype=torch.float32)
            
            # Neural Net predicts Eligibility Probability
            prob = agent(features)
            
            # Sample action (1 = Accept Trial, 0 = Reject Trial)
            # We use Bernoulli so the network "explores" randomly at first
            m = torch.distributions.Bernoulli(prob)
            action = m.sample()
            
            # Save the log probability for the Loss calculation later
            action_log_probs.append(m.log_prob(action))
            
            if action.item() == 1.0:
                proposed_trials_with_prob.append((trial.id, prob.item()))
                
        # For the 'hard' task, the environment expects a RANKED list.
        # We sort by the highest neural network probability first.
        if task == "hard":
            proposed_trials_with_prob.sort(key=lambda x: x[1], reverse=True)
            
        proposed_trial_ids = [t_id for t_id, p in proposed_trials_with_prob]

        # 2. Step Environment (Action)
        step_action = CTMatchAction(
            action_type="rank_trials" if task == "hard" else "list_eligible", 
            task=task,
            proposed_trial_ids=proposed_trial_ids
        )
        result = env.step(step_action)
        reward = result.reward  # Objective grading from OpenEnv
        
        # 3. Backpropagation (The Learning)
        # We calculate the Loss. If reward is HIGH, we make the log_probs more likely in the future.
        # If reward is LOW, we penalize those probabilities.
        loss_val = 0.0
        if len(action_log_probs) > 0:
            loss = 0
            for log_prob in action_log_probs:
                # Basic REINFORCE object: -log(π) * R
                # We subtract 0.5 as a baseline to punish bad rewards
                loss += -log_prob * (reward - 0.5) 
                
            loss = loss / len(action_log_probs)
            loss_val = float(loss.item())
            
            optimizer.zero_grad()
            loss.backward()  # Compute gradients magically
            optimizer.step() # Update neuronal weights!

        # For the UI, we capture the data step by step
        ground_truth = get_ground_truth_eligible(patient, trials, task)
        
        # We only send large payload objects on the very LAST episode to save network bandwidth, 
        # or we just send it every time since it's local. Let's send the basics.
        metrics.append({
            "episode": ep,
            "patient_condition": patient.condition.replace("_", " "),
            "reward": float(reward),
            "loss": loss_val,
            "proposed_count": len(proposed_trial_ids),
            "truth_count": len(ground_truth),
            
            # UI display data
            "patient_full": {
                "condition": patient.condition,
                "age": patient.age,
                "gender": patient.gender,
                "city": patient.city,
                "biomarkers": patient.biomarkers
            },
            "trials_summary": [f"[{t.id[-6:]}] {t.title[:30]}..." for t in trials[:3]],
            "trials_total": len(trials),
            "proposed_trials": [tid[-6:] for tid in proposed_trial_ids]
        })
        
        # Early Stopping: Break if we hit a high reward consecutively 
        if reward > 0.90:
            success_streak += 1
            if success_streak >= 3:
                # Model converged! Save the trained weights to disk.
                import os
                save_path = os.path.join(os.path.dirname(__file__), "..", "clinical_trial_model_weights.pt")
                torch.save(agent.state_dict(), save_path)
                metrics[-1]["saved_path"] = save_path
                break
        else:
            success_streak = 0
            
    return metrics
