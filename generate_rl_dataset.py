"""
Large-scale data generator for RL training
Generates a dataset of 1000 clinical trial matching episodes
"""

import json
import random
from server.data_generator import generate_episode

def generate_rl_dataset(num_episodes=1000, output_file="rl_training_data.json"):
    print(f"Generating {num_episodes} RL episodes...")
    
    dataset = []
    tasks = ["easy", "medium", "hard"]
    
    for i in range(num_episodes):
        if i % 100 == 0:
            print(f"  Progress: {i}/{num_episodes}")
            
        task = random.choice(tasks)
        patient, trials = generate_episode(
            episode_id=f"rl_{i:04d}",
            task=task,
            num_trials=random.randint(5, 12),
            seed=random.randint(1, 1000000)
        )
        
        # Structure the episode for training
        episode_data = {
            "episode_id": f"rl_{i:04d}",
            "task": task,
            "patient": patient.dict(),
            "trials": [t.dict() for t in trials]
        }
        dataset.append(episode_data)
        
    print(f"Saving dataset to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print("Done! Dataset created successfully.")

if __name__ == "__main__":
    import sys
    num = 1000
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    generate_rl_dataset(num_episodes=num)
