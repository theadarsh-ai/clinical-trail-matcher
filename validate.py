"""
Pre-Submission Validation Script
Checks that all hackathon requirements are met
"""

import sys
import json
import requests
import time
from server.environment import ClinicalTrialMatcherEnv


def validate_environment():
    """Run all validation checks"""
    print("="*70)
    print("CLINICAL TRIAL MATCHER - PRE-SUBMISSION VALIDATION")
    print("="*70)
    
    checks_passed = 0
    checks_failed = 0
    
    # Check 1: Environment can be instantiated
    print("\n[1/8] Testing environment instantiation...")
    try:
        env = ClinicalTrialMatcherEnv()
        print("[PASS] Environment created successfully")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Failed to create environment: {e}")
        checks_failed += 1
        return
    
    # Check 2: Reset works for all tasks
    print("\n[2/8] Testing reset for all tasks...")
    try:
        for task in ["easy", "medium", "hard"]:
            obs = env.reset(task=task, seed=42)
            assert obs.task == task
            assert not obs.done
            assert len(obs.trials) > 0
        print("[PASS] Reset works for easy, medium, and hard tasks")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Reset failed: {e}")
        checks_failed += 1
    
    # Check 3: Step works and returns reward
    print("\n[3/8] Testing step execution...")
    try:
        env.reset(task="easy", seed=42)
        action = env.get_baseline_action()
        obs = env.step(action)
        assert obs.done == True
        assert 0.0 <= obs.reward <= 1.0
        print(f"[PASS] Step execution works (reward: {obs.reward:.3f})")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Step failed: {e}")
        checks_failed += 1
    
    # Check 4: Grading functions work
    print("\n[4/8] Testing grading functions...")
    try:
        scores = {}
        for task in ["easy", "medium", "hard"]:
            env.reset(task=task, seed=42)
            action = env.get_baseline_action()
            obs = env.step(action)
            scores[task] = obs.reward
        
        # Baseline should achieve perfect or near-perfect scores
        assert all(score >= 0.95 for score in scores.values())
        print(f"[PASS] Grading works (scores: {scores})")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Grading failed: {e}")
        checks_failed += 1
    
    # Check 5: Baseline script runs
    print("\n[5/8] Testing baseline script...")
    try:
        import baseline
        results = baseline.run_baseline_inference(seed=42)
        assert all(results[task]["reward"] >= 0.95 for task in ["easy", "medium", "hard"])
        print(f"[PASS] Baseline script works (avg: {sum(r['reward'] for r in results.values()) / 3:.3f})")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Baseline script failed: {e}")
        checks_failed += 1
    
    # Check 6: Tasks info endpoint
    print("\n[6/8] Testing tasks info...")
    try:
        tasks_info = env.get_tasks_info()
        assert "tasks" in tasks_info
        assert "action_schema" in tasks_info
        assert len(tasks_info["tasks"]) == 3
        print(f"[PASS] Tasks info available ({len(tasks_info['tasks'])} tasks)")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Tasks info failed: {e}")
        checks_failed += 1
    
    # Check 7: File structure
    print("\n[7/8] Checking file structure...")
    try:
        import os
        required_files = [
            "models.py",
            "client.py",
            "baseline.py",
            "openenv.yaml",
            "README.md",
            "server/__init__.py",
            "server/environment.py",
            "server/app.py",
            "server/data_generator.py",
            "server/graders.py",
            "server/requirements.txt",
            "server/Dockerfile"
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            print(f"[FAIL] Missing files: {missing}")
            checks_failed += 1
        else:
            print(f"[PASS] All required files present ({len(required_files)} files)")
            checks_passed += 1
    except Exception as e:
        print(f"[FAIL] File check failed: {e}")
        checks_failed += 1
    
    # Check 8: Documentation
    print("\n[8/8] Checking documentation...")
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme = f.read()
        
        required_sections = ["Problem", "Quick Start", "Usage", "API", "Tasks"]
        missing_sections = [s for s in required_sections if s.lower() not in readme.lower()]
        
        if missing_sections:
            print(f"[WARN] README might be missing sections: {missing_sections}")
        
        if len(readme) > 1000:
            print(f"[PASS] README exists and is comprehensive ({len(readme)} chars)")
            checks_passed += 1
        else:
            print(f"[WARN] README might be too short ({len(readme)} chars)")
            checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Documentation check failed: {e}")
        checks_failed += 1
    
    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"[PASS] Passed: {checks_passed}/8")
    print(f"[FAIL] Failed: {checks_failed}/8")
    
    if checks_failed == 0:
        print("\nAll checks passed! Environment is ready for submission.")
        print("\nNext steps:")
        print("1. Test Docker build: docker build -t clinical-trial-matcher -f server/Dockerfile .")
        print("2. Test Docker run: docker run -p 8000:8000 clinical-trial-matcher")
        print("3. Deploy to Hugging Face Spaces")
        print("4. Submit your HF Spaces URL to the hackathon")
        return True
    else:
        print("\n[WARN] Some checks failed. Please fix the issues before submission.")
        return False


if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
