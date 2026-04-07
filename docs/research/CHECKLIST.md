# 📋 Pre-Submission Checklist

Use this checklist to ensure your Clinical Trial Matcher environment meets all hackathon requirements before submission.

**Deadline: April 7, 2026, 11:59 PM IST**

---

## ✅ Required Files

- [ ] `models.py` - Pydantic models for Action, Observation, State
- [ ] `server/environment.py` - Core environment logic with reset/step/state
- [ ] `server/app.py` - FastAPI server with all endpoints
- [ ] `server/data_generator.py` - Synthetic data generation
- [ ] `server/graders.py` - F1 score and NDCG grading functions
- [ ] `server/requirements.txt` - Python dependencies
- [ ] `server/Dockerfile` - Container definition (port 8000)
- [ ] `Dockerfile` - Root Dockerfile for HF Spaces (port 7860)
- [ ] `inference.py` - **MANDATORY** inference script in root directory
- [ ] `baseline.py` - Baseline inference script
- [ ] `client.py` - Python client for environment
- [ ] `openenv.yaml` - OpenEnv configuration
- [ ] `README.md` - Comprehensive documentation
- [ ] `.env.example` - Environment variable template

---

## ✅ Functional Requirements

### Core OpenEnv Interface
- [ ] `reset()` method works for all tasks (easy, medium, hard)
- [ ] `step()` method accepts actions and returns observations
- [ ] `state()` method returns current episode state
- [ ] Typed Pydantic models for Action, Observation, State
- [ ] Episodes are single-step (one reset, one step, done)

### Tasks & Grading
- [ ] **Task 1 (Easy)**: Basic eligibility matching
  - Uses age, gender, condition, city criteria
  - Returns F1 score in [0.0, 1.0] range
  - Baseline achieves ~1.00 score
  
- [ ] **Task 2 (Medium)**: Medical criteria matching
  - Includes biomarkers, treatments, medications, comorbidities
  - Returns F1 score in [0.0, 1.0] range
  - Baseline achieves ~1.00 score
  
- [ ] **Task 3 (Hard)**: Optimal ranking
  - Considers phase, distance, slots, preferences
  - Returns NDCG@10 score in [0.0, 1.0] range
  - Baseline achieves ~1.00 score

### Reward Function
- [ ] Rewards are in [0.0, 1.0] range
- [ ] Partial progress is rewarded (not just binary)
- [ ] Reward function is deterministic and reproducible

---

## ✅ API Endpoints

### Core Endpoints (OpenEnv)
- [ ] `POST /reset` - Starts new episode
- [ ] `POST /step` - Executes action
- [ ] `GET /state` - Returns current state

### Hackathon Required Endpoints
- [ ] `POST /baseline` - Returns baseline scores for all 3 tasks
- [ ] `POST /grader` - Grades an episode with provided data
- [ ] `GET /tasks` - Returns task definitions and action schema

### Info Endpoints
- [ ] `GET /health` - Health check (returns 200)
- [ ] `GET /` - API information

---

## ✅ Inference Script Requirements

- [ ] File is named `inference.py` (not `baseline.py` or anything else)
- [ ] Located in **root directory** of project
- [ ] Uses `OpenAI` client for all LLM calls
- [ ] Reads environment variables:
  - `API_BASE_URL` - LLM endpoint
  - `MODEL_NAME` - Model identifier
  - `HF_TOKEN` - Hugging Face API key
  - `LOCAL_IMAGE_NAME` - Docker image name (if using containers)
  - `TASK_NAME` - Which task to run (easy/medium/hard)

### Required Output Format
- [ ] Emits `[START]` line at episode begin
- [ ] Emits `[STEP]` line per step
- [ ] Emits `[END]` line after completion
- [ ] All fields formatted correctly:
  - Rewards to 2 decimal places
  - Score to 3 decimal places
  - Booleans as lowercase (true/false)
  - Error as string or null

**Example:**
```
[START] task=easy env=clinical_trial_matcher model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=["trial_001","trial_003"] reward=0.85 done=true error=null
[END] success=true steps=1 score=0.850 rewards=0.85
```

---

## ✅ Docker & Deployment

### Docker Build
- [ ] `docker build -t clinical-trial-matcher -f Dockerfile .` succeeds
- [ ] Container runs: `docker run -p 7860:7860 clinical-trial-matcher`
- [ ] Health check endpoint accessible: `curl http://localhost:7860/health`
- [ ] Reset endpoint works: `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"easy"}'`

### Hugging Face Spaces
- [ ] Space created at `https://huggingface.co/spaces/YOUR_USERNAME/clinical-trial-matcher`
- [ ] Space is set to **Public** visibility
- [ ] Space SDK is set to **Docker**
- [ ] Space builds successfully (check logs)
- [ ] Space is **Running** (green status)
- [ ] Space URL accessible: `https://YOUR_USERNAME-clinical-trial-matcher.hf.space`

---

## ✅ Documentation

- [ ] README.md includes:
  - Problem statement and motivation
  - Task descriptions (easy/medium/hard)
  - Quick start guide
  - API endpoint documentation
  - Usage examples
  - Project structure
  - Installation instructions
  
- [ ] DEPLOYMENT.md includes:
  - Step-by-step deployment guide
  - Troubleshooting section
  - Validation instructions

- [ ] Code is well-commented
- [ ] Docstrings for all major functions

---

## ✅ Testing & Validation

### Local Testing
- [ ] Run `python validate.py` - all 8 checks pass
- [ ] Run `python baseline.py` - achieves 1.000 average score
- [ ] Run `python test_inference.py` - format checks pass
- [ ] Test all API endpoints manually with curl

### Environment Variables
- [ ] Created `.env` file from `.env.example`
- [ ] Set `HF_TOKEN` to valid token
- [ ] All required variables documented

### Baseline Performance
- [ ] Easy task: Score ≥ 0.80 (target: 1.00)
- [ ] Medium task: Score ≥ 0.70 (target: 1.00)
- [ ] Hard task: Score ≥ 0.60 (target: 1.00)
- [ ] Average score ≥ 0.70

---

## ✅ Submission Validation

### Pre-Submission Validator Script
- [ ] Downloaded validation script from OpenEnv repo
- [ ] Made executable: `chmod +x validate-submission.sh`
- [ ] Ran validator: `./validate-submission.sh https://YOUR_SPACE_URL .`
- [ ] All 3 checks passed:
  - ✅ HF Space is live and responds to /reset
  - ✅ Docker build succeeded
  - ✅ openenv validate passed

### Manual Endpoint Testing
Test each endpoint on deployed Space:

```bash
SPACE_URL="https://YOUR_USERNAME-clinical-trial-matcher.hf.space"

# Health check
curl $SPACE_URL/health

# Reset
curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'

# Baseline
curl -X POST $SPACE_URL/baseline

# Tasks
curl $SPACE_URL/tasks

# Step (after reset)
curl -X POST $SPACE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "list_eligible", "task": "easy", "proposed_trial_ids": ["trial_001"]}}'
```

---

## ✅ Final Submission

- [ ] Copied Space URL: `https://YOUR_USERNAME-clinical-trial-matcher.hf.space`
- [ ] Prepared team information
- [ ] Written project description (150-300 words)
- [ ] Listed team members
- [ ] Noted any special features or innovations
- [ ] Submitted via hackathon portal **before April 7, 11:59 PM IST**
- [ ] Received confirmation email/notification

---

## 📝 Submission Information Template

**Team Name:** Your Team Name

**Space URL:** https://YOUR_USERNAME-clinical-trial-matcher.hf.space

**GitHub Repository:** (Optional) https://github.com/YOUR_USERNAME/clinical-trial-matcher

**Project Description:**
```
Clinical Trial Matcher is an OpenEnv-compatible reinforcement learning 
environment that addresses the real-world problem of matching patients to 
suitable clinical trials. The environment features three progressive tasks 
(easy, medium, hard) that challenge AI agents to handle increasingly complex 
eligibility criteria including demographics, biomarkers, medical history, and 
logistical constraints. With F1 and NDCG scoring metrics, this environment 
provides a meaningful testbed for training agents to solve a critical 
healthcare infrastructure challenge that affects millions of patients worldwide.
```

**Special Features:**
- Synthetic data generator for realistic patient and trial profiles
- Three-tier difficulty system with clear progression
- Supports both classification (F1) and ranking (NDCG) tasks
- Production-ready Docker deployment
- Comprehensive documentation and examples

**Team Members:**
1. Your Name - Role
2. (Add more as needed)

---

## 🎯 Success Criteria

Your submission is ready when ALL of these are true:

✅ All required files present  
✅ All 3 tasks implemented with working graders  
✅ Baseline script achieves 1.00 average score  
✅ Inference script outputs correct format  
✅ Docker builds and runs successfully  
✅ Deployed to HF Spaces and accessible  
✅ Validation script passes all checks  
✅ Documentation is comprehensive  
✅ Submitted before deadline  

---

## 🚀 Post-Submission

After submitting:

- [ ] Monitor Space for any downtime or errors
- [ ] Join Discord to discuss with other participants
- [ ] Prepare for potential follow-up questions from judges
- [ ] Consider improvements for finale round
- [ ] Share your work on social media (optional)

---

**Questions or Issues?**

- Discord: https://discord.gg/Dedhy5pkWD
- Email: help_openenvhackathon@scaler.com
- Documentation: https://meta-pytorch.org/OpenEnv/

---

**Good luck! 🍀**

Remember: Quality over quantity. A well-tested, well-documented environment 
is better than a complex one with bugs.
