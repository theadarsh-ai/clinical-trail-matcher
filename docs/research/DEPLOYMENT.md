# Deployment Guide for Clinical Trial Matcher

This guide walks you through deploying your Clinical Trial Matcher environment to Hugging Face Spaces for the hackathon submission.

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] All files in the `clinical_trial_matcher/` directory
- [ ] Docker installed and working
- [ ] Hugging Face account created
- [ ] Git installed
- [ ] All validation tests passed (`python validate.py`)

## 🚀 Step 1: Test Locally

### 1.1 Run Baseline Test
```bash
cd clinical_trial_matcher
python baseline.py
```

**Expected output:**
```
EASY: 1.0000 (f1_score)
MEDIUM: 1.0000 (f1_score)
HARD: 1.0000 (ndcg@10)
Average Score: 1.0000
```

### 1.2 Test Inference Script
```bash
# Set environment variables
export HF_TOKEN=your_token_here
export TASK_NAME=easy

# Run inference
python inference.py
```

**Expected output:**
```
[START] task=easy env=clinical_trial_matcher model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=["trial_..."] reward=X.XX done=true error=null
[END] success=true steps=1 score=X.XXX rewards=X.XX
```

### 1.3 Test Docker Build
```bash
# Build the image
docker build -t clinical-trial-matcher -f server/Dockerfile .

# Run the container
docker run -p 8000:8000 clinical-trial-matcher

# In another terminal, test the API
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'
```

## 🤗 Step 2: Create Hugging Face Space

### 2.1 Create New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `clinical-trial-matcher` (or your chosen name)
   - **License**: MIT
   - **Space SDK**: Docker
   - **Visibility**: Public
   - **Space hardware**: CPU basic (free tier works fine)

3. Click **Create Space**

### 2.2 Clone Your Space Repository

```bash
# Clone the empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/clinical-trial-matcher
cd clinical-trial-matcher
```

## 📦 Step 3: Prepare Files for Deployment

### 3.1 Copy Files to Space Directory

```bash
# Copy all necessary files
cp -r /path/to/clinical_trial_matcher/* .

# Ensure these files are present:
# - models.py
# - server/ (directory with all server files)
# - inference.py
# - baseline.py
# - client.py
# - openenv.yaml
# - README.md
# - .env.example
```

### 3.2 Create Hugging Face Space Configuration

Create a file named `README.md` in the root (this is your Space's landing page):

```yaml
---
title: Clinical Trial Matcher
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - clinical-trials
  - reinforcement-learning
  - healthcare
  - patient-matching
---

# Clinical Trial Matcher Environment

OpenEnv environment for matching patients to clinical trials using AI agents.

[View Full Documentation](./README.md)
```

### 3.3 Ensure Dockerfile is Correct

Your `server/Dockerfile` should be at `./server/Dockerfile` (already created).

For HF Spaces, you might want to create a root-level Dockerfile that points to the server:

```dockerfile
# Root Dockerfile for HF Spaces
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/

# Install Python packages
RUN pip install --no-cache-dir -r server/requirements.txt

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application on port 7860 (HF Spaces default)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## 🚢 Step 4: Deploy to Hugging Face

### 4.1 Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment of Clinical Trial Matcher environment"

# Push to HF Spaces
git push
```

### 4.2 Wait for Build

- Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/clinical-trial-matcher`
- Watch the build logs (click "Logs" tab)
- Wait for "Running on" message
- The Space will be available at: `https://YOUR_USERNAME-clinical-trial-matcher.hf.space`

## ✅ Step 5: Validate Deployment

### 5.1 Test the Deployed Space

```bash
# Test reset endpoint
curl -X POST https://YOUR_USERNAME-clinical-trial-matcher.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'

# Test baseline endpoint
curl -X POST https://YOUR_USERNAME-clinical-trial-matcher.hf.space/baseline

# Test tasks endpoint
curl https://YOUR_USERNAME-clinical-trial-matcher.hf.space/tasks
```

### 5.2 Run Validation Script

```bash
# Download validation script
curl -fsSL https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/scripts/validate-submission.sh -o validate-submission.sh
chmod +x validate-submission.sh

# Run validation
./validate-submission.sh https://YOUR_USERNAME-clinical-trial-matcher.hf.space .
```

**Expected output:**
```
[PASSED] -- HF Space is live and responds to /reset
[PASSED] -- Docker build succeeded
[PASSED] -- openenv validate passed
All 3/3 checks passed!
Your submission is ready to submit.
```

## 📝 Step 6: Submit to Hackathon

### 6.1 Submission Information

Go to the hackathon submission form and provide:

1. **Team Name**: Your team name
2. **Space URL**: `https://YOUR_USERNAME-clinical-trial-matcher.hf.space`
3. **GitHub Repository** (optional): Link to your code repository
4. **Description**: 
   ```
   Clinical Trial Matcher - An OpenEnv environment that helps AI agents learn 
   to match patients to suitable clinical trials based on complex eligibility 
   criteria. Includes three progressive tasks (easy/medium/hard) with F1 and 
   NDCG scoring metrics.
   ```

### 6.2 Final Checklist

Before submitting, verify:

- [ ] Space is publicly accessible
- [ ] `/reset` endpoint works
- [ ] `/baseline` endpoint returns scores for all 3 tasks
- [ ] `/tasks` endpoint returns task definitions
- [ ] `/grader` endpoint works
- [ ] `inference.py` is in the root directory
- [ ] README.md is comprehensive
- [ ] Dockerfile builds successfully
- [ ] Validation script passes all checks

## 🐛 Troubleshooting

### Problem: Space won't build

**Solution**: Check the build logs in HF Spaces. Common issues:
- Missing dependencies in `requirements.txt`
- Wrong Dockerfile path
- Port conflicts (HF Spaces uses port 7860 by default)

### Problem: `/reset` returns error

**Solution**: Check that:
- The server is running on port 7860 (not 8000)
- All imports are working correctly
- Models are properly defined

### Problem: Validation script fails

**Solution**:
1. Test each endpoint manually with `curl`
2. Check server logs for errors
3. Ensure Dockerfile builds locally first
4. Verify all required files are present

## 📞 Getting Help

If you encounter issues:

1. Check the [OpenEnv Documentation](https://meta-pytorch.org/OpenEnv/)
2. Join the [Hackathon Discord](https://discord.gg/Dedhy5pkWD)
3. Email: help_openenvhackathon@scaler.com
4. Check the [FAQ](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#faqs)

## 🎉 Post-Submission

After submitting:

1. Monitor your Space for any issues
2. Be ready to answer questions from judges
3. Consider adding more features for the finale round
4. Share your submission on social media!

---

**Good luck! 🍀**

Remember: The deadline is **April 7, 2026, 11:59 PM IST**
