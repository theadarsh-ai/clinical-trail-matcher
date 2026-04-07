# 🎤 Clinical Trial Matcher - Hackathon Presentation Guide

This guide will help you "wow" the judges by explaining the Reinforcement Learning (RL) logic and showing off the scale of your project.

---

## 🏗️ 1. The Opening (The "Idea")
**What to say:**
> "Hi, I'm [Your Name]. I built an OpenEnv-compatible Reinforcement Learning environment to solve the critical problem of **Clinical Trial Matching**. 
> 
> Most trials fail because they can't find the right patients. My system uses an RL Agent to bridge that gap by scanning 50+ medical criteria in seconds."

---

## 🔬 2. The Live Demo (Run tasks separately)
**What to show:**
Open two terminals. In Terminal 1, run the server. In Terminal 2, run the `inference.py` script.

**Action: Show the Easy Task**
```powershell
$env:TASK_NAME='easy'; python inference.py
```
**Say:** *"In the **Easy** task, the AI matches patients based on basic demographics like Age, Gender, and City. You can see the Reward is high because it matched perfectly."*

**Action: Show the Hard Task**
```powershell
$env:TASK_NAME='hard'; python inference.py
```
**Say:** *"In the **Hard** task, the AI has to perform **Ranking**. It doesn't just find matches; it finds the **Best** ones considering travel distance and clinical phase. This is the core of our RL Reward system."*

---

## 📊 3. The "Big Scale" Proof (Show the 1,000 Patients)
**What to show:**
Run the automation script to show the judges the scale of your work.

**Action: Run the Final Submission**
```powershell
python run_final_submission.py
```
**Say:** *"But we didn't just stop at one patient. We evaluated our system across a population of **1,000 unique patients** and **8,000 trials**. 
By simulating the entire 1,000-patient RL journey, we've proven that the system is stable, scalable, and ready for real-world healthcare infrastructure."*

---

## 🏆 4. The Conclusion
**What to say:**
> "By combining Pydantic-validated medical models with Reinforcement Learning, we've created a system that can automate the future of clinical enrollment. 
> 
> The project is fully Dockerized and deployed on Hugging Face Spaces for the community to explore. Thank you!"

---

## 🛠️ Technical Highlights for Judges
If they ask technical questions, mention these 3 things:
1.  **OpenEnv Standard**: Fully compatible with Meta's Pytorch Reinforcement Learning framework.
2.  **Metrics**: Uses **F1 Score** for matching and **NDCG** for ranking quality.
3.  **Pydantic V2**: Uses modern Data Models for 100% type-safe medical data.
