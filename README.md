# 🎧 Music Recommendation System using Reinforcement Learning

A personalized music recommendation engine built using Reinforcement Learning.  
The system interacts with users, learns their preferences through feedback,
and recommends songs they are likely to enjoy.

---

# 🚀 Features

✔ User registration & login  
✔ Personalized preference learning  
✔ Epsilon-greedy exploration for new songs  
✔ Adaptive recommendations based on interaction history  
✔ Genre & era-based feature representation  
✔ PDF Report included  

---

# 🧠 Reinforcement Learning Approach

The system represents each music track with a **21-dimensional feature vector**  
(genre, mood, decade, etc.).  
User preferences are learned from ratings on recommended songs.

A **utility function** and **ε-greedy strategy** drive exploration vs exploitation:

- Early phase: more exploration  
- Later phase: more exploitation (better personalized recommendations)

Cumulative reward improves over time → better recommendations.

---

# 📊 Performance Evaluation

Metrics used:

| Metric | What it Indicates |
|--------|------------------|
| **MSE** | Error between model-predicted preference & actual rating |
| **Spearman Rank** | Ranking quality of recommended songs |
| **Cumulative Reward** | Learning progress of the agent |

Model shows **increasing reward** & **better ranking** with training.

---

```md
# 📂 Project Structure
---

Music-Reinforcement-Recommendation-System/
├── main.py
├── requirements.txt
└── data/
    └── songs.csv
```




# ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```
