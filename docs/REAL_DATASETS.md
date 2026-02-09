# Real-World Datasets for Cognitive State Inference

## 📊 Recommended Datasets (Publicly Available)

### 1. **BEST CHOICE: Keystroke Dynamics Dataset**
**Source:** Carnegie Mellon University / Kaggle
**URL:** https://www.cs.cmu.edu/~keystroke/

**Why Perfect for FocusLab:**
- ✅ Contains typing patterns that correlate with fatigue/focus
- ✅ Has timing data (similar to reaction times)
- ✅ Includes error patterns
- ✅ Multiple users for cross-validation
- ✅ Real behavioral data

**Features Available:**
- Key hold time (dwell time)
- Key-to-key time (flight time)
- Error rates
- Typing speed variability
- Session duration

**Download:**
```bash
# Available on Kaggle
kaggle datasets download -d imsparsh/keystroke-dynamics-dataset
```

### 2. **Mouse Movement & Cognitive Load**
**Source:** University of Maryland / UCI ML Repository
**URL:** https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling

**Features:**
- Mouse trajectory smoothness
- Movement speed
- Pause durations
- Click accuracy
- Task completion time

**Good for:** Detecting distraction and cognitive load

### 3. **N-Back Task Performance Data**
**Source:** OpenNeuro / Cognitive Atlas
**URL:** https://openneuro.org/

**What it is:** Working memory task where users must remember items N steps back
**Features:**
- Reaction times
- Accuracy/error rates
- Response variability
- Session-level fatigue indicators

**Perfect for:** Mental fatigue detection

### 4. **Driver Alertness Dataset**
**Source:** PhysioNet / University of Iowa
**URL:** https://physionet.org/content/drivedb/1.0.0/

**Features:**
- Reaction times to stimuli
- Lane deviation (proxy for distraction)
- Response accuracy
- Time-of-day effects
- Fatigue indicators

**Use Case:** Fatigue vs alertness classification

### 5. **Continuous Performance Task (CPT) Data**
**Source:** ADHD-200 Consortium
**URL:** http://fcon_1000.projects.nitrc.org/indi/adhd200/

**Features:**
- Sustained attention metrics
- Response times
- Commission/omission errors
- Variability measures

**Good for:** Attention states (focused vs distracted)

---

## 🚀 IMPLEMENTATION PLAN

### Phase 1: Easy Integration (Start Today)
Use **Keystroke Dynamics** - most similar to our synthetic data structure

### Phase 2: Advanced Integration
Add **N-Back** or **CPT** data for richer cognitive state labels

---

## Dataset Feature Mapping to Our Model

| Dataset Feature | Maps to Our Feature | State Correlation |
|----------------|-------------------|------------------|
| Key hold time (ms) | reaction_time | Longer = Fatigued |
| Error rate | error_rate | Higher = Distracted |
| Typing speed variance | reaction_time_std | Higher = Fatigued |
| Pause frequency | - | Higher = Distracted |
| Session time | time_of_day | Late = Fatigued |

---

## Quick Start: Keystroke Dataset

### Step 1: Download
```bash
# Manual download from:
# https://www.cs.cmu.edu/~keystroke/
# Or use this alternative: DSL-StrongPasswordData.csv

# Alternative: Generate from typing.com API (free)
```

### Step 2: Data Structure Expected
```csv
user_id,timestamp,hold_time_ms,flight_time_ms,error_count,session_duration
user_001,2024-01-01 10:00:00,145,280,0,300
user_001,2024-01-01 10:00:01,152,290,1,301
...
```

### Step 3: Map to Our Features
```python
# hold_time_ms → reaction_time
# error_count / total_keystrokes → error_rate
# flight_time_ms → additional feature
```

---

## 📁 Alternative: Public Datasets We Can Use Immediately

### Option A: Simulated But Validated
**PhysioNet Alertness Database**
- Real driving simulator data
- Validated fatigue labels
- Download: `wget -r -np https://physionet.org/files/drivedb/1.0.0/`

### Option B: Academic Research Data
**UCI Human Activity Recognition**
- Not cognitive directly, but has timing patterns
- URL: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

### Option C: Gaming Performance Data
**Kaggle: League of Legends Diamond Ranked Games**
- Reaction times in high-stakes scenarios
- Performance degradation over time
- URL: https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min

---

## 🎯 IMMEDIATE ACTION: Best Dataset for You

**RECOMMENDATION: Start with typing/keystroke data**

I'll show you two approaches:

### Approach 1: Use Existing Public Dataset
- CMU Keystroke Dynamics
- Pre-labeled with user sessions
- Direct mapping to reaction_time + error_rate

### Approach 2: Collect Your Own (Quick!)
- Use simple typing test
- Record yourself for 30 minutes
- Simulate different states (focused morning, tired evening)

---

## Next Steps

Which dataset interests you most? I'll help you:
1. Download and preprocess it
2. Map features to our HMM
3. Create a hybrid dataset (50% real + 50% synthetic)
4. Retrain model with real data
5. Compare performance

Let me know which dataset you want to start with!
