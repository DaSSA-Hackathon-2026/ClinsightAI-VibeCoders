# ClinsightAI: AI-Powered Healthcare Review Intelligence

ClinsightAI transforms raw hospital reviews into decision intelligence for clinic owners and operations teams.  
It detects operational themes, quantifies their rating impact, identifies recurring/systemic issues, and produces a prioritized action roadmap with evidence and confidence.

## 1) Problem Framing
### Problem
Healthcare review data is unstructured, noisy, and hard to convert into operational action. Teams usually stop at sentiment analysis, which is not enough for business decisions.

### Primary Users
- Business owner / clinic administrator
- Ops or growth team
- Quality/process improvement team

### Why This Matters
- Ratings influence trust, patient retention, and revenue.
- Without structured analysis, repeated operational failures stay hidden.
- Teams need priority-ranked interventions, not just text summaries.

### What Success Looks Like
- Clear top operational themes from reviews
- Quantified theme impact on ratings
- Separation of isolated vs recurring vs systemic issues
- Action roadmap with priorities, KPIs, and expected lift
- Explainable outputs with confidence and evidence

### Key Assumptions
- Ratings (1-5) are a useful proxy for experience quality
- Reviews contain mixed themes; sentence-level analysis improves clarity
- Theme intensity (frequency x severity) is a meaningful predictor of rating movement

### Measurable Goals
- Cross-validated MAE for rating impact model
- AUC for 1-star vs 5-star classification
- Per-theme confidence (bootstrap + cluster quality)
- Coverage of recommendation KPIs in roadmap

## 2) Data Handling & System Design
### A. Data Exploration & Preprocessing
Files: `src/preprocess.py`, `src/themes.py`

Implemented:
- Schema normalization (`feedback`, `rating`, optional metadata)
- Cleaning: unicode normalization, artifact removal, spacing normalization, lowercasing
- Validation: rating coercion and filtering to `1..5`
- De-duplication and empty-text filtering
- Sentence chunking (one review -> multiple chunks)

Why chunking:
- A single review can mention multiple issues (e.g., doctor praise + billing complaint).
- Sentence-level chunking reduces mixed-topic noise and improves theme purity.

Feature extraction:
- Semantic embeddings for chunks
- Theme cluster IDs and auto labels
- Chunk severity scores
- Review-level intensity features

### B. End-to-End Architecture
Pipeline:
1. Ingest reviews (`hospital.csv`)
2. Clean and normalize
3. Sentence chunking
4. Embeddings (SentenceTransformer; fallback TF-IDF + SVD)
5. Theme clustering (MiniBatchKMeans + silhouette-based K selection)
6. Cluster auto-labeling with top n-grams
7. Severity scoring per chunk
8. Review-theme matrix (`intensity = frequency x severity`)
9. Impact models (Ridge + Logistic)
10. Bootstrap confidence
11. Recurrence/systemic detection
12. Risk scoring
13. Roadmap generation
14. Structured JSON report + Streamlit dashboard

Architecture is shown in the Streamlit `Pipeline Flow` tab.

### Architecture Diagram
```text
+---------------------------------------------------------------+
| Hospital Reviews CSV (feedback + rating + optional metadata) |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Preprocessing (schema normalize, clean text, dedupe)         |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Sentence Chunking (one review -> multiple chunks)            |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Embeddings (SentenceTransformer / TF-IDF + SVD fallback)     |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Theme Clustering (MiniBatchKMeans + silhouette-based K)      |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Theme Auto-Labeling (top TF-IDF n-grams + evidence)          |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Severity Scoring (HF sentiment / phrase-lexicon fallback)    |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Review-Theme Matrix (intensity = frequency x severity)       |
+---------------------------------------------------------------+
             |                                      |
             v                                      v
+-------------------------------+     +----------------------------+
| Ridge Regression              |     | Logistic Regression        |
| (impact on rating + CV MAE)   |     | (1-star vs 5-star + AUC)  |
+-------------------------------+     +----------------------------+
             \                                      /
              \                                    /
               v                                  v
        +----------------------------------------------+
        | Bootstrap Robustness (CI + confidence)       |
        +----------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Recurrence Metrics + Scope Labeling                           |
| (frequency, coverage, repetition -> isolated/recurring/systemic) |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Risk Scoring (impact + severity + frequency + coverage)      |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Action Roadmap (priority, quick wins, KPIs, expected lift)   |
+---------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------+
| Structured JSON Report + Streamlit Dashboard                  |
+---------------------------------------------------------------+
```

## 3) Technical Depth
Beyond sentiment analysis, this system provides:
- Semantic theme discovery
- Quantified rating impact coefficients
- Extreme-outcome drivers (1-star vs 5-star)
- Recurrence and systemic labeling
- Risk scoring and action prioritization
- Explainable evidence samples for each theme

## 4) Modeling & AI Strategy
### Model Choices and Why
- `SentenceTransformer (all-MiniLM-L6-v2)`:
  - Captures semantic similarity across differently-worded complaints
- `MiniBatchKMeans`:
  - Fast clustering for chunk-scale text embeddings
- `Ridge Regression`:
  - Stable coefficients with correlated theme features
- `Logistic Regression (1-star vs 5-star)`:
  - Clear separation of worst vs best experience drivers
- `Bootstrap resampling`:
  - Robust confidence from coefficient stability + CI behavior

### Alternatives Considered
- LDA / BERTopic for topic modeling
- Tree models (XGBoost/RandomForest) for impact
- SHAP-heavy interpretation stack

Current approach chosen for:
- interpretability,
- speed,
- reproducibility,
- hackathon-friendly delivery.

### Hybrid Strategy
- Primary semantic models with offline-safe fallbacks:
  - Embeddings fallback: TF-IDF + SVD
  - Severity fallback: dataset-derived phrase lexicon + rules

### Limitations
- Cluster IDs are arbitrary and require auto-label + evidence review
- Theme meanings may shift across reruns
- Recurrence thresholds can need dataset-specific tuning
- No guaranteed causal inference (associational impact)

## 5) Evaluation, Grounding & Metrics
### Metrics Used
- `CV MAE` (Ridge): average rating prediction error in stars
- `AUC` (1-star vs 5-star): discrimination quality for extremes
- `Cluster confidence`: silhouette-based cohesion/separation signal
- `Theme impact confidence`: bootstrap sign stability + CI precision

### Why These Metrics
- MAE is intuitive in star units.
- AUC directly answers "what drives worst vs best outcomes."
- Bootstrap confidence avoids overtrusting one model fit.
- Cluster confidence validates unsupervised theme quality.

### Limitations of Metrics
- High model score does not imply causal effect.
- Silhouette may not perfectly reflect business usefulness.
- Confidence depends on sample size and class balance.

### Test Cases 
1. Strong wait-time complaint -> should map to scheduling/wait theme, high severity, negative impact
2. Positive doctor-care review -> positive/mild severity, low risk
3. Billing transparency complaint -> billing theme + roadmap KPI
4. Mixed review (praise + complaint) -> multiple chunk themes
5. Extreme 1-star complaint -> stronger influence in extreme classifier

## 6) Business Actionability
The output is decision-ready:
- Ranked high-risk themes
- Priority actions with `quick_win` vs `high_effort`
- Suggested KPI for each recommendation
- Expected rating lift (directional)
- Supporting evidence snippets

## 7) Visualization & UX
Streamlit app (`streamlit_app.py`) provides:
- Vertical pipeline flow blocks
- Theme analysis table with evidence
- Decision-critical visuals:
  - Impact vs frequency priority matrix
  - Risk Pareto
  - Confidence robustness
  - Scope distribution
- Roadmap table and execution mix
- Full JSON viewer in diagnostics tab

## Output Schema
The report JSON includes:
- `clinic_summary`
- `theme_analysis`
  - theme, frequency, coverage, impact, severity, risk, scope, confidence, evidence
- `improvement_roadmap`
  - priority, recommendation, workstream, KPI, expected lift, effort
- `diagnostics`
  - model/cluster quality metrics

Saved in `outputs/` with versioned filenames.

## Project Structure
```text
.
|-- hospital.csv
|-- streamlit_app.py
|-- requirements.py
|-- outputs/
|-- src/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- themes.py
|   |-- impact.py
|   |-- systemic.py
|   |-- roadmap.py
|   |-- report.py
|   `-- pipeline.py
`-- README.md
```

## Setup
```powershell
python requirements.py --with-optional > requirements.txt
pip install -r requirements.txt
```

## Run Pipeline
```powershell
python -m src.pipeline --input hospital.csv --output outputs/report.json --severity-method rule
```

Notes:
- `--severity-method auto|hf|rule`
- Default save is versioned (`report.json`, `report_001.json`, ...)
- Use `--no-version-output` to overwrite

## Run Dashboard
```powershell
streamlit run streamlit_app.py
```
