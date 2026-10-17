# IT Helpdesk Ticket Classification and Analyzing System

A deep-learning pipeline and dashboard for IT helpdesk tickets. It reads the free-text description of a ticket and does three things:

1. **Classifies the category** (Network, Hardware, Software, Access/Account, Email, Security).
2. **Predicts the priority** (Low, Medium, High, Critical).
3. **Flags unusual tickets**, such as off-topic or junk submissions, with an autoencoder.

The results are shown in an interactive Streamlit dashboard, and an evaluation notebook goes through the model's performance in detail.

---

## Features

- **Multi-task BiLSTM** with a shared encoder and two output heads, one for category and one for priority.
- **Autoencoder anomaly detector** trained on TF-IDF features. Tickets with a high reconstruction error are flagged.
- **Realistic synthetic dataset** with typos, ambiguous categories, label noise and off-topic tickets, so the task isn't trivially easy.
- **Leakage-safe setup**: the vocabulary, the TF-IDF vectorizer and the anomaly threshold are all fit on training or validation data only.
- **Streamlit dashboard** with KPI cards and charts for volume, departments, categories, priorities, resolution times and anomalies.
- **Evaluation notebook** with confusion matrices, error analysis, resolution-time analysis and anomaly evaluation.

---

## Project Structure

```
.
├── data/
│   └── tickets.csv               # Synthetic dataset (5,171 tickets, with train/val/test split)
├── notebooks/
│   └── HelpDesk_Model_Analysis.ipynb   # Evaluation and analysis notebook
├── outputs/
│   ├── metrics.json              # Validation and test metrics
│   └── predictions.csv           # Dataset plus model predictions and anomaly scores
├── .streamlit/
│   └── config.toml               # Streamlit light theme
├── generate_data.py              # Builds the synthetic ticket dataset
├── train.py                      # Trains the models and writes outputs/
├── dashboard.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Data generation (`generate_data.py`)

Creates a noisy synthetic dataset of IT tickets. Each ticket is built from an issue phrasing, an optional greeting prefix and an optional impact suffix such as "production is affected" or "low impact, not urgent". The generator then adds realistic imperfections:

- Random typos (swapped characters).
- **Confusable categories.** For example, a shared-drive problem is sometimes labelled Access/Account, and a slow machine is sometimes labelled Software.
- About 5% random label noise for category and priority.
- About 2.5% off-topic tickets ("order catering for the office party", random words). These are marked `true_outlier = 1` and serve as ground truth for anomaly detection.
- Priority driven by impact keywords in the text, plus category-level base rates (Security tickets skew high).
- Resolution time drawn from a log-normal distribution that depends on category and priority.

Duplicate descriptions are dropped, and the data is split **70 / 15 / 15** into train, validation and test.

| Split | Tickets |
|-------|---------|
| Train | 3,619 |
| Validation | 775 |
| Test | 777 |

### 2. Training (`train.py`)

**Classifier (category and priority)**

- Tokenisation by regex, with a vocabulary built from the training split only (words that appear at least twice).
- Embedding (48-d), then a bidirectional LSTM (48 hidden units), then dropout and max-pooling over time.
- Two linear heads share the encoder. The loss is the sum of the category and priority cross-entropies.
- Adam optimizer with weight decay, batch size 64, up to 40 epochs, and early stopping (patience 4) on validation loss.

**Anomaly detector**

- TF-IDF features (400 terms), fit on the training split.
- A fully connected autoencoder (`400 → 32 → 8 → 32 → 400`) trained on the training data.
- The anomaly score is the reconstruction error. The **threshold is the 95th percentile of the validation error**, and anything above it is flagged.

### 3. Dashboard (`dashboard.py`)

A single-page Streamlit app that reads `outputs/predictions.csv` and shows:

- **KPIs:** total tickets, share of High/Critical tickets, mean resolution time, and number of flagged anomalies.
- **Volume and classification:** monthly volume, tickets by department, tickets by predicted category, and priority distribution by category.
- **Resolution and anomalies:** resolution-time distribution, top recurring issues, resolution action breakdown and anomaly distribution.

---

## Results

Metrics on the held-out **test** split (from `outputs/metrics.json`):

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
| Category classification | 87.4% | 0.873 |
| Priority prediction | 65.4% | 0.662 |

Validation: category 85.0% accuracy / 0.851 F1, priority 58.6% accuracy / 0.599 F1.

| Anomaly detection (test) | Value |
|--------------------------|-------|
| Tickets flagged | 2.8% |
| Recall on synthetic outliers | 0.524 |
| Precision | 0.500 |

The synthetic data intentionally includes label noise and ambiguous categories, so these numbers are well below 100% by design. Priority is harder than category because it depends on subtle impact wording and carries its own noise.

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- pip

### Installation

```bash
git clone https://github.com/<your-username>/IT-Helpdesk-Ticket-Classification-and-Analyzing-System.git
cd IT-Helpdesk-Ticket-Classification-and-Analyzing-System

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

Run the commands from the project root, since the scripts use relative paths.

```bash
# 1. (Optional) Regenerate the synthetic dataset
python generate_data.py

# 2. Train the models and write predictions and metrics to outputs/
python train.py

# 3. Launch the dashboard
streamlit run dashboard.py
```

The repository already includes `data/tickets.csv` and `outputs/`, so you can go straight to step 3 to explore the dashboard.

To explore the evaluation, open the notebook:

```bash
jupyter notebook notebooks/HelpDesk_Model_Analysis.ipynb
```

The notebook doesn't retrain anything. It loads the saved predictions and metrics and covers confusion matrices, classification reports, error analysis, resolution-time and action analysis, and anomaly detector evaluation. You will need `matplotlib` and `jupyter` installed in addition to `requirements.txt`.

---

## Dataset Schema

| Column | Description |
|--------|-------------|
| `ticket_id` | Unique ID (`INC100000`, ...) |
| `created_at` | Creation timestamp (all of 2025) |
| `department` | Requesting department |
| `description` | Free-text ticket description (model input) |
| `issue` | Specific issue type, e.g. "VPN drops" |
| `category` | Target label: one of 6 categories |
| `priority` | Target label: Low / Medium / High / Critical |
| `resolution_hours` | Time taken to resolve |
| `resolution_action` | Action taken to resolve the ticket |
| `true_outlier` | 1 for synthetic off-topic tickets (ground truth for anomaly detection) |
| `split` | `train`, `val` or `test` |

`outputs/predictions.csv` adds `pred_category`, `pred_priority`, `anomaly_score` and `is_anomaly`.

---

## Tech Stack

Python · PyTorch · scikit-learn · pandas · NumPy · Streamlit · Plotly · Jupyter

---

## Limitations and Future Work

- The data is synthetic, so real-world performance will differ. Retraining on real ticket history is the natural next step.
- Priority prediction (about 65% accuracy) has the most room to improve, for example with pretrained language-model embeddings or by adding metadata like department.
- Anomaly detection precision and recall are both around 50%. Better text representations or a tuned threshold could help.
- The dashboard shows results for the full dataset, with no filters yet.
- Possible additions: a live "predict a new ticket" form in the dashboard, and saving the trained model weights for reuse.

---
