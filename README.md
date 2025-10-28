
# NexGen Logistics — Customer Experience Dashboard (Option 4)

An interactive Streamlit app to analyze customer feedback, sentiment, and its relationship with delivery performance.
Identifies **at‑risk customers** and surfaces actionable insights for operations and CX teams.

## 🚀 Features
- Upload or auto‑load CSVs from `data/` (customer_feedback, delivery_performance, orders)
- KPIs: Average rating, % positive sentiment, average delay, complaint rate
- Visuals: Ratings trend, sentiment pie, issue frequency bar, delay vs rating scatter (+ optional word cloud)
- Filters: Date range, customer segment, delivery priority, product category
- At‑risk list (rating ≤ 2, negative sentiment, or delay > 2 days) with **CSV export**
- Robust to missing columns; friendly warnings instead of crashes
- Caching for faster reloads

## 📦 Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Data Files
Place the CSV files in `./data/` or use the sidebar file‑uploader:
- `customer_feedback.csv` — ratings, feedback text, issue categories, dates
- `delivery_performance.csv` — promised vs actual dates, delay (computed), quality issues
- `orders.csv` — customer segment, product category, priority

> Column names are auto‑detected (OrderID, Rating, FeedbackText, IssueCategory, etc.).

## 🧠 Sentiment
Uses **NLTK VADER** if available (downloads `vader_lexicon` on first run). Falls back to a simple keyword‑based heuristic if VADER is unavailable.

## 🧪 Derived Metrics
- `DelayDays = ActualDeliveryDate - PromisedDeliveryDate` (days)
- **At‑risk** = rating ≤ 2 OR negative sentiment OR `DelayDays > 2`

## 📝 Why it fits the brief
- Python + Streamlit app with interactivity and downloads
- Multiple datasets merged and analyzed
- ≥ 4 chart types with insights
- Clear business impact through at‑risk detection and root‑cause views

## 🧩 Project Structure
```
.
├─ app.py
├─ requirements.txt
├─ README.md
└─ data/
   ├─ customer_feedback.csv
   ├─ delivery_performance.csv
   └─ orders.csv
```

## 🔐 Notes
- If you can't or don't want to allow NLTK to download the VADER lexicon, the app still works with the fallback heuristic.
- WordCloud is optional; installable via `requirements.txt`.

## 🏁 Evaluation Mapping
- Problem selection & justification: “Improve CX and reduce churn by analyzing feedback and delivery performance”
- Innovation: NLP scoring + at‑risk alerts
- Technical: Clean code, caching, error handling
- Data Analysis: Derived metrics, correlation views
- UX: Filters, KPIs, responsive layout
- Visualization: Trend, pie, bar, scatter (+ word cloud)
- Business Impact: Targeted recovery actions for low ratings or delays
