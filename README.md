# AI-Powered Business Intelligence Agent

An NLP-driven Business Intelligence system that transforms large-scale customer review data into actionable business insights using Sentiment Analysis, Predictive Analytics, Chatbot Interaction, and Power BI Visualization.

---

# Project Overview

The AI-Powered Business Intelligence Agent is designed to analyze customer reviews and generate meaningful business intelligence insights. The system combines Natural Language Processing (NLP), sentiment analysis, predictive trend forecasting, and interactive dashboard visualization.

The project uses the Amazon Fine Food Reviews dataset to:

* Analyze customer sentiment
* Identify major complaints and praise themes
* Generate actionable business recommendations
* Forecast sentiment trends
* Provide chatbot-based interaction
* Visualize insights using Power BI

---

# Key Features

## Sentiment Analysis Engine

* Positive / Negative / Neutral classification
* Polarity-based sentiment scoring
* Customer satisfaction analysis
* Accuracy comparison with star ratings

## Business Intelligence Chatbot

The chatbot can answer queries related to:

* Overall sentiment
* Top complaints
* Customer satisfaction (CSAT)
* Product performance
* Forecast trends
* Business recommendations
* Risk analysis

## Predictive Analytics

* Monthly trend analysis
* Future negative sentiment forecasting
* Risk estimation
* Scenario analysis

## Python BI Dashboard

Interactive dashboard containing:

* KPI cards
* Sentiment distribution
* Trend analysis
* Product analytics
* Rating breakdown
* Complaint analysis

---

# Dataset

## Dataset Used

Amazon Fine Food Reviews Dataset

Dataset Link:
[https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## Dataset Information

| Column      | Description               |
| ----------- | ------------------------- |
| ProductId   | Unique product identifier |
| UserId      | Customer identifier       |
| ProfileName | Customer profile          |
| Score       | Product rating (1–5)      |
| Time        | Unix timestamp            |
| Summary     | Review summary            |
| Text        | Full review text          |

## Dataset Size

* Original dataset: ~568,000 reviews
* Working dataset: 10,000 sampled reviews

---

# System Architecture

```text
                    ┌────────────────────┐
                    │ Amazon Reviews     │
                    │ Dataset            │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Data Preprocessing │
                    │ • Text Cleaning    │
                    │ • Stopword Removal │
                    │ • Date Conversion  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Sentiment Analysis │
                    │ • TextBlob NLP     │
                    │ • Polarity Score   │
                    │ • Classification   │
                    └─────────┬──────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
 │ Keyword Mining │ │ Trend Analysis │ │ Forecast Engine│
 │ Complaint Themes│ │ Time-Series BI │ │ Predictive BI │
 └────────────────┘ └────────────────┘ └────────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Business Insights  │
                    │ • Recommendations  │
                    │ • Risk Analysis    │
                    │ • CSAT Analysis    │
                    └─────────┬──────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   ┌────────────────────┐          ┌────────────────────┐
   │ Chatbot Interface  │          │ Python BI Dashboard │
   │ Tkinter-based UI   │          │ Interactive Visuals│
   └────────────────────┘          └────────────────────┘
```

---

# Workflow

```text
Dataset Loading
        ↓
Text Preprocessing
        ↓
Sentiment Classification
        ↓
Keyword Extraction
        ↓
Trend & Forecast Analysis
        ↓
Business Insight Generation
        ↓
Chatbot Query Processing
        ↓
Dashboard Visualization
```

---

# Technologies Used

| Technology | Purpose                 |
| ---------- | ----------------------- |
| Python     | Core development        |
| Pandas     | Data processing         |
| NumPy      | Numerical operations    |
| TextBlob   | Sentiment analysis      |
| NLTK       | NLP preprocessing       |
| Matplotlib | Data visualization      |
| Tkinter    | Chatbot GUI             |

---

# Sentiment Classification Logic

| Polarity Score | Sentiment |
| -------------- | --------- |
| > 0.1          | Positive  |
| < -0.1         | Negative  |
| -0.1 to 0.1    | Neutral   |

---

# Power BI Dashboard Components

## KPI Metrics

* Total Reviews
* Positive Percentage
* Negative Percentage
* Average Rating
* CSAT Score

## Visualizations

* Sentiment Distribution
* Star Rating Breakdown
* Product Analysis
* Trend Analysis
* Complaint Keywords
* Praise Keywords

---

# Business Intelligence Insights

The system identifies:

* Customer satisfaction levels
* Product quality issues
* Delivery and packaging problems
* Customer loyalty indicators
* Future sentiment risks
* Product performance trends

---

# Business Recommendations

Based on analysis, the system recommends:

* Improving product quality control
* Reducing delivery-related issues
* Enhancing packaging standards
* Leveraging positive reviews in marketing
* Monitoring sentiment continuously
* Responding to negative reviews proactively

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/AI-Powered-Business-Intelligence-Agent.git
```

## Install Dependencies

```bash
pip install pandas numpy matplotlib textblob nltk
```

## Run Project

```bash
python bi_agent.py
```

---

# Project Structure

```text
AI-Powered-Business-Intelligence-Agent/
│
├── dataset/
│   └── reviews.csv
│
├── screenshots/
│   ├── chatbot_ui.png
│   ├── dashboard.png
│
├── ppt/
│   └── presentation.pptx
│
├── report/
│   └── final_report.pdf
│
├── bi_agent.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Results

## Key Findings

* Majority of reviews were positive
* Main complaints involved quality and delivery
* Strong customer satisfaction observed
* Forecasting identified future risk trends
* Business recommendations generated successfully

---

# Conclusion

The AI-Powered Business Intelligence Agent demonstrates how NLP and Business Intelligence techniques can convert unstructured customer reviews into actionable strategic insights.

The project successfully integrates:

* Sentiment Analysis
* Business Intelligence
* Predictive Analytics
* Chatbot Interaction
* Dashboard Visualization

This system provides an effective solution for customer feedback analysis and business decision-making.

---

# Author

Subash R
MBA – Data Science & Artificial Intelligence

---

# License

This project is licensed under the MIT License.
