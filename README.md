# Sentiment analysis of hotel reviews

Scoring Google reviews of a hotel with NLTK's VADER sentiment analyser. A small learning project.

## Data
`customer_reviews.csv` - 55 reviews.

## What the script does (`sentiment-analysis.py`)
- Scores each review with VADER and takes the compound score.
- Labels it Positive (>= 0.05), Negative (<= -0.05) or Neutral (in between), the thresholds recommended by the VADER authors.

## Result
42 positive, 8 negative, 5 neutral.

Only 55 reviews and no human-labelled ground truth, so this shows the workflow rather than a validated result. VADER also struggles with sarcasm and mixed reviews (for example complaints about AC inside an otherwise "ok" review).

## Run it
```
pip install -r requirements.txt
python sentiment-analysis.py
```
The script downloads the `vader_lexicon` resource on first run.
