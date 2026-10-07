# Brand & Social Media Content Analyzer

An AI-powered NLP project that analyzes social media captions to identify brand voice, evaluate content consistency, and generate actionable recommendations for improving social media content.

## Overview

Brands need a consistent identity across their social media platforms. This project uses Google's Gemini API to analyze individual posts and aggregate the results into a brand-level report.

It evaluates the tone and language of captions, identifies posts that deviate from the dominant brand voice, and compares AI-assigned engagement signals with actual engagement metrics.

## Features

- **Brand Voice Analysis:** Identifies the tone and voice descriptors of individual posts.
- **Content Consistency:** Calculates how closely each post aligns with the dominant brand voice.
- **Outlier Detection:** Flags posts that differ from the established brand voice.
- **Engagement Analysis:** Compares AI-assigned engagement categories with actual likes, comments, and shares.
- **AI-Powered Recommendations:** Generates suggestions to improve captions and encourage audience interaction.
- **JSON Report Export:** Saves the complete analysis for further inspection or processing.

## Tech Stack

- **Language:** Python
- **AI / NLP:** Google Gemini API
- **Libraries:** Google Gen AI SDK, pandas (if used), PyYAML, python-dotenv
- **Data Format:** CSV and JSON
- **Development Environment:** Visual Studio Code

## Project Structure

```text
brand-content-analyzer/
├── data/
│   └── sample_posts.csv
├── prompts/
│   └── content_analysis_prompt.txt
├── src/
│   ├── main.py
│   ├── llm_client.py
│   ├── aggregator.py
│   └── report_generator.py
├── .env.example
├── .gitignore
├── config.yaml
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd brand-content-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root using `.env.example` as a template.

Add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit your actual API key to GitHub.

### 5. Run the analyzer

Run the default dataset:

```bash
python src/main.py
```

Export the complete report to JSON:

```bash
python src/main.py --save report.json
```

Analyze a different CSV file:

```bash
python src/main.py --input data/sample_posts.csv --save report.json
```

The CSV should contain the columns `date`, `caption`, `hashtags`, `likes`, `comments`, and `shares`.

## Output

The application generates a console report containing:

- Dominant brand voice descriptors
- Average voice consistency score
- Engagement-signal correlation, when calculable
- Posts identified as brand-voice outliers
- Individual post analyses and recommendations

An optional JSON export contains the full report for further analysis.

## Limitations

- Analysis quality depends on the input captions and the language model's responses.
- Engagement categories are AI-generated judgments, not guaranteed performance predictions.
- Correlation results depend on the size and variability of the dataset and should not be interpreted as proof of causation.
- API access and usage may be subject to provider limits and charges.

## Future Improvements

- Add a dashboard for interactive visualizations.
- Support platform-specific content analysis.
- Compare brand voice across multiple time periods.
- Integrate sentiment analysis and topic modeling.
- Evaluate recommendations against historical engagement data.

## Author

Developed as a Python, NLP, and generative-AI project.
