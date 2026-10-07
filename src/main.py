"""
Brand & Social Media Content Analyzer - batch CLI.

Usage:
    python src/main.py                 -> runs on data/sample_posts.csv
    python src/main.py --input FILE    -> runs on a different CSV
    python src/main.py --save report.json  -> also saves full report as JSON
"""

import sys
import os
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import ContentAnalysisClient
from aggregator import build_brand_report
from report_generator import print_report


def load_posts(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_batch(client: ContentAnalysisClient, posts: list) -> list:
    analyses = []
    total = len(posts)

    for i, post in enumerate(posts, 1):
        print(f"Analyzing post {i}/{total}...", end="\r")

        result = client.analyze_post(
            caption=post.get("caption", ""),
            hashtags=post.get("hashtags", ""),
        )

        analyses.append(result)

    print(" " * 40, end="\r")

    failed = sum(
        1 for result in analyses
        if result.get("tone") == "unknown"
    )

    print(f"Completed analysis for {total - failed}/{total} posts.")
    if failed:
        print(
            f"Warning: {failed} post(s) returned fallback results. "
            "Review the API errors before interpreting the report."
        )

    return analyses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                         help="Path to a CSV with date,caption,hashtags,"
                              "likes,comments,shares columns")
    parser.add_argument("--save", default=None,
                         help="Optional path to save the full report as JSON")
    args = parser.parse_args()

    client = ContentAnalysisClient(config_path="config.yaml")
    input_path = args.input or client.config["post_history_path"]

    posts = load_posts(input_path)
    if not posts:
        print(f"No posts found in {input_path}")
        sys.exit(1)

    analyses = run_batch(client, posts)
    report = build_brand_report(
        posts,
        analyses,
        outlier_threshold=client.config.get("consistency_outlier_threshold", 0.34),
    )
    print_report(report)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {args.save}")
