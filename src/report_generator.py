"""Formats the brand-level report for readable console output."""


def print_report(report: dict):
    print("\n" + "=" * 70)
    print("BRAND CONTENT ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nDominant brand voice: {', '.join(report['dominant_voice']) or '(none detected)'}")
    print(f"Average voice consistency score: {report['average_consistency_score']} "
          f"(1.0 = perfectly consistent)")

    corr = report["engagement_signal_correlation"]
    if corr is None:
        print("Engagement-signal correlation: could not be computed "
              "(insufficient data or no variance)")
    else:
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "none"
        print(f"Engagement-signal correlation: {corr} ({direction})")
        print("  (This measures whether the LLM's engagement-signal judgment "
              "lines up with actual likes+comments+shares. It is a")
        print("  correlation on a small sample, not a validated predictive "
              "model - treat it as a directional signal only.)")

    outliers = report["outlier_posts"]
    print(f"\nPosts breaking from brand voice: {len(outliers)}")
    for post in outliers:
        print(f"  - [{post['date']}] \"{post['caption'][:60]}...\"")
        print(f"    tone={post['tone']} descriptors={post['voice_descriptors']}")

    print("\n" + "-" * 70)
    print("PER-POST DETAIL")
    print("-" * 70)
    for post in report["per_post_results"]:
        flag = " [OUTLIER]" if post["is_outlier"] else ""
        print(f"\n[{post['date']}]{flag} engagement_signal={post['engagement_signal']} "
              f"actual_engagement={post['actual_engagement']}")
        print(f"  Caption: {post['caption'][:80]}")
        print(f"  Tone: {post['tone']} | Voice: {post['voice_descriptors']}")
        print(f"  Reasoning: {post['reasoning']}")
        print(f"  Suggestion: {post['suggestion']}")
