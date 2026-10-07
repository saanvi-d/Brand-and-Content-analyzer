"""
Computes brand-level metrics from social media post analyses.
"""

from collections import Counter
from math import sqrt


# ---------------------------------------------------------
# ENGAGEMENT SIGNAL SCORES
# ---------------------------------------------------------

TIER_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def is_valid_analysis(analysis: dict) -> bool:
    """
    Identify successful Gemini analysis results.

    Fallback results use tone='unknown'.
    """
    return (
        isinstance(analysis, dict)
        and analysis.get("tone") not in (None, "", "unknown")
    )


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_descriptor(value) -> str:
    """
    Normalize a voice descriptor so that small formatting
    differences do not create separate descriptors.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


# ---------------------------------------------------------
# DOMINANT DESCRIPTORS
# ---------------------------------------------------------

def get_descriptor_frequencies(analyses: list) -> Counter:
    """
    Count how many posts contain each voice descriptor.

    We count each descriptor once per post so that a descriptor
    repeated within one Gemini response does not artificially
    increase its importance.
    """

    counter = Counter()

    for analysis in analyses:

        if not is_valid_analysis(analysis):
            continue

        descriptors = {
            normalize_descriptor(descriptor)
            for descriptor in analysis.get(
                "voice_descriptors",
                []
            )
        }

        descriptors.discard("")

        for descriptor in descriptors:
            counter[descriptor] += 1

    return counter


def get_dominant_descriptors(
    analyses: list,
    top_n: int = 5
) -> list:
    """
    Return the most frequently occurring brand voice descriptors.
    """

    counter = get_descriptor_frequencies(analyses)

    return [
        descriptor
        for descriptor, _ in counter.most_common(top_n)
    ]


# ---------------------------------------------------------
# DOMINANT TONE
# ---------------------------------------------------------

def get_dominant_tone(analyses: list) -> str:
    """
    Find the most common tone across successful analyses.
    """

    tones = []

    for analysis in analyses:

        if not is_valid_analysis(analysis):
            continue

        tone = str(
            analysis.get("tone", "")
        ).strip().lower()

        if tone:
            tones.append(tone)

    if not tones:
        return "unknown"

    return Counter(tones).most_common(1)[0][0]


# ---------------------------------------------------------
# CONSISTENCY SCORE
# ---------------------------------------------------------

def consistency_score(
    post_descriptors: list,
    post_tone: str,
    descriptor_frequencies: Counter,
    dominant_tone: str,
    total_valid_posts: int
) -> float:
    """
    Calculate how strongly a post aligns with the overall
    brand voice.

    The score considers:

    1. Descriptor frequency across the dataset.
    2. Tone alignment with the dominant brand tone.
    3. Rare descriptors reduce the score.
    """

    post_set = {
        normalize_descriptor(descriptor)
        for descriptor in post_descriptors
    }

    post_set.discard("")

    if not post_set or total_valid_posts == 0:
        return 0.0

    # -----------------------------------------------------
    # Descriptor alignment
    # -----------------------------------------------------

    descriptor_scores = []

    for descriptor in post_set:

        frequency = descriptor_frequencies.get(
            descriptor,
            0
        )

        # Convert frequency into a 0-1 score.
        #
        # Example:
        # descriptor appears in 80% of posts -> 0.80
        # descriptor appears in 20% of posts -> 0.20
        frequency_ratio = frequency / total_valid_posts

        descriptor_scores.append(
            frequency_ratio
        )

    descriptor_alignment = (
        sum(descriptor_scores)
        / len(descriptor_scores)
    )

    # -----------------------------------------------------
    # Tone alignment
    # -----------------------------------------------------

    normalized_tone = str(
        post_tone or ""
    ).strip().lower()

    if (
        dominant_tone != "unknown"
        and normalized_tone == dominant_tone
    ):
        tone_alignment = 1.0
    else:
        tone_alignment = 0.0

    # -----------------------------------------------------
    # Combine signals
    # -----------------------------------------------------

    score = (
        0.75 * descriptor_alignment
        + 0.25 * tone_alignment
    )

    return max(
        0.0,
        min(1.0, score)
    )


# ---------------------------------------------------------
# SAFE INTEGER CONVERSION
# ---------------------------------------------------------

def safe_int(value) -> int:
    """
    Convert an engagement value safely.

    Missing or invalid values become zero.
    """

    try:
        return int(float(value or 0))

    except (
        TypeError,
        ValueError
    ):
        return 0


# ---------------------------------------------------------
# PEARSON CORRELATION
# ---------------------------------------------------------

def pearson_correlation(
    x: list,
    y: list
):
    """
    Calculate Pearson correlation.

    Returns None when there is not enough variation
    to calculate a meaningful correlation.
    """

    pairs = [
        (a, b)
        for a, b in zip(x, y)
        if a is not None
        and b is not None
    ]

    if len(pairs) < 2:
        return None

    xs = [
        pair[0]
        for pair in pairs
    ]

    ys = [
        pair[1]
        for pair in pairs
    ]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    covariance = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in pairs
    )

    std_x = sqrt(
        sum(
            (a - mean_x) ** 2
            for a in xs
        )
    )

    std_y = sqrt(
        sum(
            (b - mean_y) ** 2
            for b in ys
        )
    )

    if std_x == 0 or std_y == 0:
        return None

    return covariance / (
        std_x * std_y
    )


# ---------------------------------------------------------
# BRAND REPORT
# ---------------------------------------------------------

def build_brand_report(
    posts: list,
    analyses: list,
    outlier_threshold: float = 0.34
) -> dict:
    """
    Build a complete brand-level report.

    The function preserves the original post-analysis
    alignment and calculates:

    - dominant brand voice
    - dominant tone
    - average voice consistency
    - voice outliers
    - engagement correlation
    - per-post analysis
    """

    # -----------------------------------------------------
    # Validate alignment
    # -----------------------------------------------------

    if len(posts) != len(analyses):

        raise ValueError(
            "The number of posts and analyses must match."
        )

    # -----------------------------------------------------
    # Keep only successful analyses
    # -----------------------------------------------------

    valid_analyses = [
        analysis
        for analysis in analyses
        if is_valid_analysis(analysis)
    ]

    total_valid_posts = len(
        valid_analyses
    )

    # -----------------------------------------------------
    # Brand-level voice statistics
    # -----------------------------------------------------

    descriptor_frequencies = (
        get_descriptor_frequencies(
            valid_analyses
        )
    )

    dominant_descriptors = (
        get_dominant_descriptors(
            valid_analyses,
            top_n=5
        )
    )

    dominant_tone = (
        get_dominant_tone(
            valid_analyses
        )
    )

    # -----------------------------------------------------
    # Per-post calculations
    # -----------------------------------------------------

    per_post_results = []

    valid_consistency_scores = []

    engagement_totals = []

    signal_scores = []

    for post, analysis in zip(
        posts,
        analyses
    ):

        valid = is_valid_analysis(
            analysis
        )

        # ---------------------------------------------
        # Engagement
        # ---------------------------------------------

        total_engagement = (
            safe_int(
                post.get("likes")
            )
            + safe_int(
                post.get("comments")
            )
            + safe_int(
                post.get("shares")
            )
        )

        # ---------------------------------------------
        # Voice consistency
        # ---------------------------------------------

        if valid:

            score = consistency_score(
                post_descriptors=analysis.get(
                    "voice_descriptors",
                    []
                ),
                post_tone=analysis.get(
                    "tone",
                    ""
                ),
                descriptor_frequencies=(
                    descriptor_frequencies
                ),
                dominant_tone=dominant_tone,
                total_valid_posts=(
                    total_valid_posts
                )
            )

            valid_consistency_scores.append(
                score
            )

            is_outlier = (
                score < outlier_threshold
            )

            # -----------------------------------------
            # Engagement signal
            # -----------------------------------------

            signal_score = TIER_SCORE.get(
                str(
                    analysis.get(
                        "engagement_signal",
                        ""
                    )
                ).strip().lower()
            )

            if signal_score is not None:

                signal_scores.append(
                    signal_score
                )

                engagement_totals.append(
                    total_engagement
                )

        else:

            score = None
            is_outlier = False

        # ---------------------------------------------
        # Store post result
        # ---------------------------------------------

        per_post_results.append({

            "date": post.get(
                "date"
            ),

            "caption": post.get(
                "caption"
            ),

            "tone": analysis.get(
                "tone"
            ),

            "voice_descriptors": (
                analysis.get(
                    "voice_descriptors",
                    []
                )
            ),

            "consistency_score": (
                round(
                    score,
                    2
                )
                if score is not None
                else None
            ),

            "is_outlier": is_outlier,

            "analysis_status": (
                "success"
                if valid
                else "failed"
            ),

            "engagement_signal": (
                analysis.get(
                    "engagement_signal"
                )
            ),

            "reasoning": (
                analysis.get(
                    "reasoning"
                )
            ),

            "suggestion": (
                analysis.get(
                    "suggestion"
                )
            ),

            "actual_engagement": (
                total_engagement
            ),
        })

    # -----------------------------------------------------
    # Engagement correlation
    # -----------------------------------------------------

    correlation = pearson_correlation(
        signal_scores,
        engagement_totals
    )

    # -----------------------------------------------------
    # Average consistency
    # -----------------------------------------------------

    if valid_consistency_scores:

        average_consistency = (
            sum(
                valid_consistency_scores
            )
            / len(
                valid_consistency_scores
            )
        )

    else:

        average_consistency = 0.0

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    return {

        "dominant_voice": (
            dominant_descriptors
        ),

        "dominant_tone": (
            dominant_tone
        ),

        "average_consistency_score": round(
            average_consistency,
            2
        ),

        "outlier_posts": [
            post
            for post in per_post_results
            if post["is_outlier"]
        ],

        "engagement_signal_correlation": (
            round(
                correlation,
                2
            )
            if correlation is not None
            else None
        ),

        "per_post_results": (
            per_post_results
        ),
    }