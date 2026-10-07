import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Make the src folder accessible
sys.path.insert(0, str(PROJECT_DIR / "src"))

from llm_client import ContentAnalysisClient
from aggregator import build_brand_report


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brand & Social Media Content Analyzer",
    page_icon="<>",
    layout="wide"
)

st.title("<>Brand & Social Media Content Analyzer<>")

st.markdown(
    """
    Analyze brand voice, content consistency, and social media
    engagement using Gemini AI.
    """
)

st.divider()


# =========================================================
# COLUMN DETECTION
# =========================================================

def find_column(
    df: pd.DataFrame,
    keywords: list[str]
) -> Any:
    """
    Find a column whose name contains one of the supplied keywords.
    """

    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for keyword in keywords:

        for normalized_name, original_name in columns.items():

            if keyword in normalized_name:
                return original_name

    return None


def detect_columns(
    df: pd.DataFrame
) -> dict[str, Any]:
    """
    Automatically detect useful columns in an uploaded dataset.
    """

    # -----------------------------------------------------
    # Text/content columns
    # -----------------------------------------------------

    text_column = find_column(
        df,
        [
            "caption",
            "content",
            "post_text",
            "text",
            "tweet",
            "message",
            "description",
            "body",
            "post",
            "title"
        ]
    )


    # -----------------------------------------------------
    # Hashtag columns
    # -----------------------------------------------------

    hashtag_column = find_column(
        df,
        [
            "hashtags",
            "hashtag",
            "tags"
        ]
    )


    # -----------------------------------------------------
    # Date columns
    # -----------------------------------------------------

    date_column = find_column(
        df,
        [
            "date",
            "created_at",
            "created",
            "published_at",
            "published",
            "timestamp",
            "time"
        ]
    )


    # -----------------------------------------------------
    # Engagement columns
    # -----------------------------------------------------

    engagement_columns: dict[str, Any] = {}

    metric_keywords = {

        "likes": [
            "likes",
            "like",
            "favorites",
            "favourites",
            "favorite"
        ],

        "comments": [
            "comments",
            "comment",
            "replies",
            "reply"
        ],

        "shares": [
            "shares",
            "share",
            "retweets",
            "retweet",
            "reposts",
            "repost"
        ],

        "views": [
            "views",
            "view"
        ],

        "impressions": [
            "impressions",
            "impression"
        ],

        "reach": [
            "reach"
        ],

        "saves": [
            "saves",
            "save",
            "bookmarks",
            "bookmark"
        ]
    }


    for metric, keywords in metric_keywords.items():

        column = find_column(
            df,
            keywords
        )

        if column is not None:

            engagement_columns[
                metric
            ] = column


    return {
        "text": text_column,
        "hashtags": hashtag_column,
        "date": date_column,
        "engagement": engagement_columns
    }


# =========================================================
# UPLOAD DATASET
# =========================================================

st.subheader(
    "<<Upload Your Social Media Dataset>>"
)

uploaded_file = st.file_uploader(
    "Upload a CSV containing social media or textual content",
    type=["csv"]
)

st.caption(
    "The analyzer automatically detects content, hashtags, "
    "date, and engagement-related columns when available."
)


if uploaded_file is not None:

    # =====================================================
    # READ CSV
    # =====================================================

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Could not read the CSV file: {e}"
        )

        st.stop()


    # =====================================================
    # BASIC VALIDATION
    # =====================================================

    if df.empty:

        st.warning(
            "The uploaded CSV does not contain any rows."
        )

        st.stop()


    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    st.success(
        f"Loaded {len(df):,} rows successfully!"
    )


    # =====================================================
    # DETECT DATASET STRUCTURE
    # =====================================================

    detected = detect_columns(
        df
    )


    st.subheader(
        ">>Detected Dataset Structure<<"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            "**Content column**"
        )

        if detected["text"]:

            st.success(
                str(detected["text"])
            )

        else:

            st.error(
                "Not detected"
            )


    with col2:

        st.write(
            "**Hashtag column**"
        )

        if detected["hashtags"]:

            st.success(
                str(detected["hashtags"])
            )

        else:

            st.info(
                "Not available"
            )


    with col3:

        st.write(
            "**Date column**"
        )

        if detected["date"]:

            st.success(
                str(detected["date"])
            )

        else:

            st.info(
                "Not available"
            )


    # =====================================================
    # ENGAGEMENT INFORMATION
    # =====================================================

    if detected["engagement"]:

        st.write(
            "**Engagement columns detected:**"
        )

        engagement_text = ", ".join(
            [
                f"{metric} → {column}"
                for metric, column
                in detected["engagement"].items()
            ]
        )

        st.write(
            engagement_text
        )

    else:

        st.info(
            "No engagement columns were detected. "
            "Brand voice analysis can still be performed."
        )


    # =====================================================
    # CONTENT VALIDATION
    # =====================================================

    if not detected["text"]:

        st.error(
            """
            I couldn't identify a text/content column.

            Please upload a dataset containing textual content,
            such as caption, content, text, post, tweet,
            description, message, or title.
            """
        )

        st.stop()


    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.subheader(
        "<<Preview Your Data>>"
    )

    st.dataframe(
        df,
        width="stretch"
    )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        ">>Analyze My Brand<<",
        type="primary"
    ):

        try:

            with st.spinner(
                "Preparing your dataset..."
            ):

                # -----------------------------------------
                # Initialize Gemini client
                # -----------------------------------------

                client = ContentAnalysisClient(
                    config_path=str(
                        PROJECT_DIR / "config.yaml"
                    )
                )


                # -----------------------------------------
                # Normalize dataset
                # -----------------------------------------

                posts: list[
                    dict[str, Any]
                ] = []


                for _, row in df.iterrows():

                    # -------------------------------------
                    # Extract content
                    # -------------------------------------

                    content_value = row.get(
                        detected["text"],
                        ""
                    )


                    if pd.isna(
                        content_value
                    ):

                        caption = ""

                    else:

                        caption = str(
                            content_value
                        )


                    # -------------------------------------
                    # Extract hashtags
                    # -------------------------------------

                    hashtags = ""


                    if detected["hashtags"]:

                        hashtag_value = row.get(
                            detected["hashtags"],
                            ""
                        )


                        if not pd.isna(
                            hashtag_value
                        ):

                            hashtags = str(
                                hashtag_value
                            )


                    # -------------------------------------
                    # Create flexible post
                    # -------------------------------------

                    post: dict[
                        str,
                        Any
                    ] = {

                        "caption": caption,

                        "hashtags": hashtags
                    }


                    # -------------------------------------
                    # Preserve original columns
                    # -------------------------------------

                    for column in df.columns:

                        value = row[column]


                        if pd.isna(
                            value
                        ):

                            value = ""


                        post[
                            str(column)
                        ] = value


                    # -------------------------------------
                    # Normalize date
                    # -------------------------------------

                    if detected["date"]:

                        date_value = row.get(
                            detected["date"],
                            ""
                        )


                        if pd.isna(
                            date_value
                        ):

                            post["date"] = ""

                        else:

                            post["date"] = str(
                                date_value
                            )

                    else:

                        post["date"] = ""


                    # -------------------------------------
                    # Compatibility fields
                    #
                    # The current aggregator expects
                    # likes/comments/shares.
                    # -------------------------------------

                    post["likes"] = 0.0
                    post["comments"] = 0.0
                    post["shares"] = 0.0


                    if "likes" in detected["engagement"]:

                        value = row.get(
                            detected["engagement"]["likes"],
                            0
                        )


                        if not pd.isna(
                            value
                        ):

                            post["likes"] = value


                    if "comments" in detected["engagement"]:

                        value = row.get(
                            detected["engagement"]["comments"],
                            0
                        )


                        if not pd.isna(
                            value
                        ):

                            post["comments"] = value


                    if "shares" in detected["engagement"]:

                        value = row.get(
                            detected["engagement"]["shares"],
                            0
                        )


                        if not pd.isna(
                            value
                        ):

                            post["shares"] = value


                    posts.append(
                        post
                    )


            # =================================================
            # BATCH GEMINI ANALYSIS
            # =================================================

            st.subheader(
                ">AI Analysis Progress<"
            )


            # ---------------------------------------------
            # Batch size
            # ---------------------------------------------

            BATCH_SIZE = 10


            total_posts = len(
                posts
            )


            total_batches = (
                total_posts
                + BATCH_SIZE
                - 1
            ) // BATCH_SIZE


            progress = st.progress(
                0
            )


            progress_text = st.empty()


            analyses: list[
                dict[str, Any]
            ] = []


            # ---------------------------------------------
            # Process batches
            # ---------------------------------------------

            for batch_number, start in enumerate(
                range(
                    0,
                    total_posts,
                    BATCH_SIZE
                ),
                start=1
            ):

                end = min(
                    start + BATCH_SIZE,
                    total_posts
                )


                batch = posts[
                    start:end
                ]


                progress_text.write(
                    f"Analyzing batch "
                    f"{batch_number} of "
                    f"{total_batches} "
                    f"— posts {start + 1:,} "
                    f"to {end:,} of "
                    f"{total_posts:,}"
                )


                batch_results = client.analyze_batch(
                    batch
                )


                analyses.extend(
                    batch_results
                )


                progress.progress(
                    end / total_posts
                )


            progress_text.success(
                f"Finished analyzing "
                f"{total_posts:,} posts."
            )


            # =================================================
            # BUILD REPORT
            # =================================================

            with st.spinner(
                "Building brand insights..."
            ):

                report = build_brand_report(
                    posts,
                    analyses,
                    outlier_threshold=client.config.get(
                        "consistency_outlier_threshold",
                        0.34
                    )
                )


            # ---------------------------------------------
            # Store results
            # ---------------------------------------------

            st.session_state[
                "brand_report"
            ] = report


            st.session_state[
                "brand_posts"
            ] = posts


            st.success(
                "Analysis completed successfully! 🎉"
            )


        except Exception as e:

            st.error(
                f"Analysis failed: {e}"
            )


# =========================================================
# DISPLAY RESULTS
# =========================================================

report = st.session_state.get(
    "brand_report"
)


if report:

    st.divider()

    st.header(
        ">>Brand Insights<<"
    )


    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    successful_posts = sum(
        post.get(
            "analysis_status"
        ) == "success"

        for post
        in report[
            "per_post_results"
        ]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Posts Analyzed",
            f"{successful_posts:,}"
        )


    with col2:

        st.metric(
            "Voice Consistency",
            f"{report['average_consistency_score']:.0%}"
        )


    with col3:

        st.metric(
            "Voice Outliers",
            f"{len(report['outlier_posts']):,}"
        )


    # =====================================================
    # DOMINANT BRAND VOICE
    # =====================================================

    st.subheader(
        ">>Dominant Brand Voice<<"
    )


    dominant_voice = report[
        "dominant_voice"
    ]


    if dominant_voice:

        st.write(
            " · ".join(
                str(item)
                for item in dominant_voice
            )
        )

    else:

        st.info(
            "No dominant voice could be identified."
        )


    # =====================================================
    # ENGAGEMENT CORRELATION
    # =====================================================

    st.subheader(
        "<<Engagement Signal Correlation>>"
    )


    correlation = report[
        "engagement_signal_correlation"
    ]


    if correlation is not None:

        st.metric(
            "Correlation",
            f"{correlation:.2f}"
        )


        st.caption(
            """
            This is an exploratory measure of the relationship
            between Gemini's engagement judgment and available
            engagement data. It is not a validated prediction model.
            """
        )

    else:

        st.info(
            """
            Correlation could not be calculated from this dataset.
            This is normal when usable engagement metrics are unavailable.
            """
        )


    # =====================================================
    # OUTLIERS
    # =====================================================

    st.subheader(
        ">>Posts Breaking From Brand Voice<<"
    )


    if report[
        "outlier_posts"
    ]:

        outliers_df = pd.DataFrame(
            report[
                "outlier_posts"
            ]
        )


        preferred_columns = [
            "date",
            "caption",
            "tone",
            "voice_descriptors"
        ]


        available_columns = [
            column
            for column
            in preferred_columns

            if column
            in outliers_df.columns
        ]


        st.dataframe(
            outliers_df[
                available_columns
            ],
            width="stretch"
        )

    else:

        st.write(
            "No voice outliers detected."
        )


    # =====================================================
    # PER-POST RESULTS
    # =====================================================

    st.subheader(
        "<<Per-Post Analysis>>"
    )


    details_df = pd.DataFrame(
        report[
            "per_post_results"
        ]
    )


    if not details_df.empty:

        preferred_columns = [

            "date",

            "caption",

            "tone",

            "voice_descriptors",

            "consistency_score",

            "engagement_signal",

            "actual_engagement",

            "analysis_status",

            "reasoning",

            "suggestion"
        ]


        display_columns = [

            column

            for column
            in preferred_columns

            if column
            in details_df.columns
        ]


        st.dataframe(
            details_df[
                display_columns
            ],
            width="stretch"
        )


        # ---------------------------------------------
        # Download results
        # ---------------------------------------------

        csv_data = details_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )


        st.download_button(
            "⬇Download Analysis Results",
            data=csv_data,
            file_name="brand_analysis_results.csv",
            mime="text/csv"
        )