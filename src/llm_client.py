"""
Handles Gemini API calls for brand content analysis.
"""

import json
import os
from typing import Any

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types


class ContentAnalysisClient:
    """
    Client for analyzing social media content with Gemini.
    """

    def __init__(
        self,
        config_path: str = "config.yaml"
    ) -> None:

        load_dotenv()

        # =================================================
        # LOAD CONFIGURATION
        # =================================================

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.config: dict[str, Any] = yaml.safe_load(f)


        # =================================================
        # LOAD API KEY
        # =================================================

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )


        if not api_key:

            raise RuntimeError(
                "GEMINI_API_KEY not found. "
                "Check your .env file."
            )


        # =================================================
        # INITIALIZE GEMINI
        # =================================================

        self.client = genai.Client(
            api_key=api_key
        )


        # =================================================
        # LOAD PROMPT TEMPLATE
        # =================================================

        prompt_path = self.config[
            "prompt_path"
        ]


        with open(
            prompt_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.prompt_template = f.read()


    # =====================================================
    # PROMPT HELPERS
    # =====================================================

    def _build_prompt(
        self,
        caption: str,
        hashtags: str = ""
    ) -> str:

        """
        Build the Gemini prompt for one post.
        """

        return (
            self.prompt_template
            .replace(
                "{{CAPTION}}",
                caption
            )
            .replace(
                "{{HASHTAGS}}",
                hashtags or "(none)"
            )
        )


    # =====================================================
    # FALLBACK RESULT
    # =====================================================

    def _fallback_result(
        self
    ) -> dict[str, Any]:

        """
        Return a safe fallback when Gemini analysis fails.
        """

        return {
            "tone": "unknown",
            "voice_descriptors": [],
            "engagement_signal": "unknown",
            "reasoning": "Analysis failed for this post.",
            "suggestion": ""
        }


    # =====================================================
    # SINGLE POST ANALYSIS
    # =====================================================

    def analyze_post(
        self,
        caption: str,
        hashtags: str = ""
    ) -> dict[str, Any]:

        """
        Analyze one social media post using Gemini.

        Kept for backward compatibility with the
        original project.
        """

        fallback = self._fallback_result()


        prompt = self._build_prompt(
            caption,
            hashtags
        )


        try:

            response = self.client.models.generate_content(

                model=self.config[
                    "model_name"
                ],

                contents=prompt,

                config=types.GenerateContentConfig(

                    temperature=self.config.get(
                        "temperature",
                        0.2
                    ),

                    max_output_tokens=self.config.get(
                        "max_output_tokens",
                        1024
                    ),

                    response_mime_type="application/json"
                )
            )


            if not response.text:

                print(
                    "[llm_client] Gemini returned "
                    "an empty response."
                )

                return fallback


            result = json.loads(
                response.text
            )


            required_fields = {
                "tone",
                "voice_descriptors",
                "engagement_signal",
                "reasoning",
                "suggestion"
            }


            if not required_fields.issubset(
                result.keys()
            ):

                print(
                    "[llm_client] Response is missing "
                    "required fields."
                )

                return fallback


            return result


        except Exception as e:

            print(
                f"[llm_client] Analysis failed: {e}"
            )

            return fallback


    # =====================================================
    # BATCH ANALYSIS
    # =====================================================

    def analyze_batch(
        self,
        posts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:

        """
        Analyze multiple social media posts in one
        Gemini request.

        The method uses structured JSON output and
        validates every returned result.
        """

        if not posts:

            return []


        # =================================================
        # BUILD BATCH INPUT
        # =================================================

        batch_items: list[
            dict[str, Any]
        ] = []


        for index, post in enumerate(
            posts
        ):

            caption = str(
                post.get(
                    "caption",
                    ""
                )
            )


            hashtags = str(
                post.get(
                    "hashtags",
                    ""
                )
            )


            batch_items.append(
                {
                    "post_index": index,
                    "caption": caption,
                    "hashtags": hashtags
                }
            )


        # =================================================
        # BATCH PROMPT
        # =================================================

        batch_prompt = f"""
You are analyzing social media posts for a
brand voice and content analysis system.

Analyze EVERY post provided below.

For EACH post return:

- post_index
- tone
- voice_descriptors
- engagement_signal
- reasoning
- suggestion

Rules:

1. post_index must exactly match the input post_index.
2. Return exactly one result for every input post.
3. Do not omit posts.
4. Keep reasoning concise.
5. Keep suggestion concise.
6. voice_descriptors must be a JSON array of strings.
7. Return only structured JSON.

Input posts:

{json.dumps(
    batch_items,
    ensure_ascii=False,
    indent=2
)}
"""


        # =================================================
        # STRUCTURED RESPONSE SCHEMA
        # =================================================

        response_schema = {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "post_index": {
                        "type": "integer"
                    },

                    "tone": {
                        "type": "string"
                    },

                    "voice_descriptors": {

                        "type": "array",

                        "items": {
                            "type": "string"
                        }
                    },

                    "engagement_signal": {
                        "type": "string"
                    },

                    "reasoning": {
                        "type": "string"
                    },

                    "suggestion": {
                        "type": "string"
                    }
                },

                "required": [
                    "post_index",
                    "tone",
                    "voice_descriptors",
                    "engagement_signal",
                    "reasoning",
                    "suggestion"
                ]
            }
        }


        # =================================================
        # GEMINI REQUEST
        # =================================================

        try:

            response = self.client.models.generate_content(

                model=self.config[
                    "model_name"
                ],

                contents=batch_prompt,

                config=types.GenerateContentConfig(

                    temperature=self.config.get(
                        "temperature",
                        0.2
                    ),

                    # More room for multiple results
                    max_output_tokens=self.config.get(
                        "batch_max_output_tokens",
                        8192
                    ),

                    response_mime_type="application/json",

                    response_schema=response_schema
                )
            )


            # =================================================
            # EMPTY RESPONSE
            # =================================================

            if not response.text:

                print(
                    "[llm_client] Gemini returned "
                    "an empty batch response."
                )

                return [
                    self._fallback_result()
                    for _ in posts
                ]


            # =================================================
            # PARSE JSON
            # =================================================

            try:

                result = json.loads(
                    response.text
                )

            except json.JSONDecodeError as e:

                print(
                    "[llm_client] Invalid JSON returned "
                    "by Gemini."
                )

                print(
                    f"[llm_client] JSON error: {e}"
                )

                return [
                    self._fallback_result()
                    for _ in posts
                ]


            # =================================================
            # VALIDATE ARRAY
            # =================================================

            if not isinstance(
                result,
                list
            ):

                print(
                    "[llm_client] Batch response "
                    "was not a JSON array."
                )

                return [
                    self._fallback_result()
                    for _ in posts
                ]


            # =================================================
            # MAP RESULTS BY POST INDEX
            # =================================================

            indexed_results: dict[
                int,
                dict[str, Any]
            ] = {}


            for item in result:

                if not isinstance(
                    item,
                    dict
                ):

                    continue


                index = item.get(
                    "post_index"
                )


                if not isinstance(
                    index,
                    int
                ):

                    continue


                required_fields = {
                    "tone",
                    "voice_descriptors",
                    "engagement_signal",
                    "reasoning",
                    "suggestion"
                }


                if not required_fields.issubset(
                    item.keys()
                ):

                    continue


                indexed_results[
                    index
                ] = {

                    "tone": str(
                        item.get(
                            "tone",
                            "unknown"
                        )
                    ),

                    "voice_descriptors": [

                        str(value)

                        for value
                        in item.get(
                            "voice_descriptors",
                            []
                        )

                    ],

                    "engagement_signal": str(
                        item.get(
                            "engagement_signal",
                            "unknown"
                        )
                    ),

                    "reasoning": str(
                        item.get(
                            "reasoning",
                            ""
                        )
                    ),

                    "suggestion": str(
                        item.get(
                            "suggestion",
                            ""
                        )
                    )
                }


            # =================================================
            # RESTORE ORIGINAL ORDER
            # =================================================

            final_results: list[
                dict[str, Any]
            ] = []


            for index in range(
                len(posts)
            ):

                if index in indexed_results:

                    final_results.append(
                        indexed_results[
                            index
                        ]
                    )

                else:

                    print(
                        "[llm_client] Missing result "
                        f"for post {index}."
                    )

                    final_results.append(
                        self._fallback_result()
                    )


            return final_results


        except Exception as e:

            print(
                f"[llm_client] Batch analysis failed: {e}"
            )

            return [
                self._fallback_result()
                for _ in posts
            ]