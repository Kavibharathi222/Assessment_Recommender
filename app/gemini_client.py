from __future__ import annotations

import os

from dotenv import load_dotenv


SYSTEM_PROMPT = """You are an SHL assessment recommendation assistant.
Use only the provided recommendation list as product facts.
Be concise, practical, and conversational.
If recommendations is null, ask one useful clarifying question.
Never invent SHL products or URLs.
If the user asks an off-topic question that is not related to hiring or SHL assessments, politely decline to answer and steer the conversation back to assessments."""


load_dotenv()
_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            from google import genai

            _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


async def generate_answer(
    user_query: str,
    recommendations: list[dict] | None,
    end_of_conversation: bool,
) -> str:
    client = _get_client()
    if not client or os.getenv("USE_GEMINI", "true").lower() != "true":
        return _fallback_answer(user_query, recommendations)

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        prompt = f"""{SYSTEM_PROMPT}

User request:
{user_query}

Structured recommendations:
{recommendations}

end_of_conversation:
{end_of_conversation}

Write the assistant message. Do not output JSON."""

        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text or _fallback_answer(user_query, recommendations)
    except Exception as exc:
        return (
            _fallback_answer(user_query, recommendations)
            + f"\n\nGemini response unavailable: {exc}"
        )


def _fallback_answer(user_query: str, recommendations: list[dict] | None) -> str:
    if recommendations is None:
        return "Happy to help. What role, seniority level, and hiring purpose is this for?"

    names = ", ".join(item["name"] for item in recommendations[:5])
    return f"Based on your request, the strongest SHL matches are: {names}."
