from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.catalog import load_catalog
from app.gemini_client import generate_answer
from app.recommender import (
    apply_user_edits,
    is_confirmation,
    recommend,
    should_ask_clarifying_question,
)


app = FastAPI(title="SHL Assessment Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    current_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = 1


class ChatResponse(BaseModel):
    answer: str
    recommendations: list[dict[str, Any]] | None
    end_of_conversation: bool


@lru_cache(maxsize=1)
def _products() -> tuple[dict[str, Any], ...]:
    return tuple(load_catalog())


@app.get("/health")
def health() -> dict[str, Any]:
    products = _products()
    return {
        "status": "ok",
        "catalog_products": len(products),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    products = list(_products())
    edited = apply_user_edits(payload.message, payload.current_recommendations)
    end_of_conversation = is_confirmation(payload.message) or payload.turn_count >= 8

    if edited != payload.current_recommendations:
        recommendations = edited
    elif should_ask_clarifying_question(payload.message):
        recommendations = None
    else:
        recommendations = recommend(payload.message, products)

    answer = await generate_answer(payload.message, recommendations, end_of_conversation)
    return ChatResponse(
        answer=answer,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
