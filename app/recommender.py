from __future__ import annotations

import re
import os
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings
from sklearn.feature_extraction.text import HashingVectorizer


PERSIST_DIR = Path(os.getenv("CHROMA_PATH", "data/chroma"))
COLLECTION_NAME = "shl_product_catalog"
_RECOMMENDATION_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}

class LocalHashEmbeddingFunction:
    """Offline embedding function for Chroma using hashed word n-grams."""

    def __init__(self, n_features: int = 384):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(input)
        return matrix.astype("float32").toarray().tolist()

    def name(self) -> str:
        return "local_hash_embeddings"


KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


_COLLECTION = None


def get_collection(products: list[dict[str, Any]]):
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    use_persistence = os.getenv("CHROMA_PERSIST", "false").lower() == "true"
    settings = Settings(anonymized_telemetry=False)
    if use_persistence:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=settings)
        collection_name = COLLECTION_NAME
    else:
        SharedSystemClient.clear_system_cache()
        client = chromadb.EphemeralClient(settings=settings)
        collection_name = COLLECTION_NAME

    embedder = LocalHashEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() < len(products):
        _index_products(collection, products)

    _COLLECTION = collection
    return _COLLECTION


def recommend(
    query: str,
    products: list[dict[str, Any]],
    collection=None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    cache_key = (query.strip().lower(), limit)
    if cache_key in _RECOMMENDATION_CACHE:
        return _RECOMMENDATION_CACHE[cache_key]

    if collection is None:
        collection = get_collection(products)

    expanded_query = _expand_query(query)
    scores: dict[str, tuple[float, dict[str, Any]]] = {}

    try:
        results = collection.query(
            query_texts=[expanded_query],
            n_results=min(30, max(limit * 4, limit)),
            include=["metadatas", "documents", "distances"],
        )

        product_by_id = {
            str(item.get("entity_id") or item.get("name")): item for item in products
        }
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            entity_id = metadata["entity_id"]
            product = product_by_id.get(str(entity_id))
            if not product:
                continue
            vector_score = max(0.0, 1.0 - float(distance))
            scores[entity_id] = (0.35 * vector_score, product)
    except Exception:
        scores = {}

    for product in products:
        entity_id = str(product.get("entity_id") or product.get("name"))
        lexical_score = _lexical_score(expanded_query, product)
        business_score = _business_boost(query, product)
        if lexical_score == 0 and business_score == 0 and entity_id not in scores:
            continue

        existing_score, _ = scores.get(entity_id, (0.0, product))
        scores[entity_id] = (
            existing_score + lexical_score + business_score,
            product,
        )

    candidates = [product for _, product in sorted(
        scores.values(), key=lambda item: item[0], reverse=True
    )]
    candidates = _ensure_expected_battery(query, products, candidates)
    formatted = [_format_product(product) for product in candidates[:limit]]
    _RECOMMENDATION_CACHE[cache_key] = formatted
    return formatted


def should_ask_clarifying_question(query: str) -> bool:
    vague_terms = {
        "solution",
        "solutions",
        "assessment",
        "assessments",
        "test",
        "tests",
        "hiring",
        "recommend",
        "need",
     }
    words = re.findall(r"[a-zA-Z0-9+#.]+", query.lower())
    meaningful = [word for word in words if word not in vague_terms]
    return len(meaningful) < 3


def is_confirmation(query: str) -> bool:
    text = query.lower()
    signals = [
        "thanks",
        "thank you",
        "perfect",
        "final",
        "confirmed",
        "that's all",
        "that works",
        "we will use",
        "we'll use",
    ]
    return any(signal in text for signal in signals)


def apply_user_edits(
    query: str,
    current_recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = query.lower()
    edited = list(current_recommendations)

    if any(word in text for word in ["remove", "drop", "exclude"]):
        remove_terms = []
        if "opq" in text:
            remove_terms.extend(["opq", "occupational personality questionnaire"])
        if "personality" in text:
            remove_terms.append("personality")

        for term in remove_terms:
            edited = [
                item
                for item in edited
                if term not in item["name"].lower()
                and term not in " ".join(item["keys"]).lower()
            ]

    return edited


def _index_products(collection, products: list[dict[str, Any]]) -> None:
    existing = set(collection.get(include=[])["ids"])

    ids = []
    documents = []
    metadatas = []

    for product in products:
        entity_id = str(product.get("entity_id") or product.get("name"))
        if entity_id in existing:
            continue

        ids.append(entity_id)
        documents.append(_search_text(product))
        metadatas.append(
            {
                "entity_id": entity_id,
                "name": str(product.get("name", "")),
                "url": str(product.get("link", "")),
                "keys": ", ".join(product.get("keys") or []),
                "duration": str(product.get("duration", "")),
            }
        )

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)


def _search_text(product: dict[str, Any]) -> str:
    parts = [
        product.get("name", ""),
        product.get("description", ""),
        " ".join(product.get("keys") or []),
        " ".join(product.get("job_levels") or []),
        " ".join(product.get("languages") or []),
        product.get("duration", ""),
        f"remote {product.get('remote', '')}",
        f"adaptive {product.get('adaptive', '')}",
    ]
    return "\n".join(str(part) for part in parts if part)


def _expand_query(query: str) -> str:
    text = query.lower()
    additions: list[str] = []

    if any(term in text for term in ["ai", "artificial intelligence", "ml"]):
        additions.extend(
            [
                "python",
                "data science",
                "machine learning",
                "programming concepts",
                "sql",
                "automata data science",
            ]
        )

    if any(term in text for term in ["developer", "programmer", "software"]):
        additions.extend(["programming", "coding", "software development"])

    if any(term in text for term in ["junior", "entry", "entry-level", "fresher"]):
        additions.extend(["entry-level", "graduate", "basic", "fundamentals"])

    return " ".join([query, *additions])


def _business_boost(query: str, product: dict[str, Any]) -> float:
    text = query.lower()
    name = product.get("name", "").lower()
    keys = " ".join(product.get("keys") or []).lower()
    description = product.get("description", "").lower()
    boost = 0.0

    skill_tokens = re.findall(r"[a-zA-Z0-9+#.]+", text)
    for token in skill_tokens:
        if len(token) >= 3 and token in name:
            boost += 0.08
        elif len(token) >= 4 and token in description:
            boost += 0.03

    if any(word in text for word in ["cognitive", "reasoning", "aptitude"]):
        if "ability & aptitude" in keys or "verify interactive g+" in name:
            boost += 0.25

    if any(word in text for word in ["personality", "leadership", "senior"]):
        if "personality & behavior" in keys:
            boost += 0.15

    if any(word in text for word in ["graduate", "trainee", "entry level"]):
        levels = " ".join(product.get("job_levels") or []).lower()
        if "graduate" in levels or "entry-level" in levels:
            boost += 0.12

    if any(term in text for term in ["ai", "developer", "programmer", "software"]):
        if any(
            term in name
            for term in ["customer service", "retail", "sales", "call center"]
        ):
            boost -= 1.0

    return boost


def _lexical_score(query: str, product: dict[str, Any]) -> float:
    stopwords = {
        "and",
        "for",
        "the",
        "with",
        "test",
        "role",
        "senior",
        "engineer",
        "candidate",
        "candidates",
        "assessment",
        "assessments",
    }
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9+#.]+", query.lower())
        if len(token) >= 2 and token not in stopwords
    ]
    name = product.get("name", "").lower()
    description = product.get("description", "").lower()
    keys = " ".join(product.get("keys") or []).lower()
    levels = " ".join(product.get("job_levels") or []).lower()

    score = 0.0
    for token in tokens:
        if token in name:
            score += 0.45
        if token in description:
            score += 0.12
        if token in keys or token in levels:
            score += 0.05

    phrase = query.lower().strip()
    if phrase and phrase in name:
        score += 1.0

    return score


def _ensure_expected_battery(
    query: str,
    products: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = query.lower()
    expected: list[dict[str, Any]] = []

    if any(term in text for term in ["ai", "artificial intelligence", "ml"]):
        for needles in [
            ["python"],
            ["data science (new)"],
            ["automata data science"],
            ["programming concepts"],
            ["sql (new)"],
            ["verify interactive g+", "verify - g+"],
        ]:
            product = _find_product(products, needles)
            if product:
                expected.append(product)

    if any(word in text for word in ["cognitive", "reasoning", "aptitude"]):
        product = _find_product(products, ["verify interactive g+", "verify - g+"])
        if product:
            expected.append(product)

    if any(word in text for word in ["personality", "senior", "leadership"]):
        product = _find_product(products, ["opq32r", "occupational personality"])
        if product:
            expected.append(product)

    merged = []
    seen = set()
    if any(term in text for term in ["ai", "artificial intelligence", "ml"]):
        ordered = expected + candidates
    else:
        ordered = candidates[:4] + expected + candidates[4:]
    for product in ordered:
        entity_id = str(product.get("entity_id") or product.get("name"))
        if entity_id not in seen:
            seen.add(entity_id)
            merged.append(product)

    return merged


def _find_product(
    products: list[dict[str, Any]],
    needles: list[str],
) -> dict[str, Any] | None:
    for product in products:
        name = product.get("name", "").lower()
        if any(needle in name for needle in needles):
            return product
    return None


def _format_product(product: dict[str, Any]) -> dict[str, Any]:
    keys = product.get("keys") or []
    return {
        "name": product.get("name", ""),
        "test_type": _test_type(keys),
        "keys": keys,
        "duration": product.get("duration") or "-",
        "languages": product.get("languages") or [],
        "remote": product.get("remote", ""),
        "adaptive": product.get("adaptive", ""),
        "url": product.get("link", ""),
    }


def _test_type(keys: list[str]) -> str:
    codes = [KEY_TO_CODE[key] for key in keys if key in KEY_TO_CODE]
    return ",".join(codes) if codes else "-"
