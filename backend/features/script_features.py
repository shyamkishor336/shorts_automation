"""
Stage 1 Feature Extraction — Script quality metrics.
"""

import logging
import math
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_nlp = None

_sentence_model = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        logger.info("Loading spaCy en_core_web_sm model…")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer (all-MiniLM-L6-v2) for first time…")
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def extract_readability_score(scenes: List[Dict]) -> Optional[float]:
   
    try:
        import textstat
        text = " ".join(s.get("narration", "") for s in scenes)
        return float(textstat.flesch_reading_ease(text))
    except Exception as exc:
        logger.warning(f"readability_score extraction failed: {exc}")
        return None


def extract_lexical_diversity(scenes: List[Dict]) -> Optional[float]:
  
    try:
        text = " ".join(s.get("narration", "") for s in scenes)
        words = text.lower().split()
        if not words:
            return None
        return round(len(set(words)) / len(words), 4)
    except Exception as exc:
        logger.warning(f"lexical_diversity extraction failed: {exc}")
        return None


def extract_prompt_coverage(prompt: str, scenes: List[Dict]) -> Optional[float]:
    """
    using sentence-transformers (all-MiniLM-L6-v2).
    Range: 0–1. Higher = script better covers the prompt.
    """
    try:
        import numpy as np

        model = _get_sentence_model()
        script_text = " ".join(s.get("narration", "") for s in scenes)

        emb_prompt = model.encode([prompt], normalize_embeddings=True)
        emb_script = model.encode([script_text], normalize_embeddings=True)

        similarity = float(np.dot(emb_prompt[0], emb_script[0]))
        return round(similarity, 4)
    except Exception as exc:
        logger.warning(f"prompt_coverage extraction failed: {exc}")
        return None


def extract_sentence_redundancy(scenes: List[Dict]) -> Optional[float]:
   
    try:
        sentences = [s.get("narration", "").strip() for s in scenes if s.get("narration")]
        total = len(sentences)
        if total == 0:
            return None
        unique = len(set(sentences))
        return round(1.0 - (unique / total), 4)
    except Exception as exc:
        logger.warning(f"sentence_redundancy extraction failed: {exc}")
        return None


def extract_entity_consistency(scenes: List[Dict]) -> float:
   
    try:
        nlp = _get_nlp()

        scene_entities: List[set] = []
        for scene in scenes:
            text = scene.get("narration") or ""
            doc = nlp(text)
            entities = {ent.text.lower() for ent in doc.ents}
            scene_entities.append(entities)

        all_entities: dict = {}
        for ents in scene_entities:
            for e in ents:
                all_entities[e] = all_entities.get(e, 0) + 1

        if not all_entities:
            # No named entities found — educational topics often use few proper nouns.
            # 0.0 is a legitimate, meaningful result here (not a failure).
            logger.debug("entity_consistency: no named entities found in any scene → 0.0")
            return 0.0

        multi_scene = sum(1 for cnt in all_entities.values() if cnt > 1)
        result = round(multi_scene / len(all_entities), 4)
        logger.debug(
            f"entity_consistency: {multi_scene}/{len(all_entities)} entities span "
            f"multiple scenes → {result}"
        )
        return result

    except Exception as exc:
        logger.error(f"entity_consistency extraction failed: {exc}", exc_info=True)
        return 0.5


def extract_topic_coherence(scenes: List[Dict]) -> Optional[float]:
   
    try:
        import numpy as np

        narrations = [s.get("narration", "") for s in scenes if s.get("narration")]
        if len(narrations) < 2:
            return None

        model = _get_sentence_model()
        embeddings = model.encode(narrations, normalize_embeddings=True)

        scores = []
        n = len(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                scores.append(float(np.dot(embeddings[i], embeddings[j])))

        return round(float(np.mean(scores)), 4)
    except Exception as exc:
        logger.warning(f"topic_coherence extraction failed: {exc}")
        return None


def extract_factual_conflict_flag(scenes: List[Dict]) -> Optional[int]:
  
    try:
        from transformers import pipeline as hf_pipeline

        nli = hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )
        narrations = [s.get("narration", "") for s in scenes if s.get("narration")]

        for i in range(len(narrations)):
            for j in range(i + 1, len(narrations)):
                result = nli(f"{narrations[i]} [SEP] {narrations[j]}")
                label = result[0]["label"].lower() if result else ""
                if "contradiction" in label:
                    return 1
        return 0
    except Exception as exc:
        logger.warning(f"factual_conflict_flag extraction failed: {exc}")
        return None


def extract_prompt_ambiguity(prompt: str) -> Optional[float]:
  
    try:
        import numpy as np

        # Use a small reference corpus of educational concepts
        reference_concepts = [
            "photosynthesis", "water cycle", "earthquakes", "human heart",
            "world war", "vaccines", "black holes", "democracy", "internet",
            "stars", "solar energy", "rainforest", "DNA", "ocean tides",
            "renewable energy", "binary", "seasons", "immune system",
            "inflation", "volcanoes", "physics", "chemistry", "biology",
            "history", "technology", "economics", "geography", "astronomy",
        ]

        model = _get_sentence_model()
        prompt_emb = model.encode([prompt], normalize_embeddings=True)[0]
        concept_embs = model.encode(reference_concepts, normalize_embeddings=True)

        similarities = np.array([
            float(np.dot(prompt_emb, ce)) for ce in concept_embs
        ])

        # Top-5 similarities as a probability distribution
        top5 = np.sort(similarities)[-5:]
        top5 = top5 - top5.min() + 1e-9  # shift to positive
        probs = top5 / top5.sum()

        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        return round(entropy, 4)
    except Exception as exc:
        logger.warning(f"prompt_ambiguity extraction failed: {exc}")
        return None


def extract_all(
    prompt: str, scenes: List[Dict]
) -> Dict[str, Any]:
    return {
        "readability_score": extract_readability_score(scenes),
        "lexical_diversity": extract_lexical_diversity(scenes),
        "prompt_coverage": extract_prompt_coverage(prompt, scenes),
        "sentence_redundancy": extract_sentence_redundancy(scenes),
        "entity_consistency": extract_entity_consistency(scenes),
        "topic_coherence": extract_topic_coherence(scenes),
        "factual_conflict_flag": extract_factual_conflict_flag(scenes),
        "prompt_ambiguity": extract_prompt_ambiguity(prompt),
    }
