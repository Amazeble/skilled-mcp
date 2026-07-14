"""Vector search engine for finding relevant skills using turbovec."""

import logging
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex

from .skill_loader import Skill

logger = logging.getLogger(__name__)


class SkillSearchEngine:
    """Search engine for finding relevant skills using turbovec.

    Attributes
    ----------
    model : SentenceTransformer | None
        Embedding model for generating vectors (lazy-loaded).
    model_name : str
        Name of the sentence-transformers model to use.
    skills : list[Skill]
        List of indexed skills.
    index : TurboQuantIndex | None
        Turbovec index for vector search.
    bit_width : int
        Bit width for quantization (2, 3, or 4).
    _lock : threading.Lock
        Lock for thread-safe access to skills and index.
    """

    def __init__(self, model_name: str, bit_width: int = 4):
        """Initialize the search engine.

        Parameters
        ----------
        model_name : str
            Name of the sentence-transformers model to use.
        bit_width : int, optional
            Bit width for quantization, by default 4.
        """
        logger.info(
            f"Search engine initialized (model: {model_name}, bit_width: {bit_width}, lazy-loading enabled)"
        )
        self.model: SentenceTransformer | None = None
        self.model_name = model_name
        self.skills: list[Skill] = []
        self.bit_width = bit_width
        self.index: TurboQuantIndex | None = None
        self._lock = threading.Lock()

    def _ensure_model_loaded(self) -> SentenceTransformer:
        """Ensure the embedding model is loaded (lazy initialization).

        Returns
        -------
        SentenceTransformer
            The loaded embedding model.
        """
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name}")
        return self.model

    def index_skills(self, skills: list[Skill]) -> None:
        """Index a list of skills by generating their embeddings.

        Parameters
        ----------
        skills : list[Skill]
            Skills to index.
        """
        with self._lock:
            if not skills:
                logger.warning("No skills to index")
                self.skills = []
                self.index = None
                return

            logger.info(f"Indexing {len(skills)} skills using turbovec...")
            self.skills = skills

            # Generate embeddings from skill descriptions
            descriptions = [skill.description for skill in skills]
            model = self._ensure_model_loaded()
            embeddings = model.encode(descriptions, convert_to_numpy=True)

            # Initialize turbovec index with detected dimension
            dim = embeddings.shape[1]
            self.index = TurboQuantIndex(dim=dim, bit_width=self.bit_width)
            self.index.add(embeddings)

            logger.info(f"Successfully indexed {len(skills)} skills with turbovec")

    def add_skills(self, skills: list[Skill]) -> None:
        """Add skills incrementally and update index.

        Parameters
        ----------
        skills : list[Skill]
            Skills to add to the index.
        """
        if not skills:
            return

        with self._lock:
            logger.info(f"Adding {len(skills)} skills to index...")

            # Generate embeddings for new skills
            descriptions = [skill.description for skill in skills]
            model = self._ensure_model_loaded()
            new_embeddings = model.encode(descriptions, convert_to_numpy=True)

            # Append to existing skills
            self.skills.extend(skills)

            if self.index is None:
                # First batch of skills
                dim = new_embeddings.shape[1]
                self.index = TurboQuantIndex(dim=dim, bit_width=self.bit_width)
                self.index.add(new_embeddings)
            else:
                # Add to existing index
                self.index.add(new_embeddings)

            logger.info(
                f"Successfully added {len(skills)} skills. Total: {len(self.skills)} skills"
            )

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for the most relevant skills based on a query.

        Parameters
        ----------
        query : str
            The task description or query to search for.
        top_k : int, optional
            Number of top results to return, by default 3.

        Returns
        -------
        list[dict[str, Any]]
            List of skill dictionaries with relevance scores, sorted by relevance.
        """
        with self._lock:
            if not self.skills or self.index is None:
                logger.warning("No skills indexed, returning empty results")
                return []

            # Ensure top_k doesn't exceed available skills
            top_k = min(top_k, len(self.skills))

            logger.info(f"Searching for: '{query}' (top_k={top_k}) using turbovec")

            # Generate embedding for the query
            model = self._ensure_model_loaded()
            query_embedding = model.encode([query], convert_to_numpy=True)

            # Perform search using turbovec
            scores, indices = self.index.search(query_embedding, k=top_k)

            # Build results (turbovec search returns 2D arrays (nq, k))
            # We only have one query, so we take the first row
            scores = scores[0]
            indices = indices[0]

            results = []
            for i in range(len(indices)):
                idx = indices[i]
                score = float(scores[i])
                
                # turbovec returns indices as int64
                skill = self.skills[idx]

                result = skill.to_dict()
                result["relevance_score"] = score
                results.append(result)

                logger.debug(f"Found skill: {skill.name} (score: {score:.4f})")

            logger.info(f"Returning {len(results)} results")
            return results
