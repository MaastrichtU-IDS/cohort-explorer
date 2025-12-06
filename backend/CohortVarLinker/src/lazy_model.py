# _model_instance = None

# def get_model():
#     global _model_instance
#     if _model_instance is None:
#         from .embed import ModelEmbedding
#         print("Loading SapBERT model (this may take a few moments)...")
#         _model_instance = ModelEmbedding()
#         print("Model loaded successfully!")
#     return _model_instance

# lazy_model.py
import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# You can plug this into your existing config system.
# For quick testing, you can also read from an env var:
#
#   export EMBEDDING_BACKEND=sapbert
#   or
#   export EMBEDDING_BACKEND=biolord
#
DEFAULT_BACKEND = os.getenv("EMBEDDING_BACKEND", "sapbert").lower()


class SapBERTEmbedding:
    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        cache_dir: str = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/models",
    ):
        cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/models",  # your local cache dir
        )
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    def _encode_tensor(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state  # [B, L, H]
        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # [B, H]
        return mean_embeddings

    def embed_text(self, text: str):
        embs = self._encode_tensor([text])
        return embs[0].cpu().numpy().tolist()


class BioLORDEmbedding:
    def __init__(self, model_name: str = "FremyCompany/BioLORD-2023"):
        self.model = SentenceTransformer(model_name, cache_folder="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/models")

    def embed_text(self, text: str):
        emb = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we can normalize later if needed
        )
        return emb.tolist()

class E5Embedding:
    def __init__(self, model_name: str = "intfloat/e5-large"):
        self.model = SentenceTransformer(model_name, cache_folder="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/models")

    def embed_text(self, text: str):
        emb = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we can normalize later if needed
        )
        
        return emb.tolist()
    
_model: Optional[object] = None
_backend: Optional[str] = None


def get_model(backend: Optional[str] = None):
    """
    Returns a singleton embedding model.
    backend: "sapbert" or "biolord".
    If None, uses DEFAULT_BACKEND (env/config).
    """
    global _model, _backend
    embedding_size = 768  # default
    if backend is None:
        backend = DEFAULT_BACKEND
    backend = backend.lower()

    if _model is None:
        if backend == "sapbert":
            _model = SapBERTEmbedding()
        elif backend == "biolord":
            _model = BioLORDEmbedding()
        elif backend == "e5":
            _model = E5Embedding()
            embedding_size = 1024  # E5 model embedding size
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")
        _backend = backend
    else:
        # Safety: if someone tries to change backend mid-run, you can either:
        if backend != _backend:
            raise RuntimeError(
                f"Embedding model already initialized with backend '{_backend}'. "
                f"Restart the process to switch to '{backend}'."
            )
    return _model, embedding_size


def get_backend_name() -> Optional[str]:
    return _backend
