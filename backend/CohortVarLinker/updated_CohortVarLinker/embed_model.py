
# import os
import torch
import numpy as np
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from .config import settings
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_MAP = {
    # --- Biomedical domain models ---
    "sapbert":  "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "biolord":  "FremyCompany/BioLORD-2023",
    # "coder":"GanjinZero/UMLSBert_ENG",
    "medembed":"abhinand/MedEmbed-base-v0.1",
    "coder":"GanjinZero/coder_eng",
    # --- General-purpose models ---
    "qwen3-0.6b":    "Qwen/Qwen3-Embedding-0.6B", 
    "qwen3-8b" :  "accounts/fireworks/models/qwen3-embedding-8b", #"qwen3-embedding:8b" ,
    "gemma": "google/embeddinggemma-300m",
    "nomic":"nomic-ai/nomic-embed-text-v2-moe",
    "e5":       "intfloat/e5-large-v2", 
    "minilm":   "all-MiniLM-L6-v2",
    "mxbai":    "mixedbread-ai/mxbai-embed-large-v1",
    "bge":      "BAAI/bge-large-en-v1.5", # same bge, cloud-hosted
    "gte":      "Alibaba-NLP/gte-modernbert-base", # same gte, cloud-hosted
    "openai": "text-embedding-3-large",
    "gemini": "models/gemini-embedding-001",
    "kalm": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    "qwen2":"Alibaba-NLP/gte-Qwen2-7B-instruct",
    "biobart":"GanjinZero/biobart-large",
    "sentence_sapbert": "amrothemich/sapbert-sentence-transformers"
              # MTEB #2  (0.6B)
    # "linq":     "Linq-AI-Research/Linq-Embed-Mistral",   # MTEB #5  (7B)
    # NOTE: gemini-embedding-001 and text-embedding-3-large are API-only.
    #       Use APIEmbeddingModel("gemini"|"openai", api_key=...) for those.

  
  
}
OLLAMA_MODELS = {"qwen3-8b"}
POOLING_STRATEGY = {
    "sapbert": "cls", "biolord": "cls", "coder": "cls", "medembed": "mean",
    "bge": "cls", "nomic": "mean", "mxbai": "cls",
    "minilm": "mean", "e5": "mean", "gte": "cls",
    "gemma": "mean", "kalm": "mean",
    "qwen3-0.6b": "last", "qwen3-8b": "last", "qwen2": "last", "linq": "last",
}
# TOGETHER_MODELS = {"gte"}

# Embedding dimensions: MUST match Qdrant VectorParams(size=...) exactly.
# The runtime get_model() also verifies via model.get_sentence_embedding_dimension().
EMBEDDING_DIMS = {
    # Biomedical
    "sapbert":  768,
    "biolord":  768,
    "medembed":  768,
    "sentence_sapbert":  768,
       # General-purpose
    "kalm": 896,
    "minilm":   384,
    "mxbai":    1024,
    "qwen3-0.6b":    1024,   # Qwen3-0.6B-Base hidden size
    "qwen3-8b":    4096,   # Qwen3-8B-Base hidden size
    "gemma":    768,
    "nomic":    768,
    "bge": 1024,
   

    "linq":     4096,   # Mistral-7B-v0.1 hidden size
    # API models (for reference / APIEmbeddingModel)
    "gemini":   3072,   # gemini-embedding-001
   
    "openai":   3072,   # text-embedding-3-large
    # Together AI hosted models
    "e5":  1024,
   
    "gte": 768,
}

# Instruction prefixes for decoder-based embedding models.
# Documents/passages are encoded WITHOUT prefix; queries WITH prefix.
# This asymmetry is by design (see E5, Qwen3, Linq-Mistral documentation).
INSTRUCTION_MODELS = {
    # "qwen3": "Instruct: Given a clinical variable description, retrieve semantically equivalent clinical variable descriptions\nQuery: ",
    "linq":  "Instruct: Given a clinical variable description, retrieve semantically equivalent clinical variable descriptions\nQuery: ",
    "qwen2": "Instruct: Given a clinical variable description, retrieve semantically equivalent clinical variable descriptions\nQuery: ",
    "e5":    "query: ",
}

# Models that use last-token pooling (decoders) vs CLS/mean pooling (encoders)
DECODER_MODELS = {"linq", "qwen2"}
PASSAGE_PREFIXES = {
    "e5": "passage: ",
}


class UnifiedEmbeddingModel:
    """HuggingFace AutoModel wrapper with configurable pooling."""

    def __init__(self, model_name_or_path: str, cache_dir: str = None,
                 backend_key: str = None):
        self.model_name = backend_key
        self.backend_key = backend_key or ""
        self._instruction_prefix = INSTRUCTION_MODELS.get(backend_key, "")
        self._passage_prefix = PASSAGE_PREFIXES.get(backend_key, "")
        self._is_decoder = backend_key in DECODER_MODELS
        self._pooling = POOLING_STRATEGY.get(backend_key, "mean")

        cache = cache_dir or settings.MODEL_CACHE_DIR
        dtype = torch.float16 if self._is_decoder else torch.float32

        print(f"🔥 Loading embedding model: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache, trust_remote_code=True,
            torch_dtype=dtype)

        if self._is_decoder:
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
        if not self._is_decoder:
            self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        print(f"✅ Model loaded: dim={self.embedding_dim}, "
              f"pool={self._pooling}, device={self.device}")

    def _pool(self, last_hidden: torch.Tensor,
          attention_mask: torch.Tensor) -> torch.Tensor:
        if self._pooling == "cls":
            return last_hidden[:, 0]
        if self._pooling == "last":
            # Left-padded batch → last column is always the final real token
            if attention_mask[:, -1].sum() == attention_mask.shape[0]:
                return last_hidden[:, -1]
            seq_lens = attention_mask.sum(dim=1) - 1
            return last_hidden[torch.arange(last_hidden.size(0),
                            device=last_hidden.device), seq_lens]
        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    # def _pool(self, last_hidden: torch.Tensor,
    #           attention_mask: torch.Tensor) -> torch.Tensor:
    #     if self._pooling == "cls":
    #         return last_hidden[:, 0]
    #     if self._pooling == "last":
    #         # last non-pad token per sequence
    #         seq_lens = attention_mask.sum(dim=1) - 1
    #         return last_hidden[torch.arange(last_hidden.size(0),
    #                            device=last_hidden.device), seq_lens]
    #     # mean pooling
    #     mask = attention_mask.unsqueeze(-1).float()
    #     return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def embed_batch(self, texts: List[str], show_progress: bool = False,
                    is_query: bool = True) -> np.ndarray:
        if is_query and self._instruction_prefix:
            texts = [self._instruction_prefix + t for t in texts]
        elif not is_query and self._passage_prefix:
            texts = [self._passage_prefix + t for t in texts]

        batch_size = 4 if self._is_decoder else 32
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            out = self.model(**encoded)
            hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
            emb = self._pool(hidden, encoded["attention_mask"])
            # emb = self._pool(out.last_hidden_state, encoded["attention_mask"])
            # L2 normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    def embed_text(self, text: str, is_query: bool = True) -> List[float]:
        return self.embed_batch([text], is_query=is_query)[0].tolist()
# =============================================================================
# API-BASED EMBEDDING MODELS  (Gemini, OpenAI — used in ADHTEB)
# =============================================================================

class APIEmbeddingModel:
    """
    Wrapper for API-based embedding models that cannot run locally.
    Provides the same embed_batch / embed_text interface.

    Usage:
        model = APIEmbeddingModel("openai", api_key="sk-...")
        model = APIEmbeddingModel("gemini", api_key="AIza...")
    """
    def __init__(self, provider: str, api_key: str = None):
        self.provider = provider.lower()
        self.model_name = provider
        self.api_key = api_key
        self.embedding_dim = EMBEDDING_DIMS.get(self.provider, 3072)
        self.device = "api"
        self.backend_key = provider
        self._MAX_BATCH_SIZE = 100

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self._model_name = "text-embedding-3-large"
            except ImportError:
                raise ImportError("pip install openai")
        elif self.provider == "gemini":
            try:
                from google import genai
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self._model_name = "gemini-embedding-001"
            except ImportError:
                raise ImportError("pip install google-genai")
        else:
            raise ValueError(f"Unknown API provider: {provider}")

        print(f"✅ API model ready: {self.provider} ({self._model_name})")

    def embed_batch(self, texts: List[str], show_progress: bool = False,
                    is_query: bool = True) -> np.ndarray:
        chunk_size = self._MAX_BATCH_SIZE
        all_embeddings = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]

            if self.provider == "openai":
                response = self.client.embeddings.create(
                    input=chunk, model=self._model_name
                )
                all_embeddings.extend([d.embedding for d in response.data])

            elif self.provider == "gemini":
                task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
                result = self._client.models.embed_content(
                    model=self._model_name,
                    contents=chunk,
                    config={"task_type": task},
                )
                all_embeddings.extend([e.values for e in result.embeddings])

        embeddings = np.array(all_embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def embed_text(self, text: str, is_query: bool = True) -> List[float]:
        return self.embed_batch([text], is_query=is_query)[0].tolist()


# =============================================================================
# TOGETHER AI EMBEDDING MODELS
# =============================================================================

class FireworksEmbeddingModel:
    """
    Wrapper for Together AI's hosted embedding models.
    Provides the same embed_batch / embed_text interface as local models.

    Advantages over local inference:
      - No GPU/MPS required (runs on Together's cloud GPUs)
      - Much faster for single-text calls (no local model startup overhead)

    Usage:
        model = TogetherEmbeddingModel("together-bge", api_key="...")
        model = TogetherEmbeddingModel("together-gte")

    Set TOGETHER_API_KEY env var or pass api_key explicitly.
    """
    # Together API has a limit of ~100 texts per request
    _MAX_BATCH_SIZE = 100

    def __init__(self, backend_key: str, api_key: str = None):
        # import os
        self.backend_key = backend_key
        self._api_key = api_key or settings.TOGETHER_API_KEY
        if not self._api_key:
            raise ValueError(
                "Together AI API key required. Set TOGETHER_API_KEY env var "
                "or pass api_key='...' to TogetherEmbeddingModel."
            )

        try:
            from together import Together
            self._client = Together(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Together SDK not installed. Run: pip install together"
            )

        self._model_name = MODEL_MAP.get(backend_key, backend_key)
        self.embedding_dim = EMBEDDING_DIMS.get(backend_key, 768)
        self.device = "api"
        self._is_decoder = False
        self._instruction_prefix = INSTRUCTION_MODELS.get(backend_key, "")
        self._passage_prefix = PASSAGE_PREFIXES.get(backend_key, "")

        print(f"✅ Together AI model ready: {self._model_name} (dim={self.embedding_dim})")

    def embed_batch(self, texts: List[str], show_progress: bool = False,
                    is_query: bool = True) -> np.ndarray:
        """
        Encode a list of strings via Together AI API.
        Automatically chunks into batches of 100.
        """

        if is_query and self._instruction_prefix:
            texts = [self._instruction_prefix + t for t in texts]
        elif not is_query and self._passage_prefix:
            texts = [self._passage_prefix + t for t in texts]
        
        all_embeddings = []
        for i in range(0, len(texts), self._MAX_BATCH_SIZE):
            batch = texts[i : i + self._MAX_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model_name,
                input=batch,
            )
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings)

        # L2-normalize for cosine consistency
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def embed_text(self, text: str, is_query: bool = True) -> List[float]:
        """Encode a single string."""
        return self.embed_batch([text], is_query=is_query)[0].tolist()


# =============================================================================
# OLLAMA EMBEDDING MODELS
# =============================================================================

class OllamaEmbeddingModel:
    """
    Wrapper for Ollama-hosted embedding models (quantized GGUF).
    Same embed_batch / embed_text interface as local models.

    Requires: pip install ollama
    And: ollama pull qwen3-embedding:8b

    Usage:
        model = OllamaEmbeddingModel("qwen3-8b")
    """
    def __init__(self, backend_key: str, base_url: str = "localhost:11434"):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError("Ollama SDK not installed. Run: pip install ollama")

        if base_url:
            from ollama import Client
            self._client = Client(host=base_url)
        else:
            self._client = None  # use module-level functions

        self.backend_key = backend_key
        self.model_name = backend_key
        self._ollama_tag = MODEL_MAP.get(backend_key, backend_key)
        self.embedding_dim = EMBEDDING_DIMS.get(backend_key, 4096)
        self.device = "ollama"
        self._instruction_prefix = INSTRUCTION_MODELS.get(backend_key, "")
        self._passage_prefix = PASSAGE_PREFIXES.get(backend_key, "")

        # Probe actual dimension with a test embed
        try:
            test = self._embed_raw(["test"])
            self.embedding_dim = len(test[0])
        except Exception as e:
            logger.warning(f"Could not probe Ollama dim: {e}")

        print(f"✅ Ollama model ready: {self._ollama_tag} (dim={self.embedding_dim})")

    def _embed_raw(self, texts: List[str]) -> List[List[float]]:
        if self._client:
            resp = self._client.embed(model=self._ollama_tag, input=texts)
        else:
            resp = self._ollama.embed(model=self._ollama_tag, input=texts)
        return resp.embeddings

    def embed_batch(self, texts: List[str], show_progress: bool = False,
                    is_query: bool = True) -> np.ndarray:
        if is_query and self._instruction_prefix:
            texts = [self._instruction_prefix + t for t in texts]
        elif not is_query and self._passage_prefix:
            texts = [self._passage_prefix + t for t in texts]

        # Ollama handles batches natively, but chunk for safety
        chunk_size = 64
        all_embs = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            all_embs.extend(self._embed_raw(chunk))

        embeddings = np.array(all_embs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def embed_text(self, text: str, is_query: bool = True) -> List[float]:
        return self.embed_batch([text], is_query=is_query)[0].tolist()

# =============================================================================
# SINGLETON MANAGEMENT & FACTORY
# =============================================================================

_model_instance: Optional[UnifiedEmbeddingModel] = None
_current_backend: Optional[str] = None


def get_model(backend: str = "biolord") -> Tuple[UnifiedEmbeddingModel, int]:
    """
    Factory returning (singleton_model, embedding_dimension).
    The dimension is verified at runtime via the loaded model.

    Args:
        backend: Key from MODEL_MAP ("sapbert", "qwen3", "linq", etc.)
                 or a raw HuggingFace model path.
    Returns:
        (UnifiedEmbeddingModel, int)  — model + embedding size for Qdrant
    """
    global _model_instance, _current_backend

    backend = backend.lower()
    target_model_name = MODEL_MAP.get(backend, backend)
    embedding_size = EMBEDDING_DIMS.get(backend, 768)
    
    if _model_instance is None or _current_backend != backend:
        if backend in ("gemini", "openai"):
            _model_instance = APIEmbeddingModel(backend, api_key=settings.OPENAI_API_KEY if backend == "openai" else settings.GEMINI_API_KEY)

        # if backend in TOGETHER_MODELS:
        #     print(f"🔥 Loading Together AI embedding model: {backend}...")
        #     _model_instance = TogetherEmbeddingModel(
        #         backend_key=backend,
        #     )
        elif backend in OLLAMA_MODELS:
            _model_instance = OllamaEmbeddingModel(backend_key=backend)
        else:
            _model_instance = UnifiedEmbeddingModel(
                target_model_name,
                cache_dir=settings.MODEL_CACHE_DIR,
                backend_key=backend,
            )
        _current_backend = backend
        # Trust the model's reported dimension over our static table
        if _model_instance.embedding_dim:
            embedding_size = _model_instance.embedding_dim

    return _model_instance, embedding_size


def similarity_score(backend_name: str, text_a: str, text_b: str) -> float:
    """Helper for cosine similarity between two strings."""
    if not text_a or not text_b:
        return 0.0
    if text_a == text_b:
        return 1.0

    model, _ = get_model(backend_name)
    vecs = model.embed_batch([text_a, text_b])
    score = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    return float(score)
