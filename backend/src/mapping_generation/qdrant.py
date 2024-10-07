from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from langchain_huggingface import HuggingFaceEmbeddings
# from qdrant_client.http.models import Distance
# import qdrant_client.http.models as rest
from .param import *
from typing import List, Optional, Sequence, Any, Iterable
from tqdm import tqdm
class CustomQdrantVectorStore(QdrantVectorStore):
    """Customized version of QdrantVectorStore with extended functionality."""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str | int]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str | int]:
        """Add texts with customized formatting based on metadata."""

        added_ids = []

        # Example customization: Format texts based on metadata
        formatted_texts = []
        for metadata in metadatas or [{}]:  # Default to empty dict if metadatas is None
            if 'synonyms' in metadata:
                # Customized formatting of text based on metadata
                formatted_text = f"<ent>{metadata.get('label', 'unknown')}</ent><syn>{metadata['synonyms']}</syn><domain>{metadata.get('domain', 'general')}</domain>"
            else:
                # Default behavior if no synonyms provided
                formatted_text = f"<ent>{metadata.get('label', 'unknown')}</ent><domain>{metadata.get('domain', 'general')}</domain>"

            formatted_texts.append(formatted_text)

        # Now use the base class's batch generation to add the vectors
        for batch_ids, points in tqdm(self._generate_batches(
            formatted_texts, metadatas, ids, batch_size
        )):
            self.client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids
