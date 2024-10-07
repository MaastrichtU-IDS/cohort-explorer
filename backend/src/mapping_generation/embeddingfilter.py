from typing import Optional, Sequence, List

import numpy as np
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_transformers.embeddings_redundant_filter import _DocumentWithState

def _get_embeddings_from_stateful_docs(
    embeddings: Embeddings, documents: Sequence[_DocumentWithState]
) -> List[List[float]]:
    """Retrieve embeddings from documents using their state if available,
    or compute them from the metadata 'label'."""
    
    # Check if the embeddings already exist in the document's state
    if len(documents) and "embedded_doc" in documents[0].state:
        embedded_documents = [doc.state["embedded_doc"] for doc in documents]
    else:
        # Use the 'label' from metadata for embedding, instead of page content
        embedded_documents = embeddings.embed_documents(
            [d.metadata['label'] for d in documents]
        )
        # Store the embeddings in the document's state for future reference
        for doc, embedding in zip(documents, embedded_documents):
            doc.state["embedded_doc"] = embedding
    
    return embedded_documents

class MyEmbeddingsFilter(EmbeddingsFilter):
    """Custom EmbeddingsFilter class that inherits from EmbeddingsFilter
    and updates the `compress_documents` function to remove state information.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter documents based on similarity of their embeddings to the query.
        This function updates the original `compress_documents` function by:
        - Removing state information from the selected documents.
        """
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                
                get_stateful_documents,
            )
        except ImportError:
            raise ImportError(
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
        # print(f"documents before compression:\n{[doc.metadata['label'] for doc in documents]}")
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings, stateful_documents
        )
        embedded_query = self.embeddings.embed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold
            )
            included_idxs = included_idxs[similar_enough]
        sorted_docs_with_scores = sorted(
        [(stateful_documents[i], similarity[i]) for i in included_idxs],
        key=lambda x: x[1], reverse=True
    )   
        # Remove state information from the documents
        compressed_documents = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc, _ in sorted_docs_with_scores
        ][:10]

        
        return compressed_documents