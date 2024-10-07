from typing import Any, List

from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document


class CustomCompressionRetriever(ContextualCompressionRetriever):
    """Retriever that first checks for exact matches and bypasses compression if found."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query and bypass compression if exact match is found."""
        docs = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()}, **kwargs)

        if not docs:
            return []

        # First check for exact matches based on the 'label' in metadata
        exact_matches = [doc for doc in docs if doc.metadata.get("label", "").strip().lower() == query.strip().lower()]
        # If exact matches are found, return them without compression
        if exact_matches:
            print(f"Exact match found: {len(exact_matches)} documents")
            return exact_matches

        # Otherwise, proceed with compression
        compressed_docs = self.base_compressor.compress_documents(docs, query, callbacks=run_manager.get_child())
        print(f"compressed documents length: {len(compressed_docs)}")
        return list(compressed_docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously get documents relevant for a query and bypass compression if exact match is found."""
        docs = await self.base_retriever.ainvoke(query, config={"callbacks": run_manager.get_child()}, **kwargs)

        if not docs:
            return []

        # First check for exact matches based on the 'label' in metadata
        exact_matches = [doc for doc in docs if doc.metadata.get("label", "").strip().lower() == query.strip().lower()]

        # If exact matches are found, return them without compression
        if exact_matches:
            print(f"Exact match found: {len(exact_matches)} documents")
            return exact_matches

        # Otherwise, proceed with compression
        compressed_docs = await self.base_compressor.acompress_documents(docs, query, callbacks=run_manager.get_child())
        print(f"Asynch len of compressed_docs: {len(compressed_docs)}")
        return list(compressed_docs)
