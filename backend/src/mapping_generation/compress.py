from typing import Any, List
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain.retrievers import MergerRetriever
from langchain_core.retrievers import BaseRetriever
import asyncio

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
    

class CustomMergeRetriever(MergerRetriever):
    retrievers: List[BaseRetriever]
    """A list of retrievers to merge."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = self.merge_documents(query, run_manager)

        return merged_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = await self.amerge_documents(query, run_manager)

        return merged_documents

    def merge_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                config={
                    "callbacks": run_manager.get_child("retriever_{}".format(i + 1))
                },
            )
            for i, retriever in enumerate(self.retrievers)
        ]
        if not all(
            isinstance(doc, Document) for docs in retriever_docs for doc in docs
        ):
            print("retriever_docs format is not correct")

        # print doc.metadata['label'] for doc in matching_documents retriever_docs is list of list of documents
        # print(f"Qeury = {query} \n {[doc.metadata['label'] for doc in retriever_docs[0]]}")
        # check if any  doc.metadata['vocab'] == 'meddra'?: {any([doc.metadata['vocab'] == 'meddra' for doc in matching_documents])}")
        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(map(len, retriever_docs), default=0)
        seens_documents = set()
        print(f"max_docs: {max_docs}")
        for i in range(max_docs):
            for _, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    label_with_code = f"{doc[i].metadata['label']}_{doc[i].metadata['sid']}"
                    if i < len(doc) and label_with_code not in seens_documents:
                        merged_documents.append(doc[i])
                        seens_documents.add(label_with_code)
        # print(f"type of merged_documents: {type(merged_documents)} and type of doc in merged_documents: {type(merged_documents[0])}")
        # print(f"merged_documents: {[doc.metadata['label'] for doc in merged_documents]}")
        return merged_documents

    async def amerge_documents(
        self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronously merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = await asyncio.gather(
            *(
                retriever.ainvoke(
                    query,
                    config={
                        "callbacks": run_manager.get_child("retriever_{}".format(i + 1))
                    },
                )
                for i, retriever in enumerate(self.retrievers)
            )
        )

        # Merge the results of the retrievers.
        merged_documents = []
        seens_documents = set()
        max_docs = max(map(len, retriever_docs), default=0)
        for i in range(max_docs):
            for _, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    label_with_code = f"{doc[i].metadata['label']}_{doc[i].metadata['sid']}"
                    if label_with_code not in seens_documents:
                        merged_documents.append(doc[i])
                        seens_documents.add(label_with_code)

        return merged_documents