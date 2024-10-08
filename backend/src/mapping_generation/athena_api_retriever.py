import asyncio
from typing import List

import requests
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field


class AthenaFilters(BaseModel):
    conceptClass: list[str] | None = Field(default=None, example=["Lab Test", "Procedure"])
    domain: list[str] | None = Field(
        default=None, example=["Condition", "Drug", "Measurement", "Observation", "Device", "Meas Value"]
    )
    standard_concept: list[str] | None = Field(default=None, example=["Standard", "Classification", "Non-standard"])
    vocabulary: list[str] | None = Field(
        default=None,
        example=["SNOMED", "LOINC", "UCUM", "OMOP Extension", "ATC", "RxNorm", "Gender", "Race", "Ethnicity"],
    )
    page: int = Field(default=1, ge=1, description="Page number for pagination. Defaults to 1.")
    pageSize: int = Field(default=20, ge=1, description="Number of results per page. Defaults to 5")

    def __init__(self, **data):
        super().__init__(**data)
        self.conceptClass = (
            [self.validate_and_format(item) for item in self.conceptClass] if self.conceptClass else None
        )
        self.domain = [self.validate_and_format(item) for item in self.domain] if self.domain else None
        self.standard_concept = (
            [self.validate_standard_concept(item) for item in self.standard_concept] if self.standard_concept else None
        )
        self.vocabulary = [self.validate_vocabulary(item) for item in self.vocabulary] if self.vocabulary else None

    def validate_and_format(self, item: str) -> str:
        """Validate that the item is a string and format it to sentence case."""
        if not isinstance(item, str):
            raise ValueError("Each item must be a string.")
        return item.capitalize()

    def validate_standard_concept(self, item: str) -> str:
        valid_concepts = ["Standard", "Classification", "Non-standard"]
        if item not in valid_concepts:
            raise ValueError(f"{item} is not a valid standard concept.")
        return item

    def validate_vocabulary(self, item: str) -> str:
        valid_vocabularies = [
            "SNOMED",
            "MeSH",
            "UCUM",
            "OMOP Extension",
            "LOINC",
            "ATC",
            "RxNorm",
            "Gender",
            "Race",
            "Ethnicity",
        ]
        if item not in valid_vocabularies:
            raise ValueError(f"{item} is not valid; must be one of {valid_vocabularies}.")
        return item


def convert_standard_concept(item: str) -> str:
    if item == "Standard":
        return "S"
    elif item == "Classification":
        return "C"
    elif item == "Non-standard":
        return None


def convert_to_documents(results) -> List[Document]:
    matching_documents = [
        Document(
            id=f"{str(item['vocabulary']).strip().lower()}:{item['code']!s}",
            page_content=str(item["name"]).lower(),
            metadata={
                "label": str(item["name"]).lower(),
                "sid": item["id"],
                "scode": item["code"],
                "domain": str(item["domain"]).strip().lower(),
                "is_standard": convert_standard_concept(str(item["standardConcept"])),
                "vocab": str(item["vocabulary"]).strip().lower(),
                "parent_term": "",
                "concept_class": str(item["className"]).strip().lower() if item["className"] else None,
            },
        )
        for item in results
    ]
    return matching_documents


class RetrieverAthenaAPI(BaseRetriever):
    # API_KEY: str = "8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"  # Use your actual API key here
    filters: AthenaFilters
    k: int = 15

    def __init__(self, filters: AthenaFilters, k: int = 5):
        super().__init__(filters=filters, k=k)
        # super().__init__(**kwargs)  # Properly initializing the BaseModel
        self.filters = filters
        self.k = k  # Maximum number of documents to return

    def _fetch_from_athena(self, query: str):
        """Fetch concepts from Athena API asynchronously and return JSON response."""
        url = "https://athena.ohdsi.org/api/v1/concepts"
        params = {
            "query": query,
            "domain": self.filters.domain,
            "vocabulary": self.filters.vocabulary,
            "standardConcept": ["Standard", "Classification"],
            "pageSize": self.filters.pageSize,
            "page": self.filters.page,
            "invalidReason": "Valid",
        }
        # We need to fake the user agent to avoid a 403 error, not good practice but it works
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            # "Authorization": f"Bearer {self.API_KEY}"
        }

        try:
            # print(f"params: {params}")
            if self.filters.vocabulary and "MeSH" in self.filters.vocabulary:
                non_mesh_vocab = [v for v in self.filters.vocabulary if v != "MeSH"]
                params["vocabulary"] = "MeSH"
                params["standardConcept"] = None
                params["domain"] = None
                response = requests.get(url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                search_res = response.json().get("content", [])[:5]
                # print(f"MeSH search_res: {search_res}")
                if non_mesh_vocab:
                    params["vocabulary"] = non_mesh_vocab
                    params["domain"] = self.filters.domain
                    params["standardConcept"] = self.filters.standard_concept
                    response = requests.get(url, params=params, headers=headers, timeout=60)
                    response.raise_for_status()
                    search_res += response.json().get("content", [])
                # Separate fetch for MeSH only

            else:
                response = requests.get(url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                search_res = response.json().get("content", [])
            # print(f"search_res: {search_res}")
            return search_res
        except Exception as e:
            raise ValueError(f"Failed to fetch documents from Athena API: {e}")

    async def _afetch_from_athena(self, query: str):
        """Fetch concepts from Athena API asynchronously and return JSON response."""
        url = "https://athena.ohdsi.org/api/v1/concepts"
        params = {
            "query": query,
            "domain": self.filters.domain,
            "vocabulary": self.filters.vocabulary,
            "standardConcept": ["Standard", "Classification"],
            "pageSize": self.filters.pageSize,
            "page": self.filters.page,
            "invalidReason": "Valid",
        }
        # We need to fake the user agent to avoid a 403 error, not good practice but it works
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            # "Authorization": f"Bearer {self.API_KEY}"
        }

        try:
            # print(f"params: {params}")
            if self.filters.vocabulary and "MeSH" in self.filters.vocabulary:
                non_mesh_vocab = [v for v in self.filters.vocabulary if v != "MeSH"]
                params["vocabulary"] = "MeSH"
                params["standardConcept"] = None
                params["domain"] = None
                response = requests.get(url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                search_res = response.json().get("content", [])[:5]
                # print(f"MeSH search_res: {search_res}")
                if non_mesh_vocab:
                    params["vocabulary"] = non_mesh_vocab
                    params["domain"] = self.filters.domain
                    params["standardConcept"] = self.filters.standard_concept
                    response = requests.get(url, params=params, headers=headers, timeout=60)
                    response.raise_for_status()
                    search_res += response.json().get("content", [])
                # Separate fetch for MeSH only

            else:
                response = requests.get(url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                search_res = response.json().get("content", [])
            # print(f"search_res: {search_res}")
            return search_res
        except Exception as e:
            raise ValueError(f"Failed to fetch documents from Athena API: {e}")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Retrieve relevant documents from the Athena API using run_manager callbacks."""
        try:
            # Trigger callback: query start
            # run_manager.on_retriever_start(query=query)

            # Fetch results from Athena
            results = self._fetch_from_athena(query)

            if not results or len(results) == 0:
                print(f"No ATHENA results found for query: {query}")
                return []
            matching_documents = convert_to_documents(results)
            print(f"length of athena documents: {len(matching_documents)}")

            return matching_documents
        except Exception as e:
            print(f"Failed to fetch documents from Athena API: {e} for query: {query}")
            return []

    # define asyn function to fetch data from athena api
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents from the Athena API."""
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self._afetch_from_athena(query))
            matching_documents = [
                Document(
                    id=f"{str(item['vocabulary']).strip().lower()}:{item['code']!s}",
                    page_content=str(item["name"]).lower(),
                    metadata={
                        "label": str(item["name"]).lower(),
                        "sid": item["id"],
                        "scode": item["code"],
                        "domain": str(item["domain"]).strip().lower(),
                        "is_standard": convert_standard_concept(str(item["standardConcept"])),
                        "vocab": str(item["vocabulary"]).strip().lower(),
                        "parent_term": "",
                        "concept_class": str(item["className"]).strip().lower() if item["className"] else None,
                    },
                )
                for item in results[: self.k]
            ]
            self.k = len(matching_documents)
            # print(f"Amatching_documents: {matching_documents}")
            return matching_documents[:5]
        except Exception as e:
            print(f"Failed to fetch documents from Athena API: {e} for query: {query}")
            return []
