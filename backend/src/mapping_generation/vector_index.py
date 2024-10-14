# !/usr/bin/env python3
# from langchain.schema import Document
import argparse
import time

import qdrant_client.http.models as rest
# from langchain.retrievers import MergerRetriever
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from .athena_api_retriever import AthenaFilters, RetrieverAthenaAPI
from .bi_encoder import SAPEmbeddings
from .compress import CustomCompressionRetriever,CustomMergeRetriever
from .data_loader import load_data
from .embeddingfilter import MyEmbeddingsFilter
from .llm_chain import pass_to_chat_llm_chain
from .param import *
from .qdrant import CustomQdrantVectorStore
from .utils import exact_match_wo_vocab, load_custom_docs_from_jsonl, load_docs_from_jsonl
from .utils import global_logger as logger


def set_merger_retriever(retrievers):
    ensemble_retriever = CustomMergeRetriever(retrievers=retrievers)
    return ensemble_retriever


def filter_results(query, results) -> list:
    prioritized = []
    non_prioritized = []
    seen_labels = set()

    # First pass: collect prioritized and non-prioritized results
    for res in results:
        label = res.metadata["label"]
        is_standard = res.metadata.get("is_standard", None)
        if query.lower() in label.lower():
            if label not in seen_labels:
                seen_labels.add(label)

                # if is_standard in ['S', 'C']:
                #     prioritized.append(res)
                # else:
                #     logger.info(f"found match: {label}--- {res.metadata['sid']} but not standard")
            else:
                non_prioritized.append(res)
        else:
            non_prioritized.append(res)
    docs = prioritized + non_prioritized
    logger.info(f"Prioritized: {[res.metadata['label'] for res in prioritized]}")
    return docs


def get_collection_vectors(client, collection_name) -> int:
    try:
        collection_status = client.get_collection(collection_name)
        # logger.info(f"collection status={collection_status}")
        vectors_count = collection_status.points_count
        logger.info(f"{collection_name} has {vectors_count} vectors.")
        return vectors_count
    except Exception as e:
        logger.info(f"Error fetching collection {collection_name} from Qdrant: {e}")
        return 0


def _create_payload_index(client, collection_name) -> None:
    client.create_payload_index(
        collection_name=collection_name, field_schema="keyword", field_name="metadata.label", wait=True
    )
    client.create_payload_index(
        collection_name=collection_name, field_schema="keyword", field_name="metadata.domain", wait=True
    )
    client.create_payload_index(
        collection_name=collection_name, field_schema="keyword", field_name="metadata.vocab", wait=True
    )
    client.create_payload_index(
        collection_name=collection_name, field_schema="keyword", field_name="metadata.is_standard", wait=True
    )
    # client.create_payload_index(
    #                 collection_name=collection_name,
    #                 field_schema= "keyword",
    #                 field_name="metadata.concept_class",
    #                 wait=True
    #             )


def generate_vector_index(
    dense_embedding,
    sparse_embedding,
    url=VECTOR_PATH,
    port=QDRANT_PORT,
    docs_file="/workspace/mapping_tool/data/output/sapbert_emb_docs_json.jsonl",
    mode="inference",
    collection_name="concept_mapping",
    topk=TOPK,
) -> CustomQdrantVectorStore:
    client = QdrantClient(url=url, port=port, https=True, timeout=300)
    # client = QdrantClient(":memory:")
    logger.info(f"collection exist: {client.collection_exists(collection_name)}")
    if client.collection_exists(collection_name):
        vector_count = get_collection_vectors(client, collection_name=collection_name)
    else:
        vector_count = 0
    if vector_count == 0 or mode == "recreate":
        docs = load_docs_from_jsonl(docs_file) if mode == "recreate" else None
        logger.info(f"Docs: {len(docs)}")
        client.delete_collection(collection_name=collection_name)
        vector_store = CustomQdrantVectorStore.from_documents(
            docs,
            embedding=dense_embedding,
            batch_size=64,
            url=VECTOR_PATH,
            port=QDRANT_PORT,
            https=True,
            vector_name="omop_dense_vector",
            sparse_vector_name="omop_sparse_vector",
            sparse_embedding=sparse_embedding,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_params={
                "size": 768,
                "distance": Distance.COSINE,
                "hnsw_config": rest.HnswConfigDiff(
                    m=38, ef_construct=64, full_scan_threshold=20000, max_indexing_threads=8, payload_m=38
                ),
                "quantization_config": rest.ScalarQuantization(
                    scalar=rest.ScalarQuantizationConfig(
                        type=rest.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
                "on_disk": True,
                # "on_disk_payload":True,
            },
            sparse_vector_params={
                "modifier": rest.Modifier.IDF,
                "index": {"full_scan_threshold": 20000, "on_disk": True},
            },
            force_recreate=True,
        )
        _create_payload_index(client, collection_name)

    else:
        # comment if if _create_payload_index is already called in if but collection exist and payload is not created than so call it
        _create_payload_index(client, collection_name)  # should be commented after first run
        vector_store = CustomQdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            vector_name="omop_dense_vector",
            sparse_vector_name="omop_sparse_vector",
            distance=rest.Distance.COSINE,
            retrieval_mode=RetrievalMode.HYBRID,
            validate_collection_config=True,
        )
        if mode == "update":
            docs = load_custom_docs_from_jsonl(docs_file)
            if vector_count > 0 and vector_count < len(docs):
                vector_store.add_documents(docs)
                vcount = get_collection_vectors(client=client, collection_name=collection_name)
                logger.info(f"Added {vcount - vector_count} vectors to collection")

    # similarity_score_threshold, 'fetch_k':100},
    #                       search_kwargs={'k': 10,'lambda_mult': 0.4},
    #     docsearch.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={'k': 6, 'lambda_mult': 0.25}

    # )
    # return vector_store.as_retriever(
    #             search_type="mmr",
    #             search_kwargs={'k':10, 'lambda_mult': 0.3}
    #         )

    # return vector_store.as_retriever(
    #             search_type="similarity_score_threshold",
    #             search_kwargs={'score_threshold': 0.3}
    #         )

    return vector_store.as_retriever(search_kwargs={"k": topk})


def initiate_api_retriever(k: str = TOPK) -> RetrieverAthenaAPI:
    #'LOINC','UCUM','OMOP Extension','ATC','RxNorm','Gender','Race','Ethnicity', excluded for BC5CDR-D and NCBI
    # FOR AAP datset: 'SNOMED','MeSH','MedDRA','LOINC
    # domain=['Condition','Observation']
    athena_api_retriever = RetrieverAthenaAPI(
        filters=AthenaFilters(
            vocabulary=["LOINC", "UCUM", "OMOP Extension", "ATC", "RxNorm", "Gender", "Race", "Ethnicity"],
            standard_concept=["Standard", "Classification"],
        ),
        k=k,
    )
    compression_retriever = set_compression_retriever(athena_api_retriever)
    return compression_retriever


# TODO: check


def update_api_search_filter(api_retriever, domain="observation", topk=10):
    if domain == "unit":
        api_retriever.filters = AthenaFilters(domain=None, vocabulary=["UCUM"], standard_concept=["Standard"])
    elif domain == "condition" or domain == "anatomic site":
        api_retriever.filters = AthenaFilters(
            domain=["Condition", "Meas Value", "Spec Anatomic Site"],
            vocabulary=["SNOMED"],
            standard_concept=["Standard"],
        )
    elif domain == "measurement":
        api_retriever.filters = AthenaFilters(
            domain=["Measurement", "Meas Value", "Observation","Spec Anatomic Site"],
            vocabulary=["LOINC", "SNOMED"],
            standard_concept=["Standard"],
        )
    elif domain == "drug":
        api_retriever.filters = AthenaFilters(
            domain=["Drug"], vocabulary=["RxNorm", "ATC", "SNOMED"], standard_concept=["Standard", "Classification"]
        )
    elif domain == "observation":
        api_retriever.filters = AthenaFilters(
            domain=["Observation", "Meas Value"], vocabulary=["SNOMED", "LOINC", "OMOP Extension"]
        )
    elif domain == "visit":
        api_retriever.filters = AthenaFilters(
            domain=["Visit", "Observation"], vocabulary=["SNOMED", "LOINC", "OMOP Extension"]
        )
    elif domain == "demographics":
        api_retriever.filters = AthenaFilters(
            domain=["Observation", "Meas Value"],
            vocabulary=["SNOMED", "LOINC", "OMOP Extension", "Gender", "Race", "Ethnicity"],
        )
    else:
        api_retriever.filters = AthenaFilters(
            vocabulary=["SNOMED", "LOINC", "OMOP Extension", "RxNorm", "ATC"],
            standard_concept=["Standard", "Classification"],
        )
    api_retriever.k = topk
    return api_retriever


def update_qdrant_search_filter(retriever, domain="all", topk=10):
    if domain == "unit":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[rest.FieldCondition(key="metadata.vocab", match=rest.MatchValue(value="ucum"))]
        )
    elif domain == "condition" or domain == "anatomic site":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[rest.FieldCondition(key="metadata.vocab", match=rest.MatchAny(any=["snomed"]))]
        )
    elif domain == "demographics":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(key="metadata.vocab", match=rest.MatchAny(any=["snomed", "loinc"])),
                rest.FieldCondition(key="metadata.domain", match=rest.MatchAny(any=["observation", "meas value"])),
            ]
        )
    elif domain == "measurement":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(key="metadata.vocab", match=rest.MatchAny(any=["loinc", "snomed"])),
                rest.FieldCondition(
                    key="metadata.domain", match=rest.MatchAny(any=['measurement','meas value','observation','spec anatomic site'])
                ),
            ]
        )
    elif domain == "drug":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab", match=rest.MatchAny(any=["rxnorm", "rxnorm extension", "atc", "snomed"])
                ),
                rest.FieldCondition(key="metadata.is_standard", match=rest.MatchAny(any=["S", "C"])),
            ]
        )
    elif (
        domain == "observation"
        or domain == "visit"
        or domain == "demographics"
        or domain == "history of event"
        or domain == "history of events"
        or domain == "life style"
    ):
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab", match=rest.MatchAny(any=["snomed", "loinc", "omop extension"])
                ),
                rest.FieldCondition(key="metadata.domain", match=rest.MatchAny(any=["observation", "meas value"])),
                rest.FieldCondition(key="metadata.is_standard", match=rest.MatchAny(any=["S", "C"])),
            ]
        )
    elif domain == "procedure":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(key="metadata.vocab", match=rest.MatchAny(any=["snomed"])),
                rest.FieldCondition(key="metadata.is_standard", match=rest.MatchAny(any=["S", "C"])),
            ]
        )
    else:
        print("No domain specified")
        retriever.search_kwargs["filter"] = rest.Filter(
            #  must=[
            #            rest.FieldCondition(
            #                 key="metadata.vocab",
            #                  match=rest.MatchExcept(**{"except": ["meddra","icd10","icd9","icd10cm","icd9cm","uk biobank","mesh"]}),
            #             )
            #         ])
            must=[
                rest.FieldCondition(
                    key="metadata.vocab", match=rest.MatchAny(any=["snomed", "loinc", "atc", "rxnorm extension"])
                ),
                rest.FieldCondition(key="metadata.is_standard", match=rest.MatchAny(any=["S", "C"])),
            ]
        )
    retriever.search_kwargs["k"] = topk
    return retriever


def update_merger_retriever(
    merger_retriever: CustomCompressionRetriever, domain="all", topk=10
) -> CustomCompressionRetriever:
    try:
        retrievers = merger_retriever.base_retriever.retrievers
        api_retriever = update_api_search_filter(retrievers[1].base_retriever, domain=domain, topk=topk)
        dense_retriever = update_qdrant_search_filter(retrievers[0], domain=domain, topk=topk)
        merger_retriever = CustomMergeRetriever(retrievers=[dense_retriever, api_retriever])
        merger_retriever = set_compression_retriever(merger_retriever)
        return merger_retriever
    except Exception as e:
        logger.info(f"Error updating merger retriever: {e}")
        return merger_retriever


def set_compression_retriever(base_retriever) -> CustomCompressionRetriever:
    embedding = SAPEmbeddings()
    embeddings_filter = MyEmbeddingsFilter(embeddings=embedding, similarity_threshold=0.5)

    compression_retriever = CustomCompressionRetriever(base_compressor=embeddings_filter, base_retriever=base_retriever)
    return compression_retriever


def log_accuracy(correct_dict, input_file="/workspace/mapping_tool/data/eval_datasets/accuracy.txt") -> None:
    with open(input_file, "w") as f:
        f.write("Accuracy for each k value for file{input_file}\n")
        for k, correct in correct_dict.items():
            accuracy = correct / len(queries)
            f.write(f"Accuracy for k={k}: {accuracy}\n")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Load Vector Store")
    parser.add_argument("--model_name", type=str, default=EMB_MODEL_NAME, help="Model identifier for embeddings")
    parser.add_argument("--llm_id", type=str, default=LLM_ID, help="Model identifier for embeddings")
    parser.add_argument("--mode", type=str, default="inference", help="The mode to run the model in")
    parser.add_argument(
        "--collection_name", type=str, default="concept_mapping", help="Generate vector index for given collection"
    )
    parser.add_argument(
        "--document_file_path",
        type=str,
        default="/workspace/mapping_tool/data/output/concepts.jsonl",
        help="Documents to index",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="/workspace/mapping_tool/data/eval_datasets/custom_data/references.txt",
        help="Documents to index",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/workspace/mapping_tool/data/eval_datasets/results.txt",
        help="Documents to index",
    )
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for filtering")
    args = parser.parse_args()
    mode = args.mode
    model_name = args.model_name
    llm_id = args.llm_id
    embeddings = SAPEmbeddings(model_id=model_name)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    hybrid_vector_retriever = generate_vector_index(
        embeddings,
        sparse_embeddings,
        docs_file=args.document_file_path,
        mode=mode,
        collection_name=args.collection_name,
        topk=10,
    )

    hybrid_vector_retriever = set_compression_retriever(hybrid_vector_retriever)
    correct = 0
    exact_founds = 0
    queries = load_data(args.input_data)
    correct_dict = {1: 0, 3: 0, 5: 0, 10: 0}
    k_values = [1, 3, 5, 10]
    # logger.info(f"Total queries: {queries}")
    max_queries = len(queries)
    # random.shuffle(queries)
    with open(args.output_file, "w") as f:
        for query in queries[:max_queries]:
            match_found = False
            # logger.info(query)
            if len(query) == 3:
                code, query, domain = query[0], query[1], query[2]
            else:
                code, query = query[0], query[1]

            codes_set = str(code).strip().lower().split("|")

            results = hybrid_vector_retriever.invoke(query)
            exact_results = exact_match_wo_vocab(query, results)
            if len(exact_results) >= 1:
                results = exact_results
                match_found = True
                exact_founds += 1
            else:
                if args.use_llm:
                    print(f"Query: {query}--\nResults: {[res.metadata['label'] for res in results]}")
                    results, _ = pass_to_chat_llm_chain(
                        query, results[:5], llm_name=llm_id, prompt_stage=2, domain="all"
                    )
            # results = filter_results(query, results)
            logger.info(f"length of results: {len(results)}")
            if results:
                for res in results:
                    if "domain" in res.metadata:
                        # logger.info(f"*{query}------{res.metadata['label']}---{res.metadata['domain']}---[{res.metadata['sid']}----[{res.metadata['is_standard']}]")
                        f.write(f"{query}\t{res.metadata['label']}\t{res.metadata['domain']}\t{res.metadata['sid']}\n")
                    else:
                        # logger.info(f"*{query}------{res.metadata['label']}---[{res.metadata['sid']}]")
                        f.write(f"{query}\t{res.metadata['label']}\t{res.metadata['sid']}\n")
            else:
                logger.info(f"*{query}------No results found")
                f.write(f"{query}\tNo results found\n")
            for k in k_values:
                if (
                    any(
                        sid in codes_set
                        for res in results[:k]
                        for sid in str(res.metadata.get("sid", "")).strip().lower().replace("+", "|").split("|")
                    )
                    or match_found
                ):
                    correct_dict[k] += 1
                else:
                    logger.info(f"{query} not found in top {k} results")
        for k in k_values:
            accuracy = correct_dict[k] / max_queries
            logger.info(f"Accuracy for k={k}: {accuracy}")
            f.write(f"Accuracy for k={k}: {accuracy}\n")
    log_accuracy(correct_dict)
    logger.info(f"Exact founds for file: {args.input_data} = {exact_founds}")
    # logger.info_results(ensemble_retriever.invoke("Progressive Familial Heart Block, Type II")) #midregional pro-atrial natriuretic peptide
    logger.info(f"Time taken fot total queries {len(queries)}: {time.time() - start_time}")
