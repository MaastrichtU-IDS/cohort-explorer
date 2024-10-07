import time

from langchain_qdrant import FastEmbedSparse

from .bi_encoder import SAPEmbeddings
from .data_loader import load_data
from .llm_chain import *
from .param import *
from .py_model import *
from .utils import append_results_to_csv
from .utils import global_logger as logger
from .vector_index import (
    generate_vector_index,
    initiate_api_retriever,
    set_compression_retriever,
    set_merger_retriever,
    update_merger_retriever,
)


def retriever_docs(query, retriever, domain="all", is_omop_data=False, topk: int = 10):
    if is_omop_data:
        retriever = update_merger_retriever(retriever, domain, topk)
    try:
        results = retriever.invoke(query)
        unique_results = filter_results(query, results)[:topk]
        # logger.info(f"Unique Results:\n {[res.metadata['label'] for res in unique_results]}")
        return unique_results
    except Exception as e:
        logger.error(f"Error retrieving docs: {e}")
        return None


def map_data(
    data, retriever, custom_data=False, output_file=None, llm_name="llama", topk=10, do_eval=False, is_omop_data=True
):
    global RETRIEVER_CACHE
    start_time = time.time()
    max_queries = len(data)  # 282
    results = []
    index = 0
    for _, item in enumerate(data[:10]):
        query_obj = item[1]
        query_result = full_query_processing(
            query_text=query_obj, retriever=retriever, llm_name=llm_name, is_omop_data=is_omop_data, topk=topk
        )
        if query_result:
            results.append(query_result)
        else:
            logger.info(f"NO RESULT FOR {item}")
        index += 1
        if (index + 1) % 15 == 0:
            time.sleep(0.05)  # Adjusted to be more appropriate than 0.0005

    end_time = time.time()
    total_time = end_time - start_time
    save_results(results, output_file)
    logger.info(f"Total execution time for {max_queries} queries is {total_time} seconds.")
    return results


def full_query_processing(
    query_text: QueryDecomposedModel, retriever: Any, llm_name: str, topk: int, is_omop_data=True
):
    try:
        logger.info(f"Processing query: {query_text}")
        # normalized_query_text = rule_base_decomposition(query_text)
        # processes_results = []
        # if not is_omop_data:
        #     query_decomposed = find_domain(query_text)
        #     query_decomposed['domain'] = 'all'
        #     processes_results = temp_process_query_details(query_decomposed, retriever, llm_name, topk, query_text)
        # else:
        # if not custom_data:
        #     query_decomposed = find_domain(query_text)
        #     processes_results = temp_process_query_details(query_decomposed, retriever, llm_name, topk, query_text)
        # else:
        if query_text is None:
            return None
        query_decomposed = extract_information(query_text.full_query, llm_name)
        logger.info(f"Query decomposed:{query_decomposed}")
        processes_results = temp_process_query_details(
            llm_query_obj=query_decomposed,
            retriever_cache=retriever,
            llm_name=llm_name,
            topk=topk,
            original_query_obj=query_text,
            is_omop_data=is_omop_data,
        )
        # logger.info("Mapping result:", processes_results)
        return processes_results
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return None


def temp_process_query_details(
    llm_query_obj: QueryDecomposedModel,
    retriever_cache: Any,
    llm_name: str,
    topk: int,
    original_query_obj: QueryDecomposedModel,
    is_omop_data=False,
):
    try:
        if llm_query_obj:
            original_query_obj.domain = llm_query_obj.domain
            original_query_obj.rel = llm_query_obj.rel
            original_query_obj.unit = llm_query_obj.unit
            original_query_obj.categories = llm_query_obj.categories
            logger.info(f"original query obj={original_query_obj}")
            variable_label_matches, additional_entities_matches, categories_matches, unit_matches = (
                None,
                None,
                None,
                None,
            )
            base_entity = original_query_obj.base_entity
            domain = original_query_obj.domain
            variable_label_matches, found_match = process_retrieved_docs(
                base_entity,
                retriever_docs(
                    original_query_obj.base_entity, retriever_cache, domain="all", is_omop_data=is_omop_data, topk=topk
                ),
                llm_name,
                "all",
                belief_threshold=0.9,
            )

            if found_match and len(variable_label_matches) > 0:
                # logger.info(f"matches={pretty_print_docs(variable_label_matches)}\n,match found={found_match}")
                main_term = base_entity  # Assign base_entity to main_term if a match is found
                llm_query_obj = original_query_obj
            else:
                # Step 5: Process the main_term only if the base_entity match was not found
                main_term = llm_query_obj.base_entity if llm_query_obj else base_entity
                variable_label_matches, _ = process_retrieved_docs(
                    main_term,
                    retriever_docs(
                        main_term, retriever_cache, domain=llm_query_obj.domain, is_omop_data=is_omop_data, topk=topk
                    ),
                    llm_name,
                    llm_query_obj.domain,
                )
            additional_entities = llm_query_obj.additional_entities
            categories = llm_query_obj.categories
            domain = llm_query_obj.domain
            unit = llm_query_obj.unit
            rel = llm_query_obj.rel
            logger.info(
                f"main_term={main_term}, context={additional_entities}, status={categories}, domain={domain}, unit={unit}"
            )
            if additional_entities:
                logger.info(f"Processing additional entities: {additional_entities}")
                additional_entities_matches = (
                    process_values(
                        additional_entities,
                        retriever_cache,
                        llm_name,
                        domain=domain,
                        values_type="additional",
                        is_omop_data=is_omop_data,
                        topk=topk,
                    )
                    if additional_entities
                    else {}
                )
            if categories:
                logger.info(f"Processing categories: {categories}")
                categories_matches = (
                    process_values(
                        categories,
                        retriever_cache,
                        llm_name,
                        domain="all",
                        values_type="status",
                        is_omop_data=is_omop_data,
                        topk=topk,
                    )
                    if categories
                    else {}
                )
            # if rel:
            #     rel_docs = process_values([rel], retriever_cache, llm_name, domain='all', values_type='rel')[rel] if rel else []
            if unit:
                logger.info(f"Processing unit: {unit}")
                unit_matches = (
                    process_unit(
                        unit, retriever_cache, llm=llm_name, domain="unit", is_omop_data=is_omop_data, topk=topk
                    )
                    if unit
                    else []
                )
            mapping_result = create_processed_result(
                ProcessedResultsModel(
                    base_entity=main_term,
                    domain=domain,
                    base_entity_matches=variable_label_matches if variable_label_matches else [],
                    categories=categories,
                    categories_matches=categories_matches if categories_matches else {},
                    unit=unit,
                    unit_matches=unit_matches if unit_matches else [],
                    original_query=llm_query_obj.original_label,
                    additional_entities=additional_entities,
                    primary_to_secondary_rel=rel,
                    additional_entities_matches=additional_entities_matches if additional_entities_matches else {},
                )
            )

            return mapping_result
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return None


def process_retrieved_docs(query, docs, llm_name=None, domain=None, belief_threshold=0.8):
    # logger.info_docs(docs)
    if docs and len(docs) > 0:
        if matched_docs := exact_match_found(query_text=query, documents=docs, domain=domain):
            return post_process_candidates(matched_docs, max=1), True
        if llm_name:
            logger.info(f"No string match found for {query} pass to {llm_name}")
            domain_specific_docs = filter_irrelevant_domain_candidates(docs, domain)
            if len(domain_specific_docs) == 0:
                return [], False
            llm_ranks, match_found = pass_to_chat_llm_chain(
                query, domain_specific_docs, llm_name=llm_name, domain=domain, threshold=belief_threshold
            )
            return post_process_candidates(llm_ranks, max=1), match_found
        else:
            return docs, False
    else:
        logger.info(f"No docs found for query={query}")
    return [], False


# def process_context(context,retriever,llm, domain = None, topk=10):
#     # context = normalize(context)
#     if context:
#         if docs :=  retriever_docs(context,retriever, domain='all',topk=topk):
#             if matched_docs := exact_match_found(query_text=context, documents=docs, domain=domain):
#                 return post_process_candidates(matched_docs, max=1)
#             llm_results,_ =  pass_to_chat_llm_chain(context, docs,llm_name=llm,domain=domain)
#             return post_process_candidates(llm_results, max=2)
#     return []


def process_values(values, retriever, llm, domain=None, values_type="additional", is_omop_data=False, topk=10):
    if isinstance(values, str):
        values = [values]
    logger.info(f"processing values={values}")
    all_values = {}
    if not values:
        return all_values
    for q_value in values:
        q_value = str(q_value).strip().lower()
        if q_value and q_value != "unknown":
            if categorical_value_results := retriever_docs(
                q_value, retriever, domain=domain, is_omop_data=is_omop_data, topk=topk
            ):
                pretty_print_docs(categorical_value_results)
                if matched_docs := exact_match_found(
                    query_text=q_value, documents=categorical_value_results, domain=domain
                ):
                    all_values[q_value] = post_process_candidates(matched_docs, max=1)
                elif categorical_value_results and len(categorical_value_results) > 0:
                    updated_results, _ = pass_to_chat_llm_chain(
                        q_value, categorical_value_results, llm_name=llm, domain=domain
                    )
                    if updated_results:
                        all_values[q_value] = post_process_candidates(updated_results, max=1)
            else:
                all_values[q_value] = [
                    RetrieverResultsModel(standard_label="na", standard_code=None, standard_omop_id=None, vocab=None)
                ]
    return all_values


def process_unit(unit, retriever, llm: Any, domain: str, is_omop_data: bool = False, topk: int = 10):
    unit_results = []
    if unit and unit != "unknown":
        unit_results = retriever_docs(unit, retriever, domain="unit", is_omop_data=is_omop_data, topk=topk)
        if unit_results:
            exact_units = exact_match_found(unit, unit_results, domain="unit")
            if len(exact_units) > 0:
                unit_results = post_process_candidates(exact_units, max=1)
            elif len(unit_results) > 0:
                llm_results, _ = pass_to_chat_llm_chain(unit, unit_results, llm_name=llm, domain=domain)
                unit_results = post_process_candidates(llm_results, max=1)
            else:
                unit_results = [
                    RetrieverResultsModel(standard_label="na", standard_code=None, standard_omop_id=None, vocab=None)
                ]

    return unit_results


def save_results(results, file_path):
    if results:
        # if file_path.endswith('.csv'):
        save_to_csv(results, file_path)
    # else:
    #     logger.info("No results to save.")


def filter_results(query, results):
    # pretty_print_docs(results)
    prioritized = []
    non_prioritized = []
    seen_metadata = []  # Use a list to track seen metadata and preserve order
    query = query.strip().lower()  # Normalize the query for comparison

    # First pass: collect prioritized and non-prioritized results
    for res in results:
        label = res.metadata["label"].strip().lower()  # Normalize the label for comparison
        metadata_str = f"{label}|{res.metadata['vocab']}|{res.metadata['scode']}|{res.metadata['sid']}"

        # Check if metadata has already been seen
        if metadata_str not in seen_metadata:
            seen_metadata.append(metadata_str)  # Mark metadata as seen

            if label == query:
                prioritized.append(res)  # Add to prioritized list if label matches the query
            else:
                non_prioritized.append(res)  # Add to non-prioritized list if label does not match

    # Combine prioritized and non-prioritized lists while preserving their original order
    combined_results = prioritized + non_prioritized
    # print(f"unique docs")
    # pretty_print_docs(combined_results)
    return combined_results


def map_csv_to_standard_codes(meta_path: str):
    """Map the data dictionary to standard codes using the LLM and save the results to a CSV file"""
    data, is_mapped = load_data(meta_path, load_custom=True)
    if is_mapped:
        return data
    embeddings = SAPEmbeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    hybrid_search = generate_vector_index(
        embeddings, sparse_embeddings, docs_file="", mode="inference", collection_name=SYN_COLLECTION_NAME, topk=10
    )
    # compressed_hybrid_retriever =  set_compression_retriever(hybrid_search)
    athena_api_retriever = initiate_api_retriever()
    merger_retriever = set_merger_retriever(retrievers=[hybrid_search, athena_api_retriever])
    merger_retriever = set_compression_retriever(merger_retriever)
    data = map_data(
        data,
        merger_retriever,
        custom_data=True,
        output_file=LOOK_UP_FILE,
        llm_name=LLM_ID,
        topk=TOPK,
        do_eval=False,
        is_omop_data=True,
    )
    print("DATA MAPPED", data)
    mapped_csv = append_results_to_csv(meta_path, data)
    mapped_csv.to_csv(meta_path, index=False)
    return mapped_csv
