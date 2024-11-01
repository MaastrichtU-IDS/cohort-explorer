from .vector_index import update_compressed_merger_retriever
import time
from .utils import global_logger as logger
from .utils import (
    pretty_print_docs,
    save_to_csv,
    create_result_dict,
    convert_db_result,
    create_processed_result,
    post_process_candidates,
    exact_match_found,
    filter_irrelevant_domain_candidates,
)
from .evalmap import perform_mapping_eval_for_variable
from .llm_chain import pass_to_chat_llm_chain, extract_information
from .eval import evaluate_with_multiple_mappings
from .py_model import QueryDecomposedModel, ProcessedResultsModel, RetrieverResultsModel
from .sql import DataManager
from .param import DB_FILE
from typing import Any

# Cache for retrievers based on domain
RETRIEVER_CACHE = {}

# def get_cached_retriever(retriever, domain, topk=10):
#     cache_key = (domain, topk)
#     if cache_key not in RETRIEVER_CACHE:
#         if domain != 'all':
#             retriever = update_merger_retriever(retriever, domain, topk=topk)
#         RETRIEVER_CACHE[cache_key] = retriever
#     return RETRIEVER_CACHE[cache_key]


# def retriever_docs(query, retriever, domain='all', is_omop_data=False, topk=10):
#     print(f"selected domain={domain}")
#     retriever = get_cached_retriever(retriever, domain, topk)
#     try:
#         results = retriever.invoke(query)
#         unique_results = filter_results(query, results)[:10]
#         print(f"length of unique results={len(unique_results)}")
#         return unique_results
#     except Exception as e:
#         logger.error(f"Error retrieving docs: {e}")
#         return None


def retriever_docs(query, retriever, domain="all", is_omop_data=False, topk=10):
    if is_omop_data:
        retriever = update_compressed_merger_retriever(retriever, domain, topk=topk)
    try:
        results = retriever.invoke(query)
        unique_results = filter_results(query, results)
        if unique_results:
            unique_results = unique_results[:10]
        else:
            unique_results = []
        print(f"length of unique results={len(unique_results)}")
        return unique_results
    except Exception as e:
        logger.error(f"Error retrieving docs: {e}")
        return []


def map_data(
    data,
    retriever,
    custom_data=False,
    output_file=None,
    llm_name="llama",
    topk=10,
    do_eval=False,
    is_omop_data=True,
):
    db = DataManager(DB_FILE)
    global RETRIEVER_CACHE
    start_time = time.time()
    max_queries = len(data)  # 282
    results = []
    for _, item in enumerate(data):
        query_obj = item[1]
        # query_result = full_query_processing(
        #     query_text=query_obj,
        #     retriever=retriever,
        #     llm_name=llm_name,
        #     is_omop_data=is_omop_data,
        #     topk=topk,
        # )
        query_result = full_query_processing_db(
            query_text=query_obj,
            retriever=retriever,
            llm_name=llm_name,
            is_omop_data=is_omop_data,
            topk=topk,
            datamanager=db,
        )
        if query_result:
            query_result = perform_mapping_eval_for_variable(
                var_=query_result, llm_id=llm_name
            )
            if query_result["prediction"].strip().lower() == "correct":
                print(db.insert_row(query_result))
            results.append(query_result)
        else:
            logger.info(f"NO RESULT FOR {item}")
        # time.sleep(0.05)  # Adjusted to be more appropriate than 0.0005
    db.close_connection()
    end_time = time.time()
    total_time = end_time - start_time
    save_results(results, output_file)
    if do_eval:
        return evaluate_with_multiple_mappings(
            data[:max_queries], results, model_name=llm_name
        )
    logger.info(
        f"Total execution time for {max_queries} queries is {total_time} seconds."
    )
    return results


def full_query_processing_db(
    query_text: QueryDecomposedModel,
    retriever: Any,
    llm_name: str,
    topk: int,
    is_omop_data=True,
    datamanager: DataManager = None,
):
    try:
        logger.info(f"Processing query: {query_text}")

        if query_text is None:
            return {}
        else:
            print(f"query_text={query_text.name}")
            results, mode = datamanager.query_variable(query_text.original_label)
            if mode == "full" and len(results) >= 4:
                logger.info(f"Found results for {query_text} in RESERVOIR")
                return create_result_dict(results)

            query_decomposed = extract_information(query_text.full_query, llm_name)
            logger.info(f"Query decomposed:{query_decomposed}")
            processes_results = temp_process_query_details_db(
                llm_query_obj=query_decomposed,
                retriever_cache=retriever,
                llm_name=llm_name,
                topk=topk,
                original_query_obj=query_text,
                is_omop_data=is_omop_data,
                db=datamanager,
            )

            # logger.info("Mapping result:", processes_results)
            return processes_results
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return {}


def full_query_processing(
    query_text: QueryDecomposedModel,
    retriever: Any,
    llm_name: str,
    topk: int,
    is_omop_data=True,
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
            return {}
        else:
            print(f"query_text={query_text.name}")
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
            logger.info("Mapping result:", processes_results)
            return processes_results
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return {}


# def create_input_dict(query_text):
#     query_dict = {}
#     rest_term = query_text
#     query_dict['base_entity'] = query_text

#     # Check if there are categorical values in the text
#     if 'categorical values' in query_text:
#         main_term = rest_term.split('|categorical values:')[0].strip()
#         query_dict['base_entity'] = main_term  # Extract base entity
#         values = rest_term.split('|categorical values:')[1].strip()  # Extract categorical values
#         query_dict['categories'] = [value.strip() for value in values.strip().replace(',', '|').split('|')]  # Convert categories to a list
#     else:
#         # If no categorical values, ensure main entity is parsed properly
#         main_term = rest_term.split('|')[0].strip()
#         query_dict['base_entity'] = main_term
#     # Check if there is a unit in the text
#     if 'unit' in rest_term:
#         unit = rest_term.split('|unit:')[1].strip()  # Extract unit without further splitting
#         query_dict['unit'] = unit

#     # Check if there is a formula in the text
#     if 'formula' in rest_term:
#         formula = rest_term.split('|formula:')[1].strip()  # Extract formula
#         query_dict['formula'] = formula
#     return query_dict


def temp_process_query_details_db(
    llm_query_obj: QueryDecomposedModel,
    retriever_cache: Any,
    llm_name: str,
    topk: int,
    original_query_obj: QueryDecomposedModel,
    is_omop_data=False,
    db: DataManager = None,
):
    try:
        if llm_query_obj:
            original_query_obj.domain = llm_query_obj.domain
            original_query_obj.rel = llm_query_obj.rel
            original_query_obj.unit = llm_query_obj.unit
            original_query_obj.categories = llm_query_obj.categories
            logger.info(f"original query Obj:{original_query_obj}")
            (
                variable_label_matches,
                additional_entities_matches,
                categories_matches,
                unit_matches,
            ) = None, None, None, None
            base_entity = original_query_obj.base_entity
            domain = original_query_obj.domain

            variable_label_matches, found_match = process_retrieved_docs(
                base_entity,
                retriever_docs(
                    original_query_obj.base_entity,
                    retriever_cache,
                    domain="all",
                    is_omop_data=is_omop_data,
                    topk=topk,
                ),
                llm_name,
                "all",
                belief_threshold=0.85,
            )
            if found_match and len(variable_label_matches) > 0:
                main_term = (
                    base_entity  # Assign base_entity to main_term if a match is found
                )
                llm_query_obj = original_query_obj
            else:
                result, mode = db.query_variable(base_entity)
                if result and mode == "subset":
                    variable_label_matches, found_match = (
                        convert_db_result(result),
                        True,
                    )
                else:
                    print("proceeding to structured completion format")
                    main_term = (
                        llm_query_obj.base_entity if llm_query_obj else base_entity
                    )
                    variable_label_matches, _ = process_retrieved_docs(
                        main_term,
                        retriever_docs(
                            main_term,
                            retriever_cache,
                            domain=llm_query_obj.domain,
                            is_omop_data=is_omop_data,
                            topk=topk,
                        ),
                        llm_name,
                        llm_query_obj.domain,
                    )
                    llm_query_obj.name = original_query_obj.name
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
                    process_values_db(
                        main_term,
                        additional_entities,
                        retriever_cache,
                        llm_name,
                        domain="observation",
                        values_type="additional",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    )
                    if additional_entities
                    else {}
                )
            if categories:
                logger.info(f"Processing categories: {categories}")
                categories_matches = (
                    process_values_db(
                        main_term,
                        categories,
                        retriever_cache,
                        llm_name,
                        domain="all",
                        values_type="categories",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    )
                    if categories
                    else {}
                )
            # if rel:
            #     rel_docs = process_values([rel], retriever_cache, llm_name, domain='all', values_type='rel')[rel] if rel else []
            if unit:
                logger.info(f"Processing unit: {unit}")
                unit_matches = (
                    process_unit_db(
                        unit,
                        retriever_cache,
                        llm=llm_name,
                        domain="unit",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    )
                    if unit
                    else []
                )
            mapping_result = create_processed_result(
                ProcessedResultsModel(
                    variable_name=llm_query_obj.name,
                    base_entity=main_term,
                    domain=domain,
                    base_entity_matches=variable_label_matches
                    if variable_label_matches
                    else [],
                    categories=categories,
                    categories_matches=categories_matches if categories_matches else {},
                    unit=unit,
                    unit_matches=unit_matches if unit_matches else [],
                    original_query=llm_query_obj.original_label,
                    additional_entities=additional_entities,
                    primary_to_secondary_rel=rel,
                    additional_entities_matches=additional_entities_matches
                    if additional_entities_matches
                    else {},
                )
            )
            logger.info(f"Mapping result: {mapping_result}")
            return mapping_result
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return {}


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
            (
                variable_label_matches,
                additional_entities_matches,
                categories_matches,
                unit_matches,
            ) = None, None, None, None
            base_entity = original_query_obj.base_entity
            domain = original_query_obj.domain
            variable_label_matches, found_match = process_retrieved_docs(
                base_entity,
                retriever_docs(
                    original_query_obj.base_entity,
                    retriever_cache,
                    domain="all",
                    is_omop_data=is_omop_data,
                    topk=topk,
                ),
                llm_name,
                "all",
                belief_threshold=0.85,
            )
            if found_match and len(variable_label_matches) > 0:
                main_term = (
                    base_entity  # Assign base_entity to main_term if a match is found
                )
                llm_query_obj = original_query_obj
            else:
                print("proceeding to structured completion format")
                main_term = llm_query_obj.base_entity if llm_query_obj else base_entity
                variable_label_matches, _ = process_retrieved_docs(
                    main_term,
                    retriever_docs(
                        main_term,
                        retriever_cache,
                        domain=llm_query_obj.domain,
                        is_omop_data=is_omop_data,
                        topk=topk,
                    ),
                    llm_name,
                    llm_query_obj.domain,
                )
                llm_query_obj.name = original_query_obj.name
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
                        main_term,
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
                        main_term,
                        categories,
                        retriever_cache,
                        llm_name,
                        domain="all",
                        values_type="categories",
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
                        unit,
                        retriever_cache,
                        llm=llm_name,
                        domain="unit",
                        is_omop_data=is_omop_data,
                        topk=topk,
                    )
                    if unit
                    else []
                )
            mapping_result = create_processed_result(
                ProcessedResultsModel(
                    variable_name=llm_query_obj.name,
                    base_entity=main_term,
                    domain=domain,
                    base_entity_matches=variable_label_matches
                    if variable_label_matches
                    else [],
                    categories=categories,
                    categories_matches=categories_matches if categories_matches else {},
                    unit=unit,
                    unit_matches=unit_matches if unit_matches else [],
                    original_query=llm_query_obj.original_label,
                    additional_entities=additional_entities,
                    primary_to_secondary_rel=rel,
                    additional_entities_matches=additional_entities_matches
                    if additional_entities_matches
                    else {},
                )
            )
            logger.info(f"Mapping result: {mapping_result}")
            return mapping_result
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return {}


def process_retrieved_docs(
    query, docs, llm_name=None, domain=None, belief_threshold=0.8
):
    # logger.info_docs(docs)
    if docs and len(docs) > 0:
        if matched_docs := exact_match_found(
            query_text=query, documents=docs, domain=domain
        ):
            return post_process_candidates(matched_docs, max=1), True
        if llm_name:
            logger.info(f"No string match found for {query} pass to {llm_name}")
            domain_specific_docs = filter_irrelevant_domain_candidates(docs, domain)
            if domain_specific_docs is None or len(domain_specific_docs) == 0:
                return [], False
            llm_ranks, match_found = pass_to_chat_llm_chain(
                query,
                domain_specific_docs,
                llm_name=llm_name,
                domain=domain,
                threshold=belief_threshold,
            )
            print(
                f"number of candidates={len(llm_ranks)} and exact match={match_found}"
            )
            return post_process_candidates(llm_ranks, max=1), match_found
        else:
            return docs, False
    else:
        logger.info(f"No docs found for query={query}")
    return [], False


def process_context(context, retriever, llm, domain=None, topk=10):
    # context = normalize(context)
    if context:
        if docs := retriever_docs(context, retriever, domain="all", topk=topk):
            if matched_docs := exact_match_found(
                query_text=context, documents=docs, domain=domain
            ):
                return post_process_candidates(matched_docs, max=1)
            llm_results, _ = pass_to_chat_llm_chain(
                context, docs, llm_name=llm, domain=domain
            )
            return post_process_candidates(llm_results, max=2)
    return []


def process_values_db(
    main_term,
    values,
    retriever,
    llm,
    domain=None,
    values_type="additional",
    is_omop_data=False,
    topk=10,
    db: DataManager = None,
):
    if isinstance(values, str):
        values = [values]
    logger.info(f"processing values={values}")
    all_values = {}
    if not values:
        return all_values
    for q_value in values:
        q_value = str(q_value).strip().lower()
        if q_value:
            result, mode = db.query_variable(q_value)
            if result and mode == "subset":
                print(f"found value in RESERVOIR={q_value}")
                all_values[q_value] = convert_db_result(result)
            else:
                contextaware_categorical_value_results = []
                categorical_value_results = retriever_docs(
                    q_value,
                    retriever,
                    domain=domain,
                    is_omop_data=is_omop_data,
                    topk=topk,
                )
                if values_type == "additional":
                    contextaware_categorical_value_results = retriever_docs(
                        f"{main_term}, {q_value}",
                        retriever,
                        domain=domain,
                        is_omop_data=is_omop_data,
                        topk=topk,
                    )[:5]
                if categorical_value_results:
                    categorical_value_results += contextaware_categorical_value_results
                    pretty_print_docs(categorical_value_results)
                    if matched_docs := exact_match_found(
                        query_text=q_value,
                        documents=categorical_value_results,
                        domain=domain,
                    ):
                        # max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                        # logger.info(f"max_results={max_results} for {updated_q_value}")
                        all_values[q_value] = post_process_candidates(
                            matched_docs, max=1
                        )
                    elif (
                        categorical_value_results and len(categorical_value_results) > 0
                    ):
                        if values_type == "additional":
                            q_value_ = f"{q_value}, context: {main_term}"
                        else:
                            q_value_ = q_value
                        updated_results, _ = pass_to_chat_llm_chain(
                            q_value_,
                            categorical_value_results,
                            llm_name=llm,
                            domain=domain,
                        )
                        if updated_results:
                            # max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                            # logger.info(f"max_results={max_results} for {updated_q_value}")
                            all_values[q_value] = post_process_candidates(
                                updated_results, max=1
                            )
        elif values_type == "categories":
            all_values[q_value] = [
                RetrieverResultsModel(
                    standard_label="na",
                    standard_code="na",
                    standard_omop_id="na",
                    vocab="na",
                )
            ]
    return all_values


def process_values(
    main_term,
    values,
    retriever,
    llm,
    domain=None,
    values_type="additional",
    is_omop_data=False,
    topk=10,
):
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
                    query_text=q_value,
                    documents=categorical_value_results,
                    domain=domain,
                ):
                    # max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                    # logger.info(f"max_results={max_results} for {updated_q_value}")
                    all_values[q_value] = post_process_candidates(matched_docs, max=1)
                elif categorical_value_results and len(categorical_value_results) > 0:
                    if values_type == "additional":
                        q_value_ = f"{q_value}, context: {main_term}"
                    else:
                        q_value_ = q_value
                    updated_results, _ = pass_to_chat_llm_chain(
                        q_value_, categorical_value_results, llm_name=llm, domain=domain
                    )
                    if updated_results:
                        # max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                        # logger.info(f"max_results={max_results} for {updated_q_value}")
                        all_values[q_value] = post_process_candidates(
                            updated_results, max=1
                        )
        elif values_type == "categories":
            all_values[q_value] = [
                RetrieverResultsModel(
                    standard_label="na",
                    standard_code=None,
                    standard_omop_id=None,
                    vocab=None,
                )
            ]
    return all_values


def process_unit_db(
    unit,
    retriever,
    llm: Any,
    domain: str,
    is_omop_data: bool = False,
    topk: int = 10,
    db: DataManager = None,
):
    unit_results = []
    if unit and unit != "unknown":
        result, model = db.query_variable(unit)
        if result and model == "subset":
            unit_results = convert_db_result(result)
            print(f"found unit in RESERVOIR={unit}")
            return unit_results
        unit_results = retriever_docs(
            unit, retriever, domain="unit", is_omop_data=is_omop_data, topk=topk
        )
        if unit_results:
            exact_units = exact_match_found(unit, unit_results, domain="unit")
            if len(exact_units) > 0:
                unit_results = post_process_candidates(exact_units, max=1)
            elif len(unit_results) > 0:
                llm_results, _ = pass_to_chat_llm_chain(
                    unit, unit_results, llm_name=llm, domain=domain
                )
                unit_results = post_process_candidates(llm_results, max=1)
            else:
                unit_results = [
                    RetrieverResultsModel(
                        standard_label="na",
                        standard_code=None,
                        standard_omop_id=None,
                        vocab=None,
                    )
                ]

    return unit_results


def process_unit(
    unit, retriever, llm: Any, domain: str, is_omop_data: bool = False, topk: int = 10
):
    unit_results = []
    if unit and unit != "unknown":
        unit_results = retriever_docs(
            unit, retriever, domain="unit", is_omop_data=is_omop_data, topk=topk
        )
        if unit_results:
            exact_units = exact_match_found(unit, unit_results, domain="unit")
            if len(exact_units) > 0:
                unit_results = post_process_candidates(exact_units, max=1)
            elif len(unit_results) > 0:
                llm_results, _ = pass_to_chat_llm_chain(
                    unit, unit_results, llm_name=llm, domain=domain
                )
                unit_results = post_process_candidates(llm_results, max=1)
            else:
                unit_results = [
                    RetrieverResultsModel(
                        standard_label="na",
                        standard_code=None,
                        standard_omop_id=None,
                        vocab=None,
                    )
                ]

    return unit_results


def save_results(results, file_path):
    if results:
        # if file_path.endswith('.csv'):
        save_to_csv(results, file_path)
    else:
        logger.info("No results to save.")


# def update_api_search_filter(api_retriever, domain='observation'):

#     if domain == 'unit':
#          api_retriever.filters = AthenaFilters(domain=None, vocabulary=['UCUM'], standard_concept=['Standard'])
#     elif domain == 'condition' or domain == 'anatomic site':
#         api_retriever.filters  = AthenaFilters(domain=['Condition','Meas Value','Spec Anatomic Site'], vocabulary=['SNOMED'],standard_concept=['Standard'])
#     elif domain == 'measurement':
#         api_retriever.filters  = AthenaFilters(domain=['Measurement','Meas Value','Observation'], vocabulary=['LOINC','MeSH','SNOMED'],standard_concept=['Standard'])
#     elif domain == 'drug':
#         api_retriever.filters  = AthenaFilters(domain=['Drug'], vocabulary=['RxNorm','ATC','SNOMED'], standard_concept=['Standard','Classification'])
#     elif domain == 'observation':
#         api_retriever.filters  = AthenaFilters(domain=['Observation','Meas Value'], vocabulary=['SNOMED','LOINC','OMOP Extension'])
#     elif domain == 'visit':
#         api_retriever.filters  = AthenaFilters(domain=['Visit','Observation'], vocabulary=['SNOMED','LOINC','OMOP Extension'])
#     elif domain == 'demographics':
#         api_retriever.filters  = AthenaFilters(domain=['Observation','Meas Value'], vocabulary=['SNOMED','LOINC','OMOP Extension','Gender','Race','Ethnicity'])
#     else:
#         api_retriever.filters  = AthenaFilters()
#     return api_retriever


# def update_qdrant_search_filter(retriever, domain='unknown'):
#     if domain == 'unit':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchValue(value='ucum')
#                         )
#                     ]
#                 )
#     elif domain == 'condition' or domain == 'anatomic site':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchAny(any=['snomed'])

#                         )
#                     ]
#                 )
#     elif domain == 'demographics':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchAny(any=['snomed','loinc'])
#                         ),
#                         rest.FieldCondition(
#                             key="metadata.domain",
#                             match=rest.MatchAny(any=['observation','meas value'])
#                         )
#                     ]
#                 )
#     elif domain == 'measurement':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchAny(any=['loinc','mesh','snomed'])
#                         ),
#                         rest.FieldCondition(
#                             key="metadata.domain",
#                             match=rest.MatchAny(any=['measurement','meas value','observation'])
#                         )
#                     ]
#                 )
#     elif domain == 'drug':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchAny(any=['rxnorm','atc','snomed'])
#                         ),
#                         rest.FieldCondition(
#                             key="metadata.is_standard",
#                             match=rest.MatchAny(any=['S','C'])
#                         )
#                     ]
#                 )
#     elif domain == 'observation' or domain == 'visit' or domain == 'demographics' or domain == 'history of events' or domain == 'life style':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchAny(any=['snomed','loinc','omop extension'])
#                         ),
#                         rest.FieldCondition(
#                             key="metadata.domain",
#                             match=rest.MatchAny(any=['observation','meas value'])
#                         )
#                     ]
#         )
#     elif domain == 'procedure':
#         retriever.search_kwargs['filter'] = rest.Filter(
#                     must=[
#                         rest.FieldCondition(
#                             key="metadata.vocab",
#                             match=rest.MatchAny(any=['snomed'])
#                         )
#                     ]
#                 )
#     else:
#         retriever.search_kwargs['filter'] = None
#     return retriever


def filter_results(query, results):
    # pretty_print_docs(results)
    prioritized = []
    non_prioritized = []
    seen_metadata = []  # Use a list to track seen metadata and preserve order
    query = query.strip().lower()  # Normalize the query for comparison

    # First pass: collect prioritized and non-prioritized results
    for res in results:
        label = (
            res.metadata["label"].strip().lower()
        )  # Normalize the label for comparison
        metadata_str = f"{label}|{res.metadata['vocab']}|{res.metadata['scode']}|{res.metadata['sid']}"

        # Check if metadata has already been seen
        if metadata_str not in seen_metadata:
            seen_metadata.append(metadata_str)  # Mark metadata as seen

            if label == query:
                prioritized.append(
                    res
                )  # Add to prioritized list if label matches the query
            else:
                non_prioritized.append(
                    res
                )  # Add to non-prioritized list if label does not match

    # Combine prioritized and non-prioritized lists while preserving their original order
    combined_results = prioritized + non_prioritized
    print("unique docs")
    pretty_print_docs(combined_results)
    return combined_results
