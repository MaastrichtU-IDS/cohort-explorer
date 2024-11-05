from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from .utils import load_mapping
from .utils import global_logger as logger
from pydantic.v1 import ValidationError
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from typing import List, Dict, Any
import time
from langchain_core.callbacks.base import BaseCallbackHandler
from .utils import fix_json_quotes, create_document_string
from collections import defaultdict
import os
from .param import MAPPING_FILE, LLM_ID
from .py_model import QueryDecomposedModel, sanitize_keys
from .manager_llm import LLMManager, ExampleSelectorManager

parsing_llm = LLMManager.get_instance("llama3.1")
parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(
    parser=parser, llm=parsing_llm, max_retries=3
)
REQUEST_LIMIT = 30
TIME_WINDOW = 60
# from langchain_core.tracers.stdout import ConsoleCallbackHandler
# import langchain
# langchain.verbose = True
# # example_selector: Optional[SemanticSimilarityExampleSelector] = None
# class CustomHandler(BaseCallbackHandler):
#     def on_llm_start(
#         self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
#     ) -> Any:
#         formatted_prompts = "\n".join(prompts)
#         _log.info(f"Prompt:\n{formatted_prompts}")

# def is_history_relevant(history_message, query):
#     content = history_message.content
#     if match := re.match(r'query:(.*?), output:(.*)', content):
#          input_ = match.group(1).strip()
#          score = reranker.compute_score([input_,query],normalize=True)
#          logger.info(f"History Relevance score={score}")
#          if score > 0.9:
#              return True
#     return False

# # Keep track of request timestamps
# request_timestamps = deque(maxlen=30)

# def rate_limit_check(request_limit: int = REQUEST_LIMIT, time_window: int = TIME_WINDOW):
#     """
#     Limit the number of requests to `request_limit` in a given `time_window` (in seconds).
#     If the limit is reached, the function will sleep until a request is allowed.

#     :param request_limit: Maximum number of requests allowed in the time window.
#     :param time_window: Time window in seconds (e.g., 60 for one minute).
#     """
#     current_time = time.time()

#     # Remove timestamps older than the time window
#     while request_timestamps and current_time - request_timestamps[0] > time_window:
#         request_timestamps.popleft()

#     # Check if we are within the request limit
#     if len(request_timestamps) >= request_limit:
#         if request_timestamps:
#             # Calculate how long to sleep until the first timestamp is outside the time window
#             time_to_wait = time_window - (current_time - request_timestamps[0])
#             print(f"Rate limit reached. Sleeping for {time_to_wait:.2f} seconds.")

#             time.sleep(300)

#     # After sleeping (or if no sleep was needed), log the new request
#     request_timestamps.append(time.time())


def get_relevant_examples(
    query: str,
    content_key: str,
    examples: List[Dict[str, str]],
    topk=2,
    min_score=0.5,
    selector_path=None,
) -> List[Dict]:
    try:
        # Obtain the singleton example selector
        if examples is None or len(examples) == 0:
            logger.info("No examples found")
            return []
        selector = ExampleSelectorManager.get_example_selector(
            content_key, examples, k=topk, score_threshold=min_score
        )

        selected_examples = selector.select_examples({"input": f"{query}"})

        return selected_examples
    except Exception as e:
        logger.info(
            f"Error in get_relevant_examples: {e} for query:{query} and content_key:{content_key}"
        )
        return []


# def extract_ir(base_entity, associated_entities, active_model):
#     if base_entity is None or associated_entities is None or len(associated_entities) == 0:
#         return None
#     relations = ['Is attribute of','Has specimen procedure', 'Has specimen source identity', 'Has specimen source morphology', 'Has specimen source topography', 'Has specimen substance', 'Has due to', 'Has subject relationship context', 'Has dose form', 'Occurs after', 'Has associated procedure', 'Has direct procedure site', 'Has indirect procedure site', 'Has procedure device', 'Has procedure morphology', 'Has finding context', 'Has procedure context', 'Has temporal context', 'Associated with finding', 'Has surgical approach', 'Using device', 'Using energy', 'Using substance', 'Using access device', 'Has clinical course', 'Has route of administration', 'Using finding method', 'Using finding informer', 'Has off-label drug indication', 'Has drug contra-indication', 'Precise ingredient of', 'Tradename of', 'Dose form of', 'Form of', 'Ingredient of', 'Consists of', 'Is contained in', 'Reformulated in', 'Recipient category of', 'Procedure site of', 'Priority of', 'Pathological process of', 'Part of', 'Severity of', 'Revision status of', 'Access of', 'Occurrence of', 'Laterality of', 'Interprets of', 'Indirect morphology of', 'Is a', 'Indirect device of', 'Specimen of', 'Interpretation of', 'Intent of', 'Focus of', 'Definitional manifestation of', 'Active ingredient of', 'Finding site of', 'Episodicity of', 'Direct substance of', 'Direct morphology of', 'Direct device of', 'Causative agent of', 'Associated morphology of', 'Associated finding of', 'Measurement method of', 'Specimen procedure of', 'Specimen source identity of', 'Specimen source morphology of', 'Specimen source topography of', 'Specimen substance of', 'Due to of', 'Subject relationship context of', 'Dose form of', 'Occurs before', 'Associated procedure of', 'Direct procedure site of', 'Indirect procedure site of', 'Procedure device of', 'Procedure morphology of', 'Finding context of', 'Procedure context of', 'Temporal context of', 'Finding associated with', 'Surgical approach of', 'Device used by', 'Energy used by', 'Substance used by', 'Access device used by', 'Has clinical course of', 'Route of administration of', 'Finding method of', 'Finding informer of', 'Is off-label indication of', 'Is contra-indication of', 'Has ingredient', 'Ingredient of', 'Module of', 'Has Extent', 'Extent of', 'Has Approach', 'Has therapeutic class', 'Therapeutic class of', 'Drug-drug interaction for', 'Is involved in drug-drug interaction', 'Has pharmaceutical preparation', 'Pharmaceutical preparation contained in', 'Approach of', 'Has quantified form', 'Has dispensed dose form', 'Dispensed dose form of', 'Has specific active ingredient', 'Specific active ingredient of', 'Has basis of strength substance', 'Basis of strength substance of', 'Has Virtual Medicinal Product', 'Virtual Medicinal Product of', 'Has Answer', 'Answer of', 'Has Actual Medicinal Product', 'Actual Medicinal Product of', 'Is pack of', 'Has pack', 'Has trade family group', 'Trade family group of', 'Has excipient', 'Excipient of', 'Follows', 'Followed by', 'Has discontinued indicator', 'Discontinued indicator of', 'Has legal category', 'Legal category of', 'Dose form group of', 'Has dose form group', 'Has precondition', 'Precondition of', 'Has inherent location', 'Inherent location of', 'Has technique', 'Technique of', 'Has relative part', 'Relative part of', 'Has process output', 'Process output of', 'Inheres in', 'Has inherent', 'Has direct site', 'Direct site of', 'Characterizes', 'Has property type', 'Property type of', 'Panel contains', 'Contained in panel', 'Is characterized by', 'Has Module', 'Topic of', 'Has Topic', 'Has presentation strength numerator unit', 'Presentation strength numerator unit of', 'During', 'Has complication', 'Has basic dose form', 'Basic dose form of', 'Has disposition', 'Disposition of', 'Has dose form administration method', 'Dose form administration method of', 'Has dose form intended site', 'Dose form intended site of', 'Has dose form release characteristic', 'Dose form release characteristic of', 'Has dose form transformation', 'Dose form transformation of', 'Has state of matter', 'State of matter of', 'Temporally related to', 'Has temporal finding', 'Has Morphology', 'Morphology of', 'Has Measured Component', 'Measured Component of', 'Caused by', 'Causes', 'Has Etiology', 'Etiology of', 'Has Stage', 'Stage of', 'Quantified form of', 'Is a', 'Inverse is a', 'Has precise ingredient', 'Has tradename', 'Has dose form', 'Has form', 'Has ingredient', 'Constitutes', 'Contains', 'Reformulation of', 'Subsumes', 'Has recipient category', 'Has procedure site', 'Has priority', 'Has pathological process', 'Has part of', 'Has severity', 'Has revision status', 'Has access', 'Has occurrence', 'Has laterality', 'Has interprets', 'Has indirect morphology', 'Has indirect device', 'Has specimen', 'Has interpretation', 'Has intent', 'Has focus', 'Has definitional manifestation', 'Has active ingredient', 'Has finding site', 'Has episodicity', 'Has direct substance', 'Has direct morphology', 'Has direct device', 'Has causative agent', 'Has associated morphology', 'Has associated finding', 'Has measurement method', 'Has precise active ingredient', 'Precise active ingredient of', 'Has scale type', 'Has property', 'Concentration strength numerator unit of', 'Is modification of', 'Has modification of', 'Has unit', 'Unit of', 'Has method', 'Method of', 'Has time aspect', 'Time aspect of', 'Has component', 'Has end date', 'End date of', 'Has start date', 'Start date of', 'Has system', 'System of', 'Process duration', 'Process duration of', 'Has precoordinated (Question-Answer/Variable-Value) pair', 'Precoordinated (Question-Answer/Variable-Value) pair of', 'Has Category', 'Category of', 'Has biosimilar', 'Biosimilar of', 'Relative to', 'Relative to of', 'Count of active ingredients', 'Is count of active ingredients in', 'Has product characteristic', 'Product characteristic of', 'Has surface characteristic', 'Surface characteristic of', 'Has device intended site', 'Device intended site of', 'Has compositional material', 'Compositional material of', 'Has filling', 'Filling material of', 'Reference to variant', 'Variant refer to concept', 'Genomic DNA transcribes to mRNA', 'mRNA Translates to protein', 'mRNA is transcribed from genomic DNA', 'Protein is translated from mRNA', 'Has coating material', 'Coating material of', 'Has absorbability', 'Absorbability of', 'Process extends to', 'Process extends from', 'Has ingredient qualitative strength', 'Ingredient qualitative strength of', 'Has surface texture', 'Surface texture of', 'Is sterile', 'Is sterile of', 'Has target population', 'Target population of', 'Has status', 'Status of', 'Process acts on', 'Affected by process', 'Before', 'After', 'Towards', 'Subject of']
#     extracted_relations = []
#     # Refined prompt with examples
#     for secondary_entity in associated_entities:
#         base_prompt = base_prompt = f"""

#         Given the **Base Entity** (primary concept) and **associated entity** (secondary concepts), select the most appropriate relationship from the provided options.

#             **Instructions:**
#             - The relationship should describe how the **Base Entity** relates **to** the **Associated Entity**.
#             - Review the Base Entity and each Associated Entity.
#             - Use the examples below to guide your selection.
#             - Choose the relationship that best fits the direction from Base Entity to Associated Entity.
#             - Return only one relationship name as a string.
#             - Do not include any explanations or additional text. Do not use external resources.

#             **Examples:**
#                 1. Base Entity: 'chronic obstructive pulmonary disease (COPD)'
#                 Associated Entity: 'cigarette smoking'
#                 Selected Relationship: 'Has Causative Agent'

#                 2. Base Entity: 'atrial fibrillation'
#                 Associated Entity: 'left atrial appendage
#                 Selected Relationship: 'Finding site of'
#                 3 . Base Entity:diastolic phase,
#                 Associated Entity:=['longitudinal echocardiography', 'pre-discharge assessment', 'Follow-up 1 month']
#                 Selected Relationship:[]
#                 Now, apply the same logic to the following:
#                 Base Entity: {base_entity}
#                 Associated Entities: {secondary_entity}
#                 **Relationship Options:**: {', '.join([rel.lower() for rel in relations])}
#                 """
#         system = "You are a helpful assistant with expertise in the biomedical domain."
#         final_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", base_prompt)])
#         chain = final_prompt | active_model
#         result = chain.invoke({"base_entity": base_entity, "secondary_entity": secondary_entity, "relations": relations}).content
#         print(f"extract_ir result={result.strip()}")
#         extracted_relations.append({
#             "subject": base_entity,
#             "predicate": result.strip(),
#             "object": secondary_entity
#         })
#     return extracted_relations


def extract_ir(base_entity, associated_entities, active_model):
    if (
        base_entity is None
        or associated_entities is None
        or len(associated_entities) == 0
    ):
        return None
    relations = [
        "Is attribute of",
        "Has specimen procedure",
        "Has specimen source identity",
        "Has specimen source morphology",
        "Has specimen source topography",
        "Has specimen substance",
        "Has due to",
        "Has subject relationship context",
        "Has dose form",
        "Occurs after",
        "Has associated procedure",
        "Has direct procedure site",
        "Has indirect procedure site",
        "Has procedure device",
        "Has procedure morphology",
        "Has finding context",
        "Has procedure context",
        "Has temporal context",
        "Associated with finding",
        "Has surgical approach",
        "Using device",
        "Using energy",
        "Using substance",
        "Using access device",
        "Has clinical course",
        "Has route of administration",
        "Using finding method",
        "Using finding informer",
        "Has off-label drug indication",
        "Has drug contra-indication",
        "Precise ingredient of",
        "Tradename of",
        "Dose form of",
        "Form of",
        "Ingredient of",
        "Consists of",
        "Is contained in",
        "Reformulated in",
        "Recipient category of",
        "Procedure site of",
        "Priority of",
        "Pathological process of",
        "Part of",
        "Severity of",
        "Revision status of",
        "Access of",
        "Occurrence of",
        "Laterality of",
        "Interprets of",
        "Indirect morphology of",
        "Is a",
        "Indirect device of",
        "Specimen of",
        "Interpretation of",
        "Intent of",
        "Focus of",
        "Definitional manifestation of",
        "Active ingredient of",
        "Finding site of",
        "Episodicity of",
        "Direct substance of",
        "Direct morphology of",
        "Direct device of",
        "Causative agent of",
        "Associated morphology of",
        "Associated finding of",
        "Measurement method of",
        "Specimen procedure of",
        "Specimen source identity of",
        "Specimen source morphology of",
        "Specimen source topography of",
        "Specimen substance of",
        "Due to of",
        "Subject relationship context of",
        "Dose form of",
        "Occurs before",
        "Associated procedure of",
        "Direct procedure site of",
        "Indirect procedure site of",
        "Procedure device of",
        "Procedure morphology of",
        "Finding context of",
        "Procedure context of",
        "Temporal context of",
        "Finding associated with",
        "Surgical approach of",
        "Device used by",
        "Energy used by",
        "Substance used by",
        "Access device used by",
        "Has clinical course of",
        "Route of administration of",
        "Finding method of",
        "Finding informer of",
        "Is off-label indication of",
        "Is contra-indication of",
        "Has ingredient",
        "Ingredient of",
        "Module of",
        "Has Extent",
        "Extent of",
        "Has Approach",
        "Has therapeutic class",
        "Therapeutic class of",
        "Drug-drug interaction for",
        "Is involved in drug-drug interaction",
        "Has pharmaceutical preparation",
        "Pharmaceutical preparation contained in",
        "Approach of",
        "Has quantified form",
        "Has dispensed dose form",
        "Dispensed dose form of",
        "Has specific active ingredient",
        "Specific active ingredient of",
        "Has basis of strength substance",
        "Basis of strength substance of",
        "Has Virtual Medicinal Product",
        "Virtual Medicinal Product of",
        "Has Answer",
        "Answer of",
        "Has Actual Medicinal Product",
        "Actual Medicinal Product of",
        "Is pack of",
        "Has pack",
        "Has trade family group",
        "Trade family group of",
        "Has excipient",
        "Excipient of",
        "Follows",
        "Followed by",
        "Has discontinued indicator",
        "Discontinued indicator of",
        "Has legal category",
        "Legal category of",
        "Dose form group of",
        "Has dose form group",
        "Has precondition",
        "Precondition of",
        "Has inherent location",
        "Inherent location of",
        "Has technique",
        "Technique of",
        "Has relative part",
        "Relative part of",
        "Has process output",
        "Process output of",
        "Inheres in",
        "Has inherent",
        "Has direct site",
        "Direct site of",
        "Characterizes",
        "Has property type",
        "Property type of",
        "Panel contains",
        "Contained in panel",
        "Is characterized by",
        "Has Module",
        "Topic of",
        "Has Topic",
        "Has presentation strength numerator unit",
        "Presentation strength numerator unit of",
        "During",
        "Has complication",
        "Has basic dose form",
        "Basic dose form of",
        "Has disposition",
        "Disposition of",
        "Has dose form administration method",
        "Dose form administration method of",
        "Has dose form intended site",
        "Dose form intended site of",
        "Has dose form release characteristic",
        "Dose form release characteristic of",
        "Has dose form transformation",
        "Dose form transformation of",
        "Has state of matter",
        "State of matter of",
        "Temporally related to",
        "Has temporal finding",
        "Has Morphology",
        "Morphology of",
        "Has Measured Component",
        "Measured Component of",
        "Caused by",
        "Causes",
        "Has Etiology",
        "Etiology of",
        "Has Stage",
        "Stage of",
        "Quantified form of",
        "Is a",
        "Inverse is a",
        "Has precise ingredient",
        "Has tradename",
        "Has dose form",
        "Has form",
        "Has ingredient",
        "Constitutes",
        "Contains",
        "Reformulation of",
        "Subsumes",
        "Has recipient category",
        "Has procedure site",
        "Has priority",
        "Has pathological process",
        "Has part of",
        "Has severity",
        "Has revision status",
        "Has access",
        "Has occurrence",
        "Has laterality",
        "Has interprets",
        "Has indirect morphology",
        "Has indirect device",
        "Has specimen",
        "Has interpretation",
        "Has intent",
        "Has focus",
        "Has definitional manifestation",
        "Has active ingredient",
        "Has finding site",
        "Has episodicity",
        "Has direct substance",
        "Has direct morphology",
        "Has direct device",
        "Has causative agent",
        "Has associated morphology",
        "Has associated finding",
        "Has measurement method",
        "Has precise active ingredient",
        "Precise active ingredient of",
        "Has scale type",
        "Has property",
        "Concentration strength numerator unit of",
        "Is modification of",
        "Has modification of",
        "Has unit",
        "Unit of",
        "Has method",
        "Method of",
        "Has time aspect",
        "Time aspect of",
        "Has component",
        "Has end date",
        "End date of",
        "Has start date",
        "Start date of",
        "Has system",
        "System of",
        "Process duration",
        "Process duration of",
        "Has precoordinated (Question-Answer/Variable-Value) pair",
        "Precoordinated (Question-Answer/Variable-Value) pair of",
        "Has Category",
        "Category of",
        "Has biosimilar",
        "Biosimilar of",
        "Relative to",
        "Relative to of",
        "Count of active ingredients",
        "Is count of active ingredients in",
        "Has product characteristic",
        "Product characteristic of",
        "Has surface characteristic",
        "Surface characteristic of",
        "Has device intended site",
        "Device intended site of",
        "Has compositional material",
        "Compositional material of",
        "Has filling",
        "Filling material of",
        "Reference to variant",
        "Variant refer to concept",
        "Genomic DNA transcribes to mRNA",
        "mRNA Translates to protein",
        "mRNA is transcribed from genomic DNA",
        "Protein is translated from mRNA",
        "Has coating material",
        "Coating material of",
        "Has absorbability",
        "Absorbability of",
        "Process extends to",
        "Process extends from",
        "Has ingredient qualitative strength",
        "Ingredient qualitative strength of",
        "Has surface texture",
        "Surface texture of",
        "Is sterile",
        "Is sterile of",
        "Has target population",
        "Target population of",
        "Has status",
        "Status of",
        "Process acts on",
        "Affected by process",
        "Before",
        "After",
        "Towards",
        "Subject of",
    ]
    print(
        f"extract_ir base_entity={base_entity}, associated_entities={associated_entities}"
    )
    # Refined prompt with examples
    base_prompt = f"""

    Given the **Base Entity** (primary concept) and **Associated Entities** (secondary concepts), select the most appropriate relationship from the provided options.

        **Instructions:**
        - The relationship should describe how the **Base Entity** relates **to** the **Associated Entity**. The relationship should be unidirectional.
        - Review the Base Entity and each Associated Entity.
        - Use the examples below to guide your selection.
        - Choose the relationship that best fits the direction from Base Entity to Associated Entity.
        - Return only one relationship name as a string.
        - Do not include any explanations or additional text. Do not use external resources.

        **Examples:**
            1. Base Entity: 'heart failure'
            Associated Entity: ['ischemic infarct']
            Selected Relationship: 'Has associated finding'
            2. Base Entity: 'gender'
            Associated Entity: ['male', 'female']
            Selected Relationship: 'Has Category'

            2. Base Entity: 'diabetes mellitus'
            Associated Entity: ['insulin']
            Selected Relationship: 'Has associated finding'

            Now, apply the same logic to the following:
            Base Entity: {base_entity}
            Associated Entities: {associated_entities}
            **Relationship Options:**: {', '.join([rel.lower() for rel in relations])}
            """

    system = "You are a helpful assistant with expertise in the biomedical domain."
    final_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", base_prompt)]
    )
    chain = final_prompt | active_model
    result = chain.invoke(
        {
            "base_entity": base_entity,
            "associated_entities": associated_entities,
            "relations": relations,
        }
    ).content

    print(f"extract_ir result={result.strip()}")
    return result.strip().lower()


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        print(f"Prompt:\n{formatted_prompts}")


# chat_history = []
# from langchain_core.messages import HumanMessage
def extract_information(query, model_name=LLM_ID, prompt=None):
    if query:
        # global chat_history
        try:
            active_model = LLMManager.get_instance(model=model_name)
            mapping_for_domain, _, _ = load_mapping(MAPPING_FILE, "all")
            if mapping_for_domain is None:
                logger.error("Failed to load mapping for domain", exc_info=True)
                return None
            examples = mapping_for_domain["examples"]
            select_examples = get_relevant_examples(
                query, "extract_information", examples, topk=3, min_score=0.6
            )
            if select_examples is None:
                logger.error("No relevant examples found", exc_info=True)
                select_examples = []
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                examples=select_examples,
                example_prompt=ChatPromptTemplate.from_messages(
                    [("human", "{input}"), ("ai", "{output}")]
                ),
                input_variables=["input"],
            )
            if prompt:
                base_prompt = prompt
            else:
                # base_prompt = f"""
                # **Task Description**:
                # Extract information from the provided medical query, including the base entity, associated entities, categories, and unit of measurement. This information will be used to link the medical query to standard medical terminologies.
                # **Instructions**:
                # 1. **Preprocessing**:
                # - Rewrite the medical query in English, ensuring all terms are expanded to their full forms.
                # - Translate all non-English terms to English.
                # - Identify and expand any acronyms or abbreviations in the medical query.
                # - Before breaking down the query, assess whether it contains more than one clinical concept.

                # 2. **Extraction**:
                # - **Domain**: Determine the most appropriate OHDSI OMOP domain from the following list:
                #     - [Condition, Anatomic Site, Body Structure, Measurement, Procedure, Drug, Device, Unit, Visit, Death, Demographics, Family History, Lifestyle, History of Events].
                # - **Entities**:
                #     - **Base Entity**: The primary concept mentioned in the medical query, representing the key medical or clinical element being measured, observed, or evaluated.
                #     - **Associated Entities**: List associated entities such as time points, anatomical locations, related procedures, or clinical events that provide context to the base entity.
                # - **Unit**: Specify the unit of measurement associated with the base entity, if mentioned.
                # - **Categories**: Provide a list of categories associated with the base entity, if mentioned. Categories are qualifiers that provide outcome context.
                # 3. **Considerations**:
                # - Do not consider categorical values as context; assume they are categorical values.
                # - Do not perform unnecessary expansion of the query when dividing it into base entity and associated entities.
                # - Do not add a unit of measurement if it is not mentioned in the query.
                # - All visits starting from 1 onwards should be considered as follow-up visits.

                # 4. **Examples**:
                # - If examples are provided, use them to guide your extraction.
                # - If no relevant examples are provided, generate new examples to aid the extraction process.
                # **Output Format**:
                # Provide the output in JSON format with the following fields:
                # - `domain`
                # - `base_entity`
                # - `additional_entities`
                # - `categories`
                # - `unit`
                # Do not include any preamble or explanations.
                # ---
                # **Medical Query**: {input}
                # """
                base_prompt = f"""Role: You are a helpful assistant with expertise in data science and the biomedical domain.
                ***Task Description:
                    - Extract information from the provided medical query which includes the base entity, associated entities, categories, and unit of measurement. This information will be used to link the medical query to standard medical terminologies.
                **Perform the following actions in order to identify relevant information:
                    -Rewrite the medical query in english language to ensure all terms are expanded to their full forms. Always translate all non-english terms to english.
                    -Identify if there are any acronyms and abbreviations in given medical query and expand them.
                    -Before breaking down the query, assess whether it contains more than one clinical concept.
                    **Extract Domain: Determine the most appropriate OHDSI OMOP standards from list of following domains: [Condition, Anatomic Site, Body Structure, Measurement, Procedure, Drug, Device, Unit,  Visit,  Death,  Demographics, Family History, Life Style, History of Events].
                    **Extract Entities:
                        - Base Entity: The primary concept mentioned in the medical query. It represents the key medical or clinical element being measured, observed, or evaluated.
                        - Associated Entities: Extract list of associated entities like time points, anatomical locations, related procedures, or clinical events that clarify the base entity's context within the query. Don't mention entities not given in the query.
                    **Extract Unit: Unit of measurement associated if mentioned.
                    **Extract categories:
                       - If mentioned, provide list of categories associated with the base entity. categories values are qualifiers that provide outcome context.
                **Considerations::
                    -Don't consider categorical values as context. Assume they are categorical values.
                    -Don't Perform an unnecessary expansion of the query to divide into base entity and associated entities.
                    -Don't add additional unit of measurement if not mentioned in the query.
                    - All visit started from 1 and onwards should be considered as follow-up visits.
                    - Before breaking down the query, assess whether it contains more than one clinical concept.
                ** Check Examples: If examples are provided, Please use them to guide your extraction. If no examples or relevant examples are provided, generate new examples to aid the extraction process.
                **Desired format: Provide the output in JSON format with the following fields: 'domain', 'base_entity', 'additional_entities', 'categories' and 'unit'. I repeat, provide the output in JSON format with the following fields: 'domain', 'base_entity', 'additional_entities', 'categories' and 'unit'.Do not add any preamble or explanations.
                medical query: {input}
                Output:
                """
            final_prompt = (
                SystemMessagePromptTemplate.from_template(base_prompt)
                + few_shot_prompt
                + HumanMessagePromptTemplate.from_template("{input}")
            )
            chain = final_prompt | active_model

            start_time = time.time()
            result = chain.invoke({"input": query})
            print(f"Time taken for extract information={time.time()-start_time}")
            if not isinstance(result, dict):
                try:
                    result = fixing_parser.parse(result.content)
                    if result is None:
                        return None
                    result = sanitize_keys(result)
                    result = validate_result(result)
                    rel = extract_ir(
                        base_entity=result.get("base_entity", None),
                        associated_entities=result.get("additional_entities", [])
                        or result.get("categories", []),
                        active_model=active_model,
                    )
                    result["rel"] = rel
                    result["full_query"] = query
                    print(f"extract_information result={result}")

                    return QueryDecomposedModel(**result)

                except ValidationError as e:
                    logger.info(f"Validation Error: {e}")
                    result = None
            else:
                result = sanitize_keys(result)
                result = validate_result(result)
                # chat_history.extend([HumanMessage(content=f"query:{query}, output:{result}")])
                rel = extract_ir(
                    result.get("base_entity", None),
                    result.get("additional_entities", []),
                    active_model=active_model,
                )
                result["rel"] = rel
                result["full_query"] = query
                print(f"extract_information result={result}")
                return QueryDecomposedModel(**result)
        except Exception as e:
            logger.info(f"Error in prompt:{e}", exc_info=True)
            return None
    else:
        return None


def validate_result(result: Dict) -> Dict:
    if isinstance(result.get("additional_entities"), str):
        result["additional_entities"] = [result["additional_entities"]]
    if isinstance(result.get("additional_entities"), dict):
        result["additional_entities"] = result["additional_entities"].values()
    if isinstance(result.get("categories"), str):
        result["categories"] = [result["categories"]]
    if isinstance(result.get("categories"), dict):
        result["categories"] = result["categories"].values()
    if isinstance(result.get("unit"), list):
        result["unit"] = result["unit"][0]
    return result


def evaluate_final_mapping(variable_object: Dict, llm_id: str = "llama3.1"):
    active_model = LLMManager.get_instance(model=llm_id)
    human_template = f"""
    Task: You are tasked with evaluating the correctness of the mapping codes for a variable in a clinical context. The variable represents a column metadata from healthcare records that may be compound in nature. Your goal is to assess whether the mapping to standard clinical codes (e.g., SNOMED, LOINC, UCUM, RxNorm, ATC etc.) is accurate. Your goal is to assess whether the mapping to standard clinical codes (e.g., SNOMED, LOINC, OMOP) is accurate. Focus solely on evaluating the correctness of the mappings without altering or judging the division of the variable itself. Note that some variables may not have additional context or unit information; take this into account in your evaluation.
    **Instructions: Perform the following steps to evaluate the mapping of variable label and its associated attributes:
        * Domain: Confirm that the Domain aligns with the type of measurement or concept.
        * Assess additional context (if provided): If additional context is present (e.g. visit ), confirm that it is accurately captured by the mapped standard labels.
        * For each provided standard label: Verify that the standard label accurately corresponds to the clinical concept(s) described by the variable. If labels are correct, code and ID is considered correct.
        * Evaluate categories (if present): If the variable includes categorical values, verify that these categories are correctly mapped to standard labels and aligned with the clinical context respectively.
        * Evaluate units (if present): If unit information is provided, ensure that the unit matches the expected unit for this measurement or concept. The absence of unit information may be acceptable for certain variables.
    **Output Format: Return the result as a single string, with the explanation and final classification on separate lines, structured as follows:
    ** Explanation: Provide a clear explanation of whether the mapping is correct, partially correct, or incorrect. Include reasoning based on the analysis of the variable and its mapped codes.
    **Final Classification: Choose one of the following final classifications and return it on a new line:
            - correct
            - partially correct
            - incorrect
    Variable : {variable_object}
    Don't add preamble or additional information. Focus on evaluating the correctness of the mapping codes.
    """
    system = "You are a helpful assistant with expertise in semantic web and biomedical domain."
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human_template)], template_format="mustache"
    )

    chain = prompt | active_model
    result = chain.invoke({"variable_object": variable_object}).content
    return result


def generate_information_triples(query, active_model):
    try:
        print(f"generate_information_triples for query={query}")
        human_template = f"""Task Description:
                - Given the query,extract list of triples.
                - Each triple should consist of the following components: 'subject', 'predicate', and 'object'. The 'subject' should be the main entity, the 'predicate' should be the relationship, and the 'object' should be the associated entity.
            ** Perform the following actions in order to generate RDF triples:
                - Determine the domain of the medical query from given OHDSI OMOP standards domain list [condition, anatomic site, body structure, measurement, procedure, drug, device, unit, visit, death, demographics, family history, life style, adverse event or history of event].
                - Determine the base entity from the medical query.
                - Find additional context that aids in understanding the base entity and infer relationships between them.
                - If unit of measurement is provided, include it in the triple with the appropriate relationship to the base entity.
                - If status values are provided, include them in the triple with the appropriate relationship to the base entity.
            ** Desired Format: Only Return the output in List of dictionaries format with the following fields: 'subject', 'predicate', and 'object'. Don't add any preamble or explanations.
            Input: {input}
            """
        system = "You are a helpful assistant with expertise in semantic web and biomedical domain."
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", human_template)], template_format="mustache"
        )
        chain = prompt | active_model
        chain_results = chain.invoke({"input": query}).content
        # rate_limit_check()
        print(f"triple_results={chain_results}")
        save_triples_to_txt(
            query,
            chain_results,
            "/workspace/mapping_tool/data/output/gissi_llama_triples.txt",
        )
    except Exception as e:
        logger.info(f"Error loading LLM: {e}")


def save_triples_to_txt(query, triples, output_file):
    # check if file exists
    if not os.path.exists(output_file):
        # create file
        with open(output_file, "w") as f:
            f.write("query\tsubject\tpredicate\tobject\n")
    with open(output_file, "a") as f:
        for triple in triples:
            f.write(
                f"{query}\t{triple['subject']}\t{triple['predicate']}\t{triple['object']}\n"
            )


def generate_link_prediction_prompt(query, documents, domain=None, in_context=True):
    if in_context:
        _, _, link_prediction_examples = load_mapping(MAPPING_FILE, "all")
        examples = get_relevant_examples(
            query, "link_prediction", link_prediction_examples, topk=1, min_score=0.6
        )
        human_template = """
        Task: Determine the relationship between a given medical query and candidate terms from standard medical terminologies (e.g., SNOMED, LOINC, MeSH, UCUM, ATC, RxNorm, OMOP Extension).  Your goal is to reassess and potentially adjust the existing rankings by categorizing each candidate term based on its relationship to the medical query within a clinical/medical context.
        **Categorization Criteria:
            * Exact Match: The term has the same meaning and context as the query.
            * Synonym: The term conveys the same concept as the query but may be phrased differently.
            * Highly Relevant: The term is closely related to the query but not an exact match or synonym.
            * Partially Relevant: The term is related to the query but includes significant differences in meaning or scope.
            * Not Relevant: The term is unrelated to the query.
       **Task Requirements:
            * Assess Accuracy: Does the candidate term accurately represent the concept described in the query, considering its clinical context?
            * Identify Exact Matches: Determine if any candidate term is an exact match to the query in meaning and context.
            * Assess Specificity: If the candidate terms are more specific than the query, determine which one most closely aligns with the intended meaning.
            * Evaluate Broad Terms: If the candidate terms are broad, determine which term is still most relevant to the core concept of the query.
        Provide a brief justification for your categorization, focusing on the term's relevance, closeness in meaning, and specificity. Avoid assigning higher scores simply because no perfect match exists.
       **Examples:
            * If provided and relevant, use examples to guide your categorization.
            * If examples are irrelevant or missing, create new relevant examples using the same format.
        **Desired Format: Provide your response as a list of dictionaries, each containing the keys "answer", "relationship", and "explanation". Do not include any additional comments or preamble.
        Candidate Terms: {documents}
        Medical Query: {query}
        """
        system = "You are a helpful assistant with expertise in clinical/medical domain and designed to respond in JSON"
        example_prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", human_template)], template_format="mustache"
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt, examples=examples
        )
        final_prompt = (
            SystemMessagePromptTemplate.from_template(system)
            + few_shot_prompt
            + HumanMessagePromptTemplate.from_template(human_template)
        )
        # logger.info(f"final_prompt={final_prompt}")
        return final_prompt
    else:
        human_template = """
        Task: Determine the relationship between a given medical query and each candidate term. Your goal is to reassess and potentially adjust the existing rankings by categorizing each candidate term based on its relationship to the query.
        **Categorization Criteria:
            * Exact Match: The term has the same meaning and context as the query.
            * Synonym: The term conveys the same concept as the query but may be phrased differently.
            * Highly Relevant: The term is closely related to the query but not an exact match or synonym.
            * Partially Relevant: The term is related to the query but includes significant differences in meaning or scope.
            * Not Relevant: The term is unrelated to the query.
        
        Provide a brief justification for your categorization, focusing on the term's relevance, closeness in meaning, and specificity.Don't classify a term higher than it deserves simply because no perfect match exists.
        **Desired Format: Provide your response as a list of dictionaries, each containing the keys "answer", "relationship", and "explanation". Do not include any additional comments or preamble.
        Candidate Terms: {documents}
        Medical Query: {query}
        """
        system = "You are a helpful assistant with expertise in medical domain and designed to output JSON"
        return ChatPromptTemplate.from_messages(
            [("system", system), ("human", human_template)], template_format="mustache"
        )


def generate_ranking_prompt(query, domain=None, in_context=True, documents=None):
    if in_context:
        _, ranking_examples, _ = load_mapping(MAPPING_FILE, domain=domain)
        print(f"{len(ranking_examples)}:ranking examples loaded")
        examples = get_relevant_examples(
            query, "ranking", ranking_examples, topk=1, min_score=0.6
        )
        # logger.info(f"selected_examples for Ranking Prediction={examples}")
        human_template = """Objective: Rank candidate terms from the Standard Medical Terminologies/Vocabularies (SNOMED, LOINC, MeSH, ATC, UCUM, RxNorm, OMOP Extension) based on their relevance and closeness in meaning to a given medical query.
            **Instructions: Reassess and rank a list of candidate terms based on their relevance and closeness in meaning to a given medical query in a clinical context. Update the existing rankings if they are incorrect where:
                *10: The candidate term is an accurate and an exact match/synonym to the input.
                *0: The candidate term is completely irrelevant to the query.
            **Scoring Guidance: Focus on the following aspects to determine the relevance of the candidate terms:
                *Exact Match: Does the term precisely match or act as a synonym for the intended concept in the query? If yes, score closer to 10.
                *Specificity: If the candidate terms are more specific than the query, determine which term adds relevant detail without deviating from the concept. Prioritize relevance to the core meaning of the query.
                *General Relevance: If the candidate terms are broad or generic, identify which term still captures the main idea or essence of the query. Consider how well it fits in a clinical context, without being overly broad or irrelevant.
            **Examples: if provided Follow the examples to understand how to rank candidate terms based on their relevance to the query.
            **Desired format: Your response should be a list of dictionaries, each containing the keys "answer", "score", and "explanation". I repeat, provide the output in list of dictionaries format with the following fields: 'answer', 'score', and 'explanation'.
            Begin your response with the '[' and include no extra comments or information. 
            Candidate Terms: {documents}
            Input: {query}
            Ranked answers:
            """
        system = "You are a helpful assistant with expertise in medical domain and designed to output JSON"
        example_prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", human_template)], template_format="mustache"
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
            # partial_variables={"format_instructions":format_instructions},
        )
        final_prompt = (
            SystemMessagePromptTemplate.from_template(system)
            + few_shot_prompt
            + HumanMessagePromptTemplate.from_template(human_template)
        )

        # logger.info(f"final_prompt={final_prompt}")
        return final_prompt
    else:
        human_template = """Objective: Rank candidate terms based on their relevance and closeness in meaning to a given query.
            Instructions: For each given candidate term, please evaluate its relevance and closeness in contextual meaning to the given query on a scale from 0 to 10 where,
                -10 indicates that system answer is an accurate and an exact match(synonym) to the input.
                -0: The term is completely irrelevant to the query.
            Provide a brief justification for each score, explaining why the assigned score was chosen. Avoid assigning higher scores simply because no perfect match exists.
            Your response should strictly adhere to a valid JSON schema as a list of dictionaries, each containing the keys "answer", "score", and "explanation". 
            Begin your response with the word '[' and include no extra comments or information.
            Input: {query}
            Candidate Terms: {documents}
            Ranked answers:
            """
        system = "You are a helpful assistant expert in medical domain and designed to output JSON"

        template_ = ChatPromptTemplate.from_messages(
            [("system", system), ("human", human_template)], template_format="mustache"
        )
        #  print(f"template={template_.format_messages()}")
        return template_


# def adjust_percentile(scores, base_percentile=75):
#     score_skewness = skew(scores)
#     if score_skewness > 1:  # highly skewed distribution
#         adjusted_percentile = base_percentile - 3  # lower the percentile slightly
#     elif score_skewness < -1:  # highly negatively skewed
#         adjusted_percentile = base_percentile + 3  # increase the percentile slightly
#     else:
#         adjusted_percentile = base_percentile
#     return np.percentile(scores, adjusted_percentile)

# def calculate_dynamic_threshold(scores, base_threshold, exact_match_found):
#     if not scores:
#         return 0.0

#     max_score = max(scores) if scores else 1
#     if max_score == 0:
#         # Handle the case where all scores are zero
#         # Possibly return a zero threshold or a decision that no candidates are valid
#         # logger.info("All scores are zero, returning zero threshold")
#         return 0.0
#     normalized_scores = [score / max_score for score in scores]
#     # Use a higher base threshold if an exact match is found
#     if exact_match_found:
#         base_threshold = max(base_threshold, 8)  # Example value, adjust as needed
#     belief_threshold = adjust_percentile(normalized_scores)
#     return max(belief_threshold, base_threshold / max_score)  # Adjust base_threshold similarly


def calculate_belief_scores(ranking_scores, base_threshold, exact_match_found):
    belief_scores = defaultdict(list)
    # logger.info(f"Ranking Scores")
    scores = [int(res.get("score", 0)) for res in ranking_scores]
    # logger.info(f"Ranking Score={ranking_scores}")
    if not scores:
        return None
    max_score = max(scores)
    if max_score == 0:
        print("all zeros")
        return None  # All scores are zero, indicating no suitable matches, return None
    for res in ranking_scores:
        score = int(res.get("score", 0))
        answer = res["answer"]
        belief_scores[answer].append(score)

    # Calculate average score for each document and determine belief score
    final_belief_scores = {}
    for answer, score_list in belief_scores.items():
        avg_score = sum(score_list) / len(score_list)
        normalized_score = avg_score / max(scores)  # Normalize the score
        final_belief_scores[answer] = (
            normalized_score if normalized_score >= base_threshold else 0
        )

    # logger.info(f"Belief Scores={final_belief_scores}")
    return final_belief_scores


def create_overlapping_segments(documents, overlap=2):
    segments = []
    # Calculate the number of segments dynamically
    if len(documents) > 5:
        num_segments = max(2, len(documents) // 10)  # Adjust this formula as needed
        step_size = len(documents) // num_segments
        for i in range(0, len(documents) - step_size + 1, step_size - overlap):
            segments.append(documents[i : i + step_size])
        # Ensure the last segment reaches the end of the documents
        if segments and segments[-1][-1] != documents[-1]:
            segments.append(documents[-step_size:])
    else:
        segments = [documents]  # No need for splitting if there are fewer documents
    return segments



def get_llm_results_with_overlap(
    prompt, query, documents, max_retries=2, llm=None, llm_name="llama3.1"
):
    if len(documents) >= 5:
        overlapping_segments = create_overlapping_segments(documents, overlap=2)
    else:
        overlapping_segments = [documents]

    start_times = time.time()

    def process_half(doc_half, half_name):
        attempt = 0
        while attempt <= max_retries:
            logger.info(f"Attempt {attempt} to invoke {llm_name} ")
            try:
                chain = prompt | llm

                # config={'callbacks': [ConsoleCallbackHandler()]}) for verbose
                results = chain.invoke({"query": query, "documents": documents})
                results = results.content
                # print(f"llm results={results}")
                if results is None:
                    logger.info("Received None result, retrying...")
                    attempt += 1
                    continue
                elif isinstance(results, list) and all(
                    isinstance(item, dict) for item in results
                ):
                    return results
                elif isinstance(results, str):
                    fixed_results = fix_json_quotes(results)
                    if isinstance(fixed_results, list) and all(
                        isinstance(item, dict) for item in fixed_results
                    ):
                        return fixed_results
                    else:
                        try:
                            results = fixing_parser.parse(results)
                            if isinstance(results, list) and all(
                                isinstance(item, dict) for item in results
                            ):
                                return results
                            else:
                                logger.info(("failed to parse results"))
                                attempt += 1
                        except Exception as e:  # Broad exception handling for any error from fixing_parser
                            logger.info(f"Error in fixing_parser: {e}", exc_info=True)
                            # time.sleep(0.00005)
                            attempt += 1
                            continue  # Retry if fixing_parser parsing fails

                # logger.info(f"Results \n{results} are not in the expected format after attempts to parse, retrying...")
                else:
                    logger.info(f"error in result {type(results)}")
                    attempt += 1

            except ValidationError as e:
                logger.info(f"Validation Error: {e}")
                attempt += 1
                continue  # Retry on validation errors

            except Exception as e:
                logger.info(f"LLM Unexpected Error: {e}")
                attempt += 1
                if attempt > max_retries:
                    logger.info(
                        "Max retries reached without a valid response, returning None"
                    )
                    return None

    results = []
    for index, segment in enumerate(overlapping_segments):
        result = process_half(segment, f"segment_{index}")
        if result:
            results.extend(result)

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_times} seconds")
    return results


def get_llm_results(
    prompt, query, documents, max_retries=2, llm=None, llm_name="llama3.1"
):
    

    def process_half(doc_half,max_retries=2):
        attempt = 0
        while attempt <= max_retries:
            logger.info(f"Attempt {attempt} to invoke {llm_name} ")
            try:
                chain = prompt | llm

                # config={'callbacks': [ConsoleCallbackHandler()]}) for verbose
                results = chain.invoke({"query": query, "documents": documents})
                results = results.content
                # print(f"llm results={results}")
                if results is None:
                    logger.info("Received None result, retrying...")
                    attempt += 1
                    continue
                elif isinstance(results, list) and all(
                    isinstance(item, dict) for item in results
                ):
                    return results
                elif isinstance(results, str):
                    fixed_results = fix_json_quotes(results)
                    if isinstance(fixed_results, list) and all(
                        isinstance(item, dict) for item in fixed_results
                    ):
                        return fixed_results
                    else:
                        try:
                            results = fixing_parser.parse(results)
                            if isinstance(results, list) and all(
                                isinstance(item, dict) for item in results
                            ):
                                return results
                            else:
                                logger.info(("failed to parse results"))
                                attempt += 1
                        except Exception as e:  # Broad exception handling for any error from fixing_parser
                            logger.info(f"Error in fixing_parser: {e}", exc_info=True)
                            # time.sleep(0.00005)
                            attempt += 1
                            continue  # Retry if fixing_parser parsing fails

                # logger.info(f"Results \n{results} are not in the expected format after attempts to parse, retrying...")
                else:
                    logger.info(f"error in result {type(results)}")
                    attempt += 1

            except ValidationError as e:
                logger.info(f"Validation Error: {e}")
                attempt += 1
                continue  # Retry on validation errors

            except Exception as e:
                logger.info(f"LLM Unexpected Error: {e}")
                attempt += 1
                if attempt > max_retries:
                    logger.info(
                        "Max retries reached without a valid response, returning None"
                    )
                    return None

    start_times = time.time()
    results = process_half(documents)
    
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_times} seconds")
    return results


def pass_to_chat_llm_chain(
    query,
    top_candidates,
    n_prompts=1,
    threshold=0.8,
    llm_name="llama",
    domain=None,
    prompt_stage: int = 2,
    in_context=True,
):
    # print(f"llm reranking for query={query} and top_candidates={top_candidates}")
    relationship_scores = {
        "synonym": 10,
        "exact match": 10,
        "highly relevant": 8,
        "partially relevant": 6,
        "not relevant": 0,
    }
    # def calculate_final_score(doc,ranking_scores):
    #     try:
    #         normalized_label = create_document_string(doc)
    #         scores = [int(result['score']) for result in ranking_scores if normalize(result['answer']) == normalized_label]
    #         final_score = np.mean(scores) if scores else 0
    #         # logger.info(f"Final Score for {normalized_label}: {final_score}")
    #         return final_score
    #     except Exception as e:
    #         logger.info(f"Error in calculate_final_score: {e}")
    #         return 0
    logger.info(f"pass_to_chat_llm_chain for query={query}")
    try:
        try:
            model = LLMManager.get_instance(llm_name)
        except Exception as e:
            logger.info(f"Error loading LLM: {e}")
        # _, ranking_examples = load_mapping(MAPPING_FILE, None)
        seen = set()
        documents = []
        exact_match_found_classification = False
        exact_match_found_rank = False
        for doc in top_candidates:
            doc_str = create_document_string(doc)
            if doc_str not in seen:
                seen.add(doc_str)
                documents.append(doc_str)
        ranking_scores = []
        link_predictions_results = []

        for _ in range(n_prompts):  # Assume n_prompts is 3
            ranking_prompt = generate_ranking_prompt(
                query=query, domain=domain, in_context=in_context, documents=documents
            )
            ranking_results = get_llm_results(
                prompt=ranking_prompt,
                query=query,
                documents=documents,
                llm=model,
                llm_name=llm_name,
            )
            if ranking_results:
                ranking_scores.extend(ranking_results)
                for result in ranking_results:
                    if isinstance(result, dict) and int(result.get("score", 0)) == 10:
                        exact_match_found_rank = (
                            True if result["answer"] in documents else False
                        )
                        logger.info(
                            f"Exact match found in Ranking: {result['answer']} = {exact_match_found_rank}. Does it exist in original documents={result['answer'] in documents}"
                        )
            link_predictions_results = []
            if prompt_stage == 2:
                link_prediction_prompt = generate_link_prediction_prompt(
                    query, documents, domain=domain, in_context=in_context
                )
                lp_results = get_llm_results(
                    prompt=link_prediction_prompt,
                    query=query,
                    documents=documents,
                    llm=model,
                    llm_name=llm_name,
                )
                if lp_results:
                    for res in lp_results:
                        if isinstance(res, dict):
                            res["score"] = relationship_scores.get(
                                res.get("relationship", "").strip().lower(), 0
                            )
                    link_predictions_results.extend(lp_results)
                    for res in lp_results:
                        if isinstance(res, dict) and (
                            res["relationship"] == "exact match"
                            or res["relationship"] == "synonym"
                        ):
                            exact_match_found_classification = True
                            # if res['answer'] not in documents:
                            logger.info(
                                f"Exact match found in Link Prediction: {res['answer']} = {exact_match_found_classification}. Does it exist in original documents={res['answer'] in documents}"
                            )
                    # print(f"{lp_results}")
        combined_scores = ranking_scores + link_predictions_results
        if isinstance(combined_scores, str):
            print(f"combined_scores={combined_scores}")
        exact_match_found = exact_match_found_rank and exact_match_found_classification
        print(f"exact_match_found={exact_match_found}")
        avg_belief_scores = calculate_belief_scores(
            combined_scores, threshold, exact_match_found=exact_match_found
        )
        if avg_belief_scores is None:
            return [], False
        sorted_belief_scores = sorted(
            avg_belief_scores.items(), key=lambda item: item[1], reverse=True
        )
        sorted_belief_scores = dict(sorted_belief_scores)
        logger.info(f"belief_threshold={threshold}")
        for doc in top_candidates:
            doc_string = create_document_string(doc)
            doc.metadata["belief_score"] = sorted_belief_scores.get(doc_string, 0)
        filtered_candidates = [
            doc
            for doc in top_candidates
            if sorted_belief_scores.get(create_document_string(doc), 0) >= threshold
        ]
        sorted_filtered_candidates = sorted(
            filtered_candidates,
            key=lambda doc: doc.metadata["belief_score"],
            reverse=True,
        )
        print(
            f"filtered_candidates={[doc.metadata['label'] for doc in sorted_filtered_candidates]}"
        )
        return sorted_filtered_candidates, exact_match_found

    except Exception as e:
        logger.info(f"Error in pass_to_chat_llm_chain: {e}")
        return [], False


def get_json_output(input_text: str):
    llm = LLMManager.get_instance("gpt3.5")
    prompt = PromptTemplate(
        template=f"""
                    Convert the given input into a valid JSON format. The input provided is:
                    {input_text}
                    You should return a list of dictionaries where each dictionary includes 'answer' and 'score' keys.
                    Json Output:
                    """,
        input_variables=["input_text"],
    )
    chain = prompt | llm | JsonOutputParser()
    results = chain.invoke({"input_text": input_text})
    # rate_limit_check()
    # logger.info(f"json results={results}")
    return results