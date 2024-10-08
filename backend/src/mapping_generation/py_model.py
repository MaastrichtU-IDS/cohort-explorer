from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field, validator

class QueryDecomposedModel(BaseModel):
    id: str = Field(default=None, description="Unique identifier for the query.")
    name: str = Field(default=None, description="Name of the query.")
    full_query: str = Field(default=None, description="The full query string.")
    base_entity: str = Field(default=None, description="The base entity of the query.")
    domain: Optional[str] = Field(default="all", description="The domain of the query, defaults to 'all'.")
    categories: Optional[List[str]] = Field(default=None, description="Categorical values related to the query.")
    unit: Optional[str] = Field(default=None, description="Unit for the entity, if applicable.")
    formula: Optional[str] = Field(default=None, description="Formula associated with the entity, if applicable.")
    additional_entities: Optional[List[str]] = Field(default=None, description="Additional entities if applicable.")
    original_label:str = Field(default=None, description="Original label for the query.")
    rel: Optional[str] = Field(default=None, description="Relationship if available.")

    # rel: Optional[List[Dict]] = Field(default=None, description="Relationship if available.")

    # @validator('rel', pre=True, always=True)
    # def parse_rel(cls, value):
    #     if value and isinstance(value, list):
    #         # each value should be dict
    #         if all(isinstance(i, dict) for i in value):
    #             return value
    #     return None

    # original label is full_query.split('|')[0]
    # first check if full_query is none, if not split and return the first element
    @validator('original_label', pre=True, always=True)
    def parse_original_label(cls, value, values):
        full_query = values.get('full_query', '')
        if full_query:
            return full_query.split('categorical values')[0].strip().lower().replace(',', '')
        return value
    

    @validator('unit', pre=True, always=True)
    def parse_unit(cls, value):
        if isinstance(value, list):
            return [str(value).strip().lower()] if value else None
        return value
    # categories should values only only if 'categories values' is present in full_query
    @validator('categories', pre=True, always=True)
    def parse_categories(cls, value, values):
        if 'categorical values' in values['full_query']:
            if value and isinstance(value, list):
                # convert all values to lowercase and string
                return [str(i).strip().lower() for i in value]
            elif value and isinstance(value, str):
                return [str(value).strip().lower()]
        return None
    
    @validator('name', pre=True, always=True)
    def assign_name(cls, value, values):
        if value is None:  # If name is None, assign full_query value to it
            return values.get('full_query', '')
        return value

    validator('additional_entities', pre=True, always=True)
    def parse_additional_entities(cls, value, values):
        categories = values.get('categories', [])
        if categories:
            if value and isinstance(value, list):
                # Preserve the order while removing categories
                return [str(entity) for entity in value if entity not in categories]
            elif value and isinstance(value, str):
                if value not in categories:
                    return [str(value).strip().lower()]
        else:
            if value and isinstance(value, list):
                return [str(i).lower() for i in value]
            elif value and isinstance(value, str):
                return [str(value).strip().lower()]
        return value
    
    # Validator for base_entity and other string values, ensuring they are cleaned
    @validator('base_entity', 'domain', 'formula','rel', pre=True, always=True)
    def clean_strings(cls, value: Optional[str]):
        if isinstance(value, list):
            return str(value[0]).strip().lower() if value else None
        elif isinstance(value, str):
            return str(value).strip().lower()
        return value


def sanitize_keys(input_dict):
    """
    Removes leading and trailing spaces from all keys in the input dictionary.

    Args:
        input_dict (dict): The dictionary with potentially unsanitized keys.

    Returns:
        dict: A new dictionary with sanitized keys.
    """
    if not isinstance(input_dict, dict):
        return input_dict
    return {key.strip(): value for key, value in input_dict.items()}

class RetrieverResultsModel(BaseModel):
    standard_label: Optional[str] = None
    domain: Optional[str] = None
    standard_code: Optional[str] = None
    standard_omop_id: Optional[str] = None
    vocab: Optional[str] = None
    score: Optional[float] = None  # In case there's a score for relevance

# mapping_result = create_processed_result(llm_query_obj.full_query, main_term, variable_label_matches, domain=domain, 
                                    # values_docs=additional_entities_matches,status_docs=status_docs, unit_docs=unit_docs,
                                    # context=context, status=status, unit=unit, primary_to_secondary_rel=rel)
                                    
                                    
class ProcessedResultsModel(BaseModel):
    variable_name: str
    original_query: str
    base_entity: str
    domain: Optional[str]
    base_entity_matches: Optional[List[RetrieverResultsModel]] = []
    categories: Optional[List[str]] = None
    categories_matches: Optional[Dict[str, List['RetrieverResultsModel']]] = {}  # Accepting dict
    unit: Optional[str] = None
    unit_matches: Optional[Any] = None
    additional_entities: Optional[List[str]] = []
    additional_entities_matches: Optional[Dict[str, List[RetrieverResultsModel]]] = {}  # Accepting dict
    primary_to_secondary_rel: Optional[str] = None
    
    #validator for additional entities
    @validator('additional_entities', pre=True, always=True)
    def parse_additional_entities(cls, value):
        if value and isinstance(value, list):
            return value
        elif value and isinstance(value, str):
            return [value]
        return None
    #validate for unit and domain and base_entity
    @validator('variable_name','unit', 'domain', 'base_entity','primary_to_secondary_rel', pre=True, always=True)
    def clean_strings(cls, value: Optional[str]):
        if value:
            return value.strip().lower()
        return value
    #    f len(base_entity_matches) > 0 and base_entity == str(base_entity_matches[0].standard_label): than remove additional entities and its matches\
        
    @validator('additional_entities', pre=True, always=True)
    def validate_additional_entities(cls, value, values):
        base_entity_matches = values.get('base_entity_matches', [])
        full_query = values.get('full_query', '')
        # Check if any of the base_entity_matches have a standard_label matching full_query
        
        for match in base_entity_matches:
            if match.standard_label.lower() == full_query.lower():
                # If there's an exact match, set additional_entities to None
                return None
        
        return value  # Return the original value if no match is found
    
    @validator('additional_entities_matches', pre=True, always=True)
     #if additional entities is None, then additional_entities_matches should be None
     
    def validate_additional_entities_matches(cls, value, values):
        additional_entities = values.get('additional_entities', [])
        if not additional_entities:
            return None
        return value
    


