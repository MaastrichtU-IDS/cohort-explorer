import os
from threading import Lock

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

from .utils import global_logger as logger


class LLMManager:
    _instances = {}
    _lock = Lock()

    @staticmethod
    def get_instance(model="llama", hugging_face=False):
        try:
            # Create a unique key for each configuration
            key = f"{model}{'_hf' if hugging_face else ''}"
            if key not in LLMManager._instances:
                with LLMManager._lock:
                    # Double check locking pattern
                    if key not in LLMManager._instances:
                        LLMManager._instances[key] = LLMManager._load_llm(model, hugging_face)
            return LLMManager._instances[key]
        except Exception as e:
            logger.info(f"Error loading LLM: {e}")
            return None

    @staticmethod
    def _load_llm(model="llama3", hugging_face=False):
        load_dotenv()
        # open_ai_key = os.getenv("OPENAI_API_KEY")
        my_openai_key = os.getenv("CT_MAPPER_OPENAI_API_KEY")
        # org_id = os.getenv("OPENAI_ORG_ID")
        my_org_id = os.getenv("CT_MAPPER_OPENAI_ORG_ID")
        groq_api = os.getenv("GROQ_API_KEY")
        togather_api = os.getenv("TOGATHER_API_KEY")
        hf_key = os.getenv("HF_API_KEY")
        # mixtral_api = os.getenv("MIXTRAL_API_KEY")
        if hugging_face and "gpt" not in model:
            if model == "llama":
                active_model = HuggingFaceEndpoint(
                    endpoint_url="https://baid4h7mdw0v6bco.us-east-1.aws.endpoints.huggingface.cloud",
                    max_new_tokens=15000,
                    top_k=10,
                    top_p=0.95,
                    typical_p=0.95,
                    temperature=0,
                    repetition_penalty=1.03,
                    huggingfacehub_api_token=hf_key,
                )
            if model == "llama_medical":
                active_model = HuggingFaceEndpoint(
                    endpoint_url="https://baid4h7mdw0v6bco.us-east-1.aws.endpoints.huggingface.cloud",
                    max_new_tokens=15000,
                    top_k=10,
                    top_p=0.95,
                    typical_p=0.95,
                    temperature=0,
                    repetition_penalty=1.03,
                    huggingfacehub_api_token=hf_key,
                )
        else:
            if model == "llama3":
                active_model = ChatGroq(temperature=0, groq_api_key=groq_api, model="llama3-8b-8192", max_retries=3)

                # active_model = ChatOllama(
                #     base_url="http://ollama:11434",  # Ollama server endpoint
                #     model="llama:3.1:8b",
                #     temperature=0,
                # )
            elif model == "llama3.1":
                active_model = ChatGroq(temperature=0,groq_api_key=groq_api, model="llama-3.1-70b-versatile",max_retries=3)
                # active_model = ChatTogether(
                #     temperature=0,
                #     together_api_key=togather_api,
                #     model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                #     max_retries=3,
                #     verbose=True,
                # )  # only 12 request for free tier to switch to groq --- Add Groq API key
                # active_model = ChatOllama(
                #     base_url="http://ollama:11434",  # Ollama server endpoint
                #     model="llama3.2",
                #     temperature=0,
                # )
            elif model == "gpt4":
                active_model = ChatOpenAI(
                    model="gpt-4-turbo",
                    temperature=0,
                    timeout=None,
                    openai_api_key=my_openai_key,
                    organization=my_org_id,
                )
            elif model == "gpt-4o":
                active_model = ChatOpenAI(
                    model="gpt-4o", temperature=0, timeout=None, openai_api_key=my_openai_key, organization=my_org_id
                )
            elif model == "gpt-4o-mini":
                active_model = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    timeout=None,
                    openai_api_key=my_openai_key,
                    organization=my_org_id,
                )
            elif model == "gpt3.5":
                active_model = ChatOpenAI(
                    model="gpt-3.5-turbo-0125",
                    temperature=0,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=my_openai_key,
                    organization=my_org_id,
                )
            elif model == "gemma":
                active_model = ChatGroq(temperature=0, groq_api_key=groq_api, model="gemma-7b-it", max_retries=3)
            elif model == "mixtral":
                active_model = ChatGroq(temperature=0, groq_api_key=groq_api, model="mixtral-8x7b-32768", max_retries=3)
                # active_model = ChatMistralAI(mistral_api_key=mixtral_api,model="open-mixtral-8x7b",temperature=0)
            elif model == "phi3":
                active_model = load_local_llm_instance(model_name="phi3")

        return active_model


def load_local_llm_instance(model_name="phi3"):
    from langchain_community.llms.ollama import Ollama

    local_path = (
        "/workspace/rag_pipeline/models/Phi-3-mini-4k-instruct-onnx"  # replace with your desired local file path
    )
    llm = Ollama(
        model=model_name
    )  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `
    return llm


import threading
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.example_selectors import (
        SemanticSimilarityExampleSelector,
    )
from typing import List, Dict, Optional, Any
import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FastEmbedEmbeddings


class CustomSemanticSimilarityExampleSelector(SemanticSimilarityExampleSelector):
    """Custom Selector to check for existing vector store before creating a new one."""
    
    @classmethod
    def from_examples(
        cls,
        examples: List[Dict[str, str]],
        embeddings: Embeddings,
        vectorstore_cls: type[FAISS],
        k: int = 4,
        input_keys: Optional[List[str]] = None,
        *,
        example_keys: Optional[List[str]] = None,
        vectorstore_kwargs: Optional[Dict] = None,
        selector_path: Optional[str] = None,
        content_key: Optional[str] = None,
        **vectorstore_cls_kwargs: Any,) -> 'CustomSemanticSimilarityExampleSelector':
        
        if selector_path is None:
                selector_path = f'../data/faiss_index_{content_key}'
                
        if os.path.exists(selector_path):
            print(f"Selector path exist: {selector_path}")
                        # Load the existing FAISS index
            vectorstore = vectorstore_cls.load_local(selector_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"Selector path does not exist: {selector_path}")
            string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
            vectorstore = vectorstore_cls.from_texts(
                string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
            )
            vectorstore.save_local(selector_path)
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )
        
class ExampleSelectorManager:
    _lock = threading.Lock()
    _selectors = {}

    
    @staticmethod
    def get_example_selector(context_key: str, examples: List[Dict[str, str]], k=4, score_threshold=0.6, selector_path=None):
        """
        Retrieves or creates a singleton example selector based on a context key.
        
        Args:
            context_key (str): A unique key to identify the selector configuration.
            examples (List[Dict[str, str]]): List of example dictionaries.
            embedding (Embedding): The embedding object to use.
            faiss_index (FAISS): The FAISS index object for vector storage and retrieval.
            k (int): Number of nearest neighbors to consider.
            score_threshold (float): Threshold for considering similarity scores.

        Returns:
            SemanticSimilarityExampleSelector: An initialized example selector.
        """
        
        with ExampleSelectorManager._lock:
            if context_key not in ExampleSelectorManager._selectors:
                try:
                    if selector_path is None:
                        selector_path = f'../data/faiss_index_{context_key}'
                        os.makedirs(os.path.dirname(selector_path), exist_ok=True)  # Create the directory if it doesn't exist
                    # Initialize the embeddings
                    embedding = FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')
                    embedding._model = embedding.model_dump().get("_model")
                    # Initialize the selector using the vector store
                    selector = CustomSemanticSimilarityExampleSelector.from_examples(
                            examples=examples,
                            embeddings=embedding, 
                            vectorstore_cls=FAISS,
                            k=k,
                            vectorstore_kwargs={"fetch_k": 40, "lambda_mult": 0.5},
                            input_keys=['input'] , # Assuming 'input' is the key in your examples dict,
                            selector_path=selector_path,
                    )
                    ExampleSelectorManager._selectors[context_key] = selector
                    logger.info(f"Example selector initialized for context: {context_key}." )
                    
                except Exception as e:
                    logger.error(f"Error initializing example selector for {context_key}: {e}", exc_info=True)
                    raise
            return ExampleSelectorManager._selectors[context_key]