#!/usr/bin/env python3
"""Test script to verify langchain imports for version 0.3.15"""

import sys
import logging

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('/tmp/langchain_import_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def test_import(module_path, class_name):
    """Test if an import works"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return True, f"✅ SUCCESS: from {module_path} import {class_name}"
    except ImportError as e:
        return False, f"❌ FAILED: from {module_path} import {class_name} - {str(e)}"
    except AttributeError as e:
        return False, f"❌ FAILED: {class_name} not found in {module_path} - {str(e)}"
    except Exception as e:
        return False, f"❌ ERROR: from {module_path} import {class_name} - {str(e)}"

# List of imports to test
imports_to_test = [
    # Core imports (should work)
    ("langchain_core.documents", "Document"),
    ("langchain_core.callbacks", "CallbackManagerForRetrieverRun"),
    ("langchain_core.retrievers", "BaseRetriever"),
    ("langchain_core.prompts", "ChatPromptTemplate"),
    ("langchain_core.output_parsers", "JsonOutputParser"),
    ("langchain_core.embeddings", "Embeddings"),
    
    # Community imports (should work)
    ("langchain_community.document_loaders.base", "BaseLoader"),
    ("langchain_community.vectorstores.faiss", "FAISS"),
    ("langchain_community.embeddings", "FastEmbedEmbeddings"),
    
    # Questionable imports from langchain package
    ("langchain.retrievers", "ContextualCompressionRetriever"),
    ("langchain.retrievers", "MergerRetriever"),
    ("langchain.embeddings.base", "Embeddings"),
    ("langchain.retrievers.document_compressors", "EmbeddingsFilter"),
    ("langchain.output_parsers", "OutputFixingParser"),
    ("langchain.output_parsers.fix", "OutputFixingParser"),
    
    # Alternative imports from langchain_community
    ("langchain_community.retrievers", "ContextualCompressionRetriever"),
    ("langchain_community.retrievers", "MergerRetriever"),
    ("langchain_community.retrievers.document_compressors", "EmbeddingsFilter"),
    ("langchain_community.output_parsers.fix", "OutputFixingParser"),
]

logging.info("=" * 80)
logging.info("Testing LangChain Imports for version 0.3.15")
logging.info("=" * 80)

successful = []
failed = []

for module_path, class_name in imports_to_test:
    success, message = test_import(module_path, class_name)
    logging.info(message)
    if success:
        successful.append((module_path, class_name))
    else:
        failed.append((module_path, class_name, message))

logging.info("\n" + "=" * 80)
logging.info(f"SUMMARY: {len(successful)} successful, {len(failed)} failed")
logging.info("=" * 80)

if successful:
    logging.info("\n✅ WORKING IMPORTS:")
    for module_path, class_name in successful:
        logging.info(f"   from {module_path} import {class_name}")

if failed:
    logging.info("\n❌ FAILED IMPORTS:")
    for module_path, class_name, msg in failed:
        logging.info(f"   from {module_path} import {class_name}")

logging.info("\nTest results saved to /tmp/langchain_import_test.log")
logging.info("To view on server: docker exec cohort-explorer-backend-1 cat /tmp/langchain_import_test.log")

sys.exit(0)
