# src/lazy_imports.py

_cache = {}

def get_rdflib_imports():
    if 'rdflib' not in _cache:
        from rdflib import Graph, Literal, RDF, RDFS, URIRef, DC
        from rdflib.namespace import XSD
        _cache['rdflib'] = {
            'Graph': Graph,
            'Literal': Literal,
            'RDF': RDF,
            'RDFS': RDFS,
            'URIRef': URIRef,
            'DC': DC,
            'XSD': XSD
        }
    return _cache['rdflib']

def get_sparql_wrapper():
    if 'sparql' not in _cache:
        from SPARQLWrapper import SPARQLWrapper, JSON, POST, TURTLE
        _cache['sparql'] = {
            'SPARQLWrapper': SPARQLWrapper,
            'JSON': JSON,
            'POST': POST,
            'TURTLE': TURTLE
        }
    return _cache['sparql']

def get_pandas():
    if 'pandas' not in _cache:
        import pandas as pd
        _cache['pandas'] = pd
    return _cache['pandas']