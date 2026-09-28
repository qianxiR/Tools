"""
utils 模块 - 包含知识库管理和向量化工具
"""
from .embeding import (
    create_vector_store,
    load_documents,
    search_knowledge_base,
    get_retriever,
    KNOWLEDGE_BASE_DIRS,
    USE_RERANK,
    RERANK_TOP_K,
    INITIAL_RETRIEVE_K
)
from .show_sources import show_retrieved_sources, show_retrieved_sources_compact

__all__ = [
    'create_vector_store',
    'load_documents',
    'search_knowledge_base',
    'get_retriever',
    'KNOWLEDGE_BASE_DIRS',
    'USE_RERANK',
    'RERANK_TOP_K',
    'INITIAL_RETRIEVE_K',
    'show_retrieved_sources',
    'show_retrieved_sources_compact'
]

