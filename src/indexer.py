import json
from src.settings import settings
import chromadb
from pathlib import Path
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader

embed_model = HuggingFaceEmbedding(
    model_name=settings.embed_model,
    device=settings.embed_device,
)

def load_knowledge_base() -> list[dict]:
    return json.loads(Path(settings.kb_path).read_text(encoding='utf-8'))

def build_document() -> list[Document]:
    raw = load_knowledge_base()
    return [
        Document(
            text=item['text'],
            metadata={
                'id': item['id'],
                'category': item['category'],
                'upsell': item['upsell']
            },
            doc_id=item['id'],
            excluded_embed_metadata_keys=["upsell", "id"],
            excluded_llm_metadata_keys=["upsell"],
        )
        for item in raw
    ]

def build_index() -> VectorStoreIndex:
    documents = build_document()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    collection = client.get_or_create_collection(settings.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index

def load_index() -> VectorStoreIndex:
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_collection(settings.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index
