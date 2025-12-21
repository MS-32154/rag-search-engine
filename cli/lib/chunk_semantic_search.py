import os
import json

import numpy as np

from lib.semantic_search import SemanticSearch, semantic_chunk
from lib.search_utils import CACHE_DIR, load_movies

CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        all_chunks = []
        metadata = []

        for doc_idx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc

            if doc["description"] == "":
                continue
            else:
                doc_chunks = semantic_chunk(doc["description"], overlap=1)
                total_doc_chunks = len(doc_chunks)
                for chunk_idx, doc_chunk in enumerate(doc_chunks):
                    all_chunks.append(doc_chunk)
                    metadata.append(
                        {
                            "movie_idx": doc_idx,
                            "chunk_idx": chunk_idx,
                            "total_chunks": total_doc_chunks,
                        }
                    )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.embeddings)

        with open(CHUNK_METADATA_PATH, "wb") as f:
            json.dump(
                {"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def embed_chunks():
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
