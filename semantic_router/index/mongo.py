import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import Field
from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import Metric, SparseEmbedding, Utterance
from semantic_router.utils.logger import logger

class MongoIndex(BaseIndex):
    """
    MongoDB Index for Semantic Router.
    Uses MongoDB Atlas Vector Search.
    """
    
    index_name: str = Field(
        default="semantic_router_index",
        description="Name of the MongoDB collection."
    )
    host: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection string."
    )
    db_name: str = Field(
        default="semantic_router",
        description="MongoDB database name."
    )
    dimensions: Union[int, None] = Field(
        default=None,
        description="Embedding dimensions."
    )
    metric: Metric = Field(
        default=Metric.COSINE,
        description="Distance metric to use for similarity search.",
    )
    client: Any = Field(default=None, exclude=True)
    collection: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "mongo"
        self._initialize_client()
        
    def _initialize_client(self):
        try:
            from pymongo import MongoClient
            self.client = MongoClient(self.host)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.index_name]
        except ImportError as e:
            raise ImportError(
                "Please install 'pymongo' to use MongoIndex. "
                "You can install it with: `pip install pymongo`"
            ) from e

    def _get_vector_index_name(self) -> str:
        return "vector_index"

    def _init_index(self, force_create: bool = False) -> None:
        """Initialize the search index if it doesn't exist."""
        # Check if collection exists, if not create it implicitly by inserting or just pass
        # MongoDB creates collection on first write.
        
        # Check for search index
        # This requires pymongo 4.7+ and Atlas
        try:
            indexes = list(self.collection.list_search_indexes())
            index_name = self._get_vector_index_name()
            if any(idx.get("name") == index_name for idx in indexes):
                return
            
            if not self.dimensions:
                 if force_create:
                     raise ValueError("Dimensions must be provided to create a new index.")
                 return

            # Create search index
            # This is a basic definition for Atlas Vector Search
            definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": self.dimensions,
                        "similarity": self._convert_metric(self.metric)
                    },
                    {
                        "type": "filter",
                        "path": "sr_route"
                    }
                ]
            }
            
            self.collection.create_search_index(
                model={"definition": definition, "name": index_name}
            )
            logger.info(f"Created Atlas Search index '{index_name}'. It may take time to become active.")
            
        except Exception as e:
            logger.warning(f"Could not check or create Atlas Search index: {e}. Ensure you are connected to MongoDB Atlas and have appropriate permissions.")

    def _convert_metric(self, metric: Metric) -> str:
        mapping = {
            Metric.COSINE: "cosine",
            Metric.EUCLIDEAN: "euclidean",
            Metric.DOTPRODUCT: "dotProduct",
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported MongoDB similarity metric: {metric}")
        return mapping[metric]

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = 100,
        **kwargs,
    ):
        self.dimensions = self.dimensions or len(embeddings[0])
        # Try to init index (create search index if needed)
        self._init_index()

        if not metadata_list or len(metadata_list) != len(utterances):
            metadata_list = [{} for _ in utterances]
            
        function_schemas = function_schemas or [{} for _ in utterances]

        documents = []
        for i, (emb, route, utterance, meta, func) in enumerate(zip(embeddings, routes, utterances, metadata_list, function_schemas)):
            doc = {
                "sr_route": route,
                "sr_utterance": utterance,
                "sr_function_schema": func,
                "embedding": emb,
                "metadata": meta
            }
            documents.append(doc)
            
            if len(documents) >= batch_size:
                self.collection.insert_many(documents)
                documents = []
                
        if documents:
            self.collection.insert_many(documents)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        
        pipeline = []
        
        # $vectorSearch stage
        search_stage = {
            "$vectorSearch": {
                "index": self._get_vector_index_name(),
                "path": "embedding",
                "queryVector": vector.tolist(),
                "numCandidates": top_k * 10, # heuristics
                "limit": top_k
            }
        }
        
        if route_filter:
            search_stage["$vectorSearch"]["filter"] = {
                "sr_route": {"$in": route_filter}
            }
            
        pipeline.append(search_stage)
        
        # Project fields
        pipeline.append({
            "$project": {
                "sr_route": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        })
        
        try:
            results = list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return np.array([]), []

        scores = [r.get("score", 0.0) for r in results]
        routes = [r.get("sr_route", "") for r in results]
        
        return np.array(scores), routes

    def delete(self, route_name: str):
        self.collection.delete_many({"sr_route": route_name})

    def delete_index(self):
        self.collection.drop()

    def describe(self) -> IndexConfig:
        count = self.collection.count_documents({})
        return IndexConfig(
            type=self.type,
            dimensions=self.dimensions or 0,
            vectors=count
        )

    def get_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        cursor = self.collection.find({})
        utterances = []
        for doc in cursor:
            utterances.append(Utterance(
                route=doc.get("sr_route"),
                utterance=doc.get("sr_utterance"),
                function_schemas=doc.get("sr_function_schema") if include_metadata else None,
                metadata=doc.get("metadata", {}) if include_metadata else {}
            ))
        return utterances

    def is_ready(self) -> bool:
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    async def ais_ready(self) -> bool:
        # Since we use sync driver, this is just a wrapper
        return self.is_ready()

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        return self.query(vector, top_k, route_filter, sparse_vector)

    async def adelete(self, route_name: str) -> list[str]:
        self.delete(route_name)
        return []

    def _remove_and_sync(self, routes_to_delete: dict):
        for route, utterances in routes_to_delete.items():
            self.collection.delete_many({
                "sr_route": route,
                "sr_utterance": {"$in": utterances}
            })

    def __len__(self):
        return self.collection.count_documents({})

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = 100,
        **kwargs,
    ):
        self.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            function_schemas=function_schemas,
            metadata_list=metadata_list,
            batch_size=batch_size,
            **kwargs,
        )

    async def aget_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        return self.get_utterances(include_metadata=include_metadata)

    def delete_all(self):
        self.collection.delete_many({})

    async def adelete_index(self):
        self.delete_index()

    async def alen(self):
        return len(self)

    async def _init_async_index(self, force_create: bool = False):
        self._init_index(force_create=force_create)

    def _get_all(self, prefix: Optional[str] = None, include_metadata: bool = False):
        cursor = self.collection.find({})
        ids = []
        metadata = []
        for doc in cursor:
            ids.append(str(doc["_id"]))
            if include_metadata:
                # We construct a record similar to what parse_route_info expects
                record = doc.get("metadata", {}).copy()
                record["sr_route"] = doc.get("sr_route")
                record["sr_utterance"] = doc.get("sr_utterance")
                # Ensure function schema is JSON string for compatibility
                func = doc.get("sr_function_schema")
                if isinstance(func, dict):
                    record["sr_function_schema"] = json.dumps(func)
                else:
                    record["sr_function_schema"] = func or "{}"
                metadata.append(record)
        return ids, metadata

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ):
        return self._get_all(prefix=prefix, include_metadata=include_metadata)


