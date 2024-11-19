from .chunker import ABCChunker, ChunkerFactory
from langchain_core.documents.base import Document
from .vectordb import ConnectorFactory
from .embeddings import HFEmbedder
from omegaconf import OmegaConf
from .loader import DocSerializer
from typing import AsyncGenerator, Optional
from .config import load_config
from loguru import logger

# Load Hydra configuration
config = load_config()

class Indexer:
    """This class bridges static files with the vector store database.
    Attributes:
        chunker (ABCChunker): The chunking strategy for splitting documents into smaller pieces.
        vectordb: The vector database connection used to store and retrieve embeddings.
        logger: The logger used for logging actions and errors.
    
    Args:
        config (OmegaConf): The configuration object containing parameters for embedding, 
                             chunking, and vector database connection. Defaults to the 
                             predefined configuration.
        config_overwrite (Optional[dict], optional): A dictionary of configuration overrides. 
                                                     Keys should correspond to fields in the 
                                                     `config` object. Defaults to `None`.
            Example config_overwrite dictionary:
                config_overwrite = {
                    "embedder.model_name": "new-embedding-model",
                    "chunker.chunk_size": 300,
                    "vectordb.host": "qdrant-host",
                    "vectordb.port": 1234,
                    "vectordb.hybrid_mode": False
                }

        logger (logging.Logger, optional): A logger instance for logging events. Defaults to 
                                           the global `loguru.logger`.
        device (Optional[str], optional): The device to be used for embedding computation 
                                          (e.g., "cuda" for GPU). Defaults to `None`.

    Methods:
        __init__: Initializes the Indexer with the provided configuration, logger, and device.
        add_files2vdb: Asynchronously adds documents from a folder path to the vector database.
    """
    def __init__(self, 
                 config: OmegaConf = config, 
                 config_overwrite: Optional[dict] = None,
                 logger = logger, 
                 device=None) -> None:
        
        # Overwrite the configuration if parameters are passed
        if config_overwrite is not None:
            for key, value in config_overwrite.items():
                OmegaConf.update(config, key, value)

        embedder = HFEmbedder(embedder_config=config.embedder, device=device)
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(config, embedder=embedder.get_embeddings())
        self.vectordb = ConnectorFactory.create_vdb(config, logger=logger, embeddings=embedder.get_embeddings())
        self.logger.info("Indexer initialized...")


    async def add_files2vdb(self, path):
        """Add a files to the vector database in async mode"""
        serializer = DocSerializer()
        try:
            doc_generator: AsyncGenerator[Document, None] = serializer.serialize_documents(path, recursive=True)
            await self.vectordb.async_add_documents(
                doc_generator=doc_generator, 
                chunker=self.chunker, 
                document_batch_size=4
            )
            self.logger.info(f"Documents from {path} added.")
        except Exception as e:
            raise Exception(f"An exception as occured: {e}")