from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

# initialise collection
client.create_collection(
    collection_name="vector-database",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)