from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

client.delete_collection(collection_name="image_db")

# initialise collection
client.create_collection(
    collection_name="image_db",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)