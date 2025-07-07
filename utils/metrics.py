"""Script for all metrics to be used

Metrics used:
Mean Reciprocal Rank
Recall@K
"""
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
import torch
from transformers import ProcessorMixin, PreTrainedModel
from transformers import BlipProcessor, BlipModel


def iterate_ds(k:int, df:pd.DataFrame, metric:str, processor:ProcessorMixin, model:PreTrainedModel, caption_column:str, client:QdrantClient, collection_name:str, device:str="cuda"):
    """
    Iterates thru entire dataset, takes each row, processes the caption through a model and depending on the metric chosen, passes it to a relevant function.

    Args:
        k: int - Number of items to query from the vector db (and thus the number of items from which we want to find the rank)
        df: pd.DataFrame - A dataframe containing all the rows
        metric: str - The metric to calculate. Accepts "mrr" or "recall@k"
        processor: ProcessorMixin - The processor used from tranformers
        model: PreTrainedModel - The model used from transformers
        caption_column: str - The column for which caption is to be extracted and converted to an embedding vector
        client: QdrantClient - Reference to the Qdrant Client so that vector database can be accessed
        collection_name: str - Name of the Qdrant collection name so that embeddings can be retrieved.
        device: str - The device name needed for PyTorch to move the model/data to the device if possible. Either "cuda" or "cpu"
    Return:
        A float corresponding to the average of a metric
    """
    total_metric = 0
    for idx, row in df.iterrows():
        try:
            # generate text embeddings
            caption = row[caption_column]
            inputs = processor(text=caption, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
            caption_embeddings = (features / features.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().squeeze()
            
            # choose a metric
            if metric == "mrr":
                total_metric += mean_reciprocal_rank(k, caption_embeddings, idx, client, collection_name)
            elif metric == "recall@k":
                total_metric += recall_at_k(k, caption_embeddings, idx, client, collection_name)
        except Exception as e:
            print(f"Exception occurred: {e}")

    return total_metric / len(df)

def mean_reciprocal_rank(k:int, caption_embeddings: np.ndarray, id_to_search: int, client:QdrantClient, collection_name:str):
    """
    This measures the quality of querying by taking the rank of the relevant item queried into the calculation.
    Formula: 1/rank of first relevant item

    In our case, there is only 1 relevant item

    Finds mrr for one row

    Args:
        k: int - Number of items to query from the vector db (and thus the number of items from which we want to find the rank)
        caption_embeddings: np.ndarray - The embeddings of a caption when passed through a model
        id_to_search: int - An integer identifier used to match an image with a caption when querying
        client: QdrantClient - Reference to the Qdrant Client so that vector database can be accessed
        collection_name: str - Name of the Qdrant collection name so that embeddings can be retrieved.
    
    Returns:
        A float value corresponding the the mean reciprocal rank of one item

    """
    search_result = client.query_points(
        collection_name=collection_name,
        query=caption_embeddings,
        with_payload=False,
        limit=k
    ).points

    # obtain rank of item 
    rank_of_searched_item = 0
    for idx, point in enumerate(search_result):
        if point.id == id_to_search:
            rank_of_searched_item = idx+1

    # ensure division by 0 does not occur
    if rank_of_searched_item == 0:
        return 0
    else:
        return 1/rank_of_searched_item

def recall_at_k(k:int, caption_embeddings: np.ndarray, id_to_search: int, client:QdrantClient, collection_name:str):
    """
    This defines recall at k. This value will either be 0 or 1 as this particular dataset only has 1 relevant item. Therefore, the implementation will be different from the actual recall_at_k to save computation costs

    Args:
        k: int - Number of items to query from the vector db (and thus the number of items from which we want to find the rank)
        caption_embeddings: np.ndarray - The embeddings of a caption when passed through a model
        id_to_search: int - An integer identifier used to match an image with a caption when querying
        client: QdrantClient - Reference to the Qdrant Client so that vector database can be accessed
        collection_name: str - Name of the Qdrant collection name so that embeddings can be retrieved.
    
    Returns:
        A float value corresponding the the recall@k of one item 

    """

    search_result = client.query_points(
        collection_name=collection_name,
        query=caption_embeddings,
        with_payload=False,
        limit=k
    ).points

    id_list = [item.id for item in search_result]

    if id_to_search in id_list:
        return 1
    else:
        return 0

if __name__ == "__main__":

    client = QdrantClient(url="http://localhost:6333")

    df = pd.read_csv("../short_generated_captions.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    avg_mrr = iterate_ds(3, df[:10], "mrr", processor, model, "paraphrased_caption", client, "normal_original_captions", device)

    print(avg_mrr)