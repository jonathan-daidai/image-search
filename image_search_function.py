from google.cloud import aiplatform_v1
from typing import Optional
import vertexai
import re

from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)

#Define variables 
PROJECT_ID = "level-lyceum-417408"
LOCATION = "asia-southeast1"

API_ENDPOINT = "1210837897.asia-southeast1-889859033984.vdb.vertexai.goog"
INDEX_ENDPOINT = "projects/889859033984/locations/asia-southeast1/indexEndpoints/388365099116527616"
DEPLOYED_INDEX_ID = "image_search_2m_deployment_1711940655732"
NEIGHBOR_COUNT=20



def get_image_embeddings(
    project_id: str,
    location: str,
    image_path: str,
    contextual_text: Optional[str] = None,
    dimension: int = 1408,) -> MultiModalEmbeddingResponse:

    vertexai.init(project=project_id, location=location)

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=dimension,
    )
    return embeddings.image_embedding

#Funtion: Get embeddings by images
def get_text_embeddings(
    project_id: str,
    location: str,
    contextual_text: str,
    dimension: int = 1408,) -> MultiModalEmbeddingResponse:

    vertexai.init(project=project_id, location=location)

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

    embeddings = model.get_embeddings(
        contextual_text=contextual_text,
        dimension=dimension,
    )
    return embeddings.text_embedding

# Configure Vector Search client
client_options = {
  "api_endpoint": API_ENDPOINT
}
vector_search_client = aiplatform_v1.MatchServiceClient(
  client_options=client_options,
)

def get_search_image_urls_and_distances(uploaded_file):



    #Step 2: Generating embedding from the selected files.
    image_vector=get_image_embeddings(
        project_id=PROJECT_ID,
        location=LOCATION, 
        image_path=uploaded_file
    )

    # Step 3: Build FindNeighborsRequest object
    datapoint = aiplatform_v1.IndexDatapoint(
        feature_vector=image_vector
    )

    # Step 4: Construct a search query, the input parameter is the image embedding
    query = aiplatform_v1.FindNeighborsRequest.Query(
      datapoint=datapoint,
      # The number of nearest neighbors to be retrieved
      neighbor_count=NEIGHBOR_COUNT
    )

    # Step 5: Send search request to Vector DB and get the similarity search result
    request = aiplatform_v1.FindNeighborsRequest(
      index_endpoint=INDEX_ENDPOINT,
      deployed_index_id=DEPLOYED_INDEX_ID,
      # Request can have multiple queries
      queries=[query],
      return_full_datapoint=False,
    )
    # Execute the request
    response = vector_search_client.find_neighbors(request)
    results = []  # 用于存储字典的列表
    for neighbor_group in response.nearest_neighbors:
        for neighbor in neighbor_group.neighbors:
            datapoint_id = neighbor.datapoint.datapoint_id
            distance = neighbor.distance
            results.append({"image_url": datapoint_id, "distance": distance})  # 将每个图片URL和它的distance作为一个字典添加到列表中
    
    return results


def get_search_txt2image_urls_and_distances(text_input):
  


    #Step 2: Generating embedding from the selected files.
    text_vector=get_text_embeddings(
        project_id=PROJECT_ID,
        location=LOCATION,
        contextual_text=text_input
    )

    # Step 3: Build FindNeighborsRequest object
    datapoint = aiplatform_v1.IndexDatapoint(
        feature_vector=text_vector
    )

    # Step 4: Construct a search query, the input parameter is the image embedding
    query = aiplatform_v1.FindNeighborsRequest.Query(
      datapoint=datapoint,
      # The number of nearest neighbors to be retrieved
      neighbor_count=NEIGHBOR_COUNT
    )

    # Step 5: Send search request to Vector DB and get the similarity search result
    request = aiplatform_v1.FindNeighborsRequest(
      index_endpoint=INDEX_ENDPOINT,
      deployed_index_id=DEPLOYED_INDEX_ID,
      # Request can have multiple queries
      queries=[query],
      return_full_datapoint=False,
    )
    # Execute the request
    response = vector_search_client.find_neighbors(request)
    results = []  # 用于存储字典的列表
    for neighbor_group in response.nearest_neighbors:
        for neighbor in neighbor_group.neighbors:
            datapoint_id = neighbor.datapoint.datapoint_id
            distance = neighbor.distance
            results.append({"image_url": datapoint_id, "distance": distance})  # 将每个图片URL和它的distance作为一个字典添加到列表中
    
    return results


'''if __name__ == "__main__":
    input_query_text = "red dot skirt"
    response = get_search_txt2image_urls_and_distances(input_query_text)
    
    # 遍历返回的结果列表，而不是尝试访问不存在的nearest_neighbors属性
    for result in response:
        image_url = result["image_url"]
        distance = result["distance"]
        print(f"Image URL: {image_url}, Distance: {distance}")'''
