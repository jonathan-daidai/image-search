#!/usr/bin/env python

import os
import sys
from urllib.parse import urlparse
from typing import Optional
from google.cloud import storage
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel, MultiModalEmbeddingResponse


#Get embeddding from images
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


def download_image(image_url):
    import requests

    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code}")


def download_and_generate_embeddding(input_file_name,output_file_name):
    # 创建GCS客户端
    #client = storage.Client()
    client = storage.Client(project="vector-search-416509")
    bucket = client.get_bucket("image2image-search-pic")
    image_dir = "onestop_temp/"
    
    output_file = open(output_file_name, "w")
    with open(input_file_name, "r") as input_file:
        lines = input_file.readlines()

    for line in lines:
        fields = line.strip().split()
        if len(fields) >= 5:
            image_urls = fields[2:5]  # 提取第3、4、5个字段作为图片URL,Shein的文件格式里，每一行包含3个图片的url

            for image_url in image_urls:
                try:
                    # 从image_url下载图片
                    image_data = download_image(image_url)

                    # 从URL去掉目录名，提取文件名
                    url_parsed = urlparse(image_url)
                    file_name = os.path.basename(url_parsed.path)

                    # 上传图片到GCS的onestop_temp目录
                    blob = bucket.blob(image_dir + file_name)
                    blob.upload_from_string(image_data, content_type="image/jpeg")
                    print(f"Successfully uploaded {file_name} to GCS.")


                    print("start to process embedding.")
                    #Generate Embedding by image file
                    origin_image = f"gs://image2image-search-pic/onestop_temp/{file_name}"
                    print({origin_image})
                    index_id = image_url
                    print({index_id})
                    image_embedding_str = get_image_embeddings(
                        project_id="vector-search-416509",
                        location="us-central1",
                        image_path=origin_image
                     )
                    
                    #Combine "id" and embedding data to create 1 record
                    output_line = f'{{"id": "{index_id}", "embedding": {image_embedding_str}}}\n'
                    #print(f"output json: {output_line}")
                    output_file.write(output_line)
                    
                except Exception as e:
                    print(f"Error downloading or uploading {image_url}: {e}")
    output_file.close()



if __name__ == "__main__":
    # 检查命令行参数数量是否正确
    if len(sys.argv) != 3:
        print("Usage: python generate-embedding-from-url-gcs.py input_file_name output_file_name")
        sys.exit(1)
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    download_and_generate_embeddding(input_file_name,output_file_name)
