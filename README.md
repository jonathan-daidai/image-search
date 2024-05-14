1.Backend：

generate-embedding-from-url-gcs.py
Parse customer image url file
Download file and save to GCS
Generate embedding from images in GCS
Batch_get_image_search_top10_from_file.py
Get url from input image file
Generate embedding by input image file
Search top10 result form input embedding
Write search result to output file

2.Frontend：
app.py: Main program
Templates: index.html
Image_search_function.py: helper function of image search
