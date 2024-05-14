#app.py
from flask import Flask, render_template, request, url_for, send_from_directory
from image_search_function import get_search_image_urls_and_distances,get_search_txt2image_urls_and_distances
from werkzeug.utils import secure_filename
import os



app = Flask(__name__)
print(f"Current working directory: {os.getcwd()}")
app.config['UPLOAD_FOLDER'] = 'uploads/'
img_path = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global img_path
    global image_url
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_path = filename       
            search_results = get_search_image_urls_and_distances( "uploads/" + img_path)
            
            return render_template('index2.html', img_path=img_path, search_results=search_results)
    return render_template('index2.html', img_path=None, image_urls=[])

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 在app.py的search_text路由中
@app.route('/text', methods=['GET', 'POST'])
def search_text():
    text_input = ''  # 初始化变量来存储文本输入
    search_status = 'none'  # 初始化搜索状态显示为none
    if request.method == 'POST':
        text_input = request.form['text_input']
        search_results = get_search_txt2image_urls_and_distances(text_input)
        search_status = 'block'  # 设置为block以显示搜索状态
        return render_template('index3.html', search_results=search_results, text_input=text_input, search_status=search_status)
    return render_template('index3.html', search_results=[], text_input=text_input, search_status=search_status)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    #app.run(debug=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))