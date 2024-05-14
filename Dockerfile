# 使用官方 Python 运行时作为父镜像
FROM python:3.11-slim

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录内容复制到位于 /app 中的容器中
COPY . /app

# 安装 requirements.txt 中指定的任何所需包
RUN pip install --no-cache-dir -r requirements.txt

# 使端口 8080 可用于世界
EXPOSE 8080

# 定义环境变量
ENV NAME World

# 在容器启动时运行 app.py
CMD ["python", "/app/app.py"]