from PIL import Image
import numpy as np
import base64
import json
import requests

# 替换为您的图片路径
image_path = '5.jpg'

# 加载图片并转换为灰度
image = Image.open(image_path).convert('L')

# 调整图片大小
image = image.resize((28, 28))

# 将图片转换为 NumPy 数组并缩放像素值
image_array = np.array(image) / 255.0

# 增加颜色通道维度
image_array = np.expand_dims(image_array, axis=-1)

# 转换为列表
image_list = image_array.tolist()

# 将图片数据封装到 JSON 对象中
data = {"image": image_list}

# 发送 POST 请求到 Flask 应用
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# Check the status code of the response
if response.status_code == 200:
    try:
        # Try to decode the response as JSON
        response_data = response.json()
        print(response_data)
        max_index = np.argmax(response_data)
        print(max_index)
    except json.JSONDecodeError as e:
        # If an error occurs, print the exception and the raw response content
        print(f"JSON Decode Error: {e}")
        print(f"Response content: {response.text}")
else:
    # If the status code is not 200, print it with the response content
    print(f"Status Code: {response.status_code}")
    print(f"Response content: {response.text}")
