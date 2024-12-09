import base64
import numpy as np
import zlib
import json
import struct
import cv2
from PIL import Image

# 예제로 사용할 이미지 불러오기
image_path = '/media/eainx/2b2b0b48-d904-4abc-a9ce-60f34c87d9f2/pascal/train/img/2007_000032.jpg'
image = Image.open(image_path)

json_path = "/media/eainx/2b2b0b48-d904-4abc-a9ce-60f34c87d9f2/pascal/train/ann/2007_000032.jpg.json"
with open(json_path) as json_file:
    json_data = json.load(json_file)

# JSON에서 객체 데이터 추출
objects = json_data['objects']
full_image = Image.open(image_path).convert('RGBA')

# 객체 정보 추출 및 처리
for obj in objects:
    # Base64로 인코딩된 PNG 데이터를 디코딩
    encoded_data = obj['bitmap']['data']
    compressed_data = base64.b64decode(encoded_data)

    # zlib을 사용하여 압축 해제
    decompressed_data = zlib.decompress(compressed_data)

    # numpy 배열로 변환
    mask_array = np.frombuffer(decompressed_data, dtype=np.uint8)

    mask_image = Image.fromarray(mask_array)
    width = struct.unpack('>I', mask_array[16:20])[0]
    height = struct.unpack('>I', mask_array[20:24])[0]

    mask_image = mask_image.resize((width, height))

    # Bitmap origin 정보 가져오기
    origin_x, origin_y = obj['bitmap']['origin']

    # 전체 이미지에 대한 마스크 이미지 생성
    full_mask = Image.new('L', (full_image.width, full_image.height), 0)
    full_mask.paste(mask_image, (origin_x, origin_y))

    # 마스크를 이용해 전체 이미지와 합성
    result_image = Image.composite(full_image, Image.new('RGBA', full_image.size, (0, 0, 0, 0)), full_mask)

    # 결과 이미지 표시
    result_image.show()