import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from django.conf import settings
from PIL import Image

def face_cut(image): {
    # あとで実装
}

def image_to_test_data(upload_image):
    # アップロードされた画像ファイルをメモリ上でOpenCVのimageに格納
    img = Image.open(upload_image)
    image = np.asarray(img)
    # image = cv2.imread(str(upload_image))
    # 1辺が256の正方形にリサイズ
    image = cv2.resize(image, (256, 256))

    # テストデータに変換
    test_array = [image]
    label_array = [0]
    test = tf.data.Dataset.from_tensor_slices(test_array)
    label = tf.data.Dataset.from_tensor_slices(label_array)
    test_data = tf.data.Dataset.zip((test, label))

    return test_data

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

def predict(data, model, path):
    # 予測
    pred = model(data)
    
    # データ保存
    file_path = './media/images/pred.png'
    plt.figure(figsize=(6, 6))
    plt.imshow(pred[0] * 0.5 + 0.5)
    plt.savefig(file_path)
    
    return file_path[1:]

def pipeline(image):
    # アップロード画像を保存
    file_name_2 = './media/images/image.png'
    im = Image.open(image)
    plt.imsave(file_name_2, im)

    # 設定からmediaファイルのパスを取
    media_file_path = settings.MEDIA_URL

    # 設定からモデルファイルのパスを取得
    # model_file_path = settings.MODEL_FILE_PATH
    model_file_path = './daishin2kanna'

    # モデルを読み込む
    model = tf.saved_model.load(model_file_path)
    
    # アップロード画像をnparrayに変換
    test_data = image_to_test_data(image)

    # 前処理
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BUFFER_SIZE = 1000
    test_data = test_data.map( 
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)

    # 予測
    for data in test_data:
        file_name = predict(data, model, media_file_path)

    return file_name, file_name_2[1:]