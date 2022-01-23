from fastapi import FastAPI
from fastapi import File, UploadFile
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# https://fastapi.tiangolo.com/tutorial/cors/
origins = [
    "https://murin-an.herokuapp.com/",
    "https://pytorch-cpu.herokuapp.com",
    "https://renewalapp.herokuapp.com/",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
def health():
    return {
        'message': 'ok'
    }

@app.post('/post')
def simple_post(param: str):
    return {
        'message': f'You posted `{param}`!'
    }

#ラベル数
n_class = 6
#モデル名
model_keras = "test.h5"
#画像サイズ
IMG_WIDTH, IMG_HEIGHT = 160, 160
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

class_names = ['建物の中から撮影したお庭', '外で撮影したお庭', 'お菓子やお抹茶', '洋館の中', '建物の外観', 'その他の写真']

@app.post('/api/inference')
async def inference(file: UploadFile = File(...)):
    contents = await file.read()
    from io import BytesIO
    from PIL import Image
    im = Image.open(BytesIO(contents))
    im.save(file.filename)
    from keras.preprocessing import image as preprocessing
    img = preprocessing.load_img(file.filename, target_size=TARGET_SIZE)
    img = preprocessing.img_to_array(img)
    import numpy as np
    x = np.expand_dims(img, axis=0)
    del im
    del contents
    del file
    from tensorflow import keras
    keras.backend.clear_session()
    import gc
    gc.collect()
    from tensorflow.keras.models import load_model
    model = load_model(model_keras)
    predict = model.predict(x)
    for p in predict:
        class_index = p.argmax()
        probablity = p.max()
        class_name = class_names[class_index] 
        return {"result":"OK", "class_index":str(class_index), "probality":str(probablity), "class_name": class_name}
