from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from cls import CLS

# создание объекта классификатора и загрузка текстов
cls = CLS("./Тексты писателей")

# класс с типами данных параметров 
class Item(BaseModel): 
    text: str

# создаем объект приложения
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# настройки для работы запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/render_html")
def index_page(request: Request):
    return templates.TemplateResponse("index.html", context= {"request": request})

@app.post("/test")
async def test_page(request: Request):
    body = await request.json()
#   return templates.TemplateResponse("test.html", context= {"request": request})
    class_name = cls.testPrediction(body["testing_text"])
    print(class_name)
    return {"message": body["testing_text"]}

# функция обработки get запроса + декоратор 
@app.get("/")
def read_root():
    return {"message": '\n'.join(cls.output_text)}

@app.get("/create_train_data")
def create_train_data():
    xTrain, xTrain01, yTrain, xTest, xTest01, yTest = cls.createTrainData(maxWordsCount = 15000, xLen = 5000)
    return {"action": "Train data created!", "xTrain": f"{xTrain.shape}", "xTrain01": f"{xTrain01.shape}"}

@app.get("/create_model")
def create_model():
    model = cls.createModel(maxWordsCount=15000, nneurons=2000, nlayers=1, factiv="relu", dropout_rate=0.2)
    return {"action": "Model created!"}

@app.get("/fit_model")
def fit_model():
    history = cls.fitModel(epochs=1, batch_size=32)
    return {"action": "Model has been fitted!"}