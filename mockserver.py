from fastapi import FastAPI, Request, Header, Depends, WebSocket
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from os.path import dirname, abspath
import config
import cvm_auth
import mockchat


def absdirname() -> str:
    return dirname(abspath(__file__))


CONFIG = config.load_json_config(f'{absdirname()}/config.json')
AUTH_CONFIG = CONFIG["auth"]
SECRET_KEY = AUTH_CONFIG["s_key"]
EXPIRE_TIME = AUTH_CONFIG["expire"]
ERROR_DICT = CONFIG["error_code"]


class ServerErrorResponse(Exception):
    def __init__(self, code: int, msg: str = None):
        self.code = code
        self.msg = msg

    def getErrorContent(self):
        msg = self.msg or ERROR_DICT[self.code]
        return {"success": False, "code": self.code, "msg": msg}


def verify_token(starchat_auth=Header("")):
    code = cvm_auth.authenticate(
        starchat_auth, auth_s_key=SECRET_KEY, expire=EXPIRE_TIME)
    if code != "0":
        raise ServerErrorResponse(code)


app = FastAPI()
app.mount("/static", StaticFiles(directory="front/dist"), name="static")
origins = [
    "http://localhost:5173",
    "http://webcdn.m.qq.com",
    "https://webcdn.m.qq.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://webcdn.m.qq.com",
        "https://webcdn.m.qq.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ServerErrorResponse)
async def server_error_json_handler(req: Request, exc: ServerErrorResponse):
    return JSONResponse(status_code=200, content=exc.getErrorContent())

class Chat_Input(BaseModel):
    session_id: str
    input: str
    sysprompt: str = ""
    generationConfig: dict = {}

@app.post("/input")
def test():
    sdata = mockchat.get_output_data()
    data = jsonable_encoder(sdata)
    return JSONResponse(status_code=200, content={"success": True, "code": 0, "data": data})

@app.websocket("/chat/ws")
async def testws(ws: WebSocket):
    await ws.accept()
    while True:
      data = await ws.receive_json()
      chat_input = Chat_Input(**data)
      print(chat_input.input)
      await ws.send_json({"success": True, "code": 0, "data": data})