from fastapi import FastAPI, Request, Header, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chat import Session
from os.path import abspath, dirname
import config
import cvm_auth
import chat


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
# app = FastAPI(dependencies=[Depends(verify_token)])
app.mount("/static", StaticFiles(directory="front/dist"), name="static")
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

# 自定义错误处理


@app.exception_handler(ServerErrorResponse)
async def server_error_json_handler(req: Request, exc: ServerErrorResponse):
    return JSONResponse(status_code=200, content=exc.getErrorContent())


class Chat_Input(BaseModel):
    session_id: str
    input: str
    sysprompt: str = ""
    generationConfig: dict = {}


@app.post("/chat/http")
async def chat_http(request_body: Chat_Input):
    session = await chat.generate_stream(
        session_id=request_body.session_id, userinput=request_body.input, sysprompt=request_body.sysprompt, generation_config=request_body.generationConfig)
    output_data_json = jsonable_encoder(session)
    return JSONResponse(status_code=200, content={"success": True, "code": 0, "data": output_data_json})

@app.websocket("/chat/ws")
async def chat_ws(ws: WebSocket):
    async def send_json_step(session: Session, is_finish: int = 0):
        output_data_json = jsonable_encoder(session)
        await ws.send_json({"success": True, "code": 0, "isFinish": is_finish, "data": output_data_json})

    await ws.accept()

    try:
        while True:
            request_dict = await ws.receive_json()
            request_body = Chat_Input(**request_dict)
            await chat.generate_stream(session_id=request_body.session_id, userinput=request_body.input,
                                    sysprompt=request_body.sysprompt, generation_config=request_body.generationConfig, step_cb=send_json_step)
    except WebSocketDisconnect:
        print("connection close")

