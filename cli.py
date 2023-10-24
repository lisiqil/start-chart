import chat
from chat import Session

session_id = ""
chat_list = None


def step(session: Session):
    print(session.latestChat().model_data)


async def main():
    global session_id
    global chat_list

    print("欢迎使用 starchat 模型")
    while True:
        userinput = input("用户：")
        session = await chat.generate_stream(session_id=session_id, userinput=userinput, step_cb=step)
        print("代码助手：" + session.latestChat().model_data)

if __name__ == "__main__":
    main()
