import platform
import random
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from threading import Thread

torch.cuda.empty_cache()

os_name = platform.system()
"""
需要在服务器上预先准备好starchat-beta模型
CHECK_POINT: 模型检查点，本地模型路径
DEVICE：驱动方式，m1/2芯片需要比较特殊的环境配置，驱动为”mps“。NVIDIA为cuda。无显卡为cpu
"""
CHECK_POINT = "../starchat-beta"
DEVICE = 'mps' if os_name == 'Darwin' else 'cuda'
if torch.cuda.is_available() != True:
    DEVICE = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT, trust_remote_code=True)
# 当前是量化模式，机器行性能足够的情况下load_in_8bit=True（显存20G以下）可以移除，默认是float 32（显存60G），或者替换为torch_dtype=torch.float16（显存32G）
model = AutoModelForCausalLM.from_pretrained(
    CHECK_POINT, trust_remote_code=True, device_map="auto", load_in_8bit=True)

system_token: str = "<|system|>"
user_token: str = "<|user|>"
assistant_token: str = "<|assistant|>"
end_token: str = "<|end|>"


def get_prompt(system="", messages=None):
    prompt = system_token + "\n" + system + end_token + "\n"
    if messages is None:
        raise ValueError("Dialogue template must have at least one message.")

    for message in messages:
        if message["role"] == "user":
            prompt += user_token + "\n" + message["content"] + end_token + "\n"
        else:
            prompt += assistant_token + "\n" + \
                message["content"] + end_token + "\n"
    prompt += assistant_token + "\n"
    return prompt


def get_generation_config(temperature=0.2, top_k=50, top_p=0.95, max_new_tokens=1024, repetition_penalty=1.2):
    return GenerationConfig(
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(end_token),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        truncate=4096,
        seed=random.randint(0, 1000000),
        stop_sequences=[end_token],
    )


class Chat_Item():
    def __init__(self, user_data: str = "", model_data: str = "") -> None:
        self.user_data = user_data
        self.model_data = model_data

    def update_model_data(self, text: str = ""):
        self.model_data = text

    def update_user_data(self, text: str = ""):
        self.user_data = text


class Session():
    def __init__(self, session_id: str = "", count: int = 0, chat_list: list = []) -> None:
        self.session_id = session_id
        self.count = count
        self.chat_list = []
        self.extend(chat_list)

    def extend(self, list: list = []) -> int:
        for item in list:
            self.chat_list.extend([Chat_Item(**item)])
        self.count = len(self.chat_list)
        return self.count

    def latestChat(self) -> Chat_Item:
        latest = self.count - 1
        return self.chat_list[latest]


session_chatbot = {}


def generate_session(session_id="") -> Session:
    global session_chatbot
    if session_id == "":  # 生成下会话id
        session_id = uuid.uuid4().hex

    if session_id not in session_chatbot:
        initSession = {"session_id": session_id, "count": 0, "chat_list": []}
        session_chatbot[session_id] = Session(**initSession)

    return session_chatbot[session_id]


def generate(session_id="", sysprompt="", userinput=None, generation_config={}) -> Session:
    if userinput is None:
        raise ValueError("用户输入不存在")

    session = generate_session(session_id)

    messages = [{"role": "user", "content": userinput}]
    past_messages = []
    for chat_item in session.chat_list:
        past_messages.extend(
            [{"role": "user", "content": chat_item.user_data}, {
                "role": "assistant", "content": chat_item.model_data.rstrip()}]
        )
    if len(past_messages) > 0:
        messages = past_messages + messages

    # 构建入参
    prompt = get_prompt(system=sysprompt, messages=messages)
    generation_config = get_generation_config(**generation_config)

    inputs = tokenizer.encode(prompt,
                              return_tensors="pt").to(DEVICE)
    outputs = model.generate(inputs, generation_config=generation_config)
    output = tokenizer.decode(outputs[0],
                              skip_special_tokens=False).rsplit(assistant_token, 1)[1].rstrip(end_token)

    session.count += 1
    session.chat_list.extend(
        [{"user_data": userinput, "model_data": output}])

    return session


async def generate_stream(session_id="", sysprompt="", userinput=None, generation_config={}, step_cb=None) -> Session:
    if userinput is None:
        raise ValueError("用户输入不存在")

    session = generate_session(session_id)

    messages = [{"role": "user", "content": userinput}]
    past_messages = []
    for chat_item in session.chat_list:
        past_messages.extend(
            [{"role": "user", "content": chat_item.user_data}, {
                "role": "assistant", "content": chat_item.model_data.rstrip()}]
        )
    if len(past_messages) > 0:
        messages = past_messages + messages

    # 构建入参
    prompt = get_prompt(system=sysprompt, messages=messages)
    generation_config = get_generation_config(**generation_config)

    streamer = TextIteratorStreamer(tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    kwargs = dict(inputs, streamer=streamer,
                  generation_config=generation_config)
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    session.extend([{"user_data": userinput, "model_data": ""}])
    count = -1
    generate_text = ""
    for new_text in streamer:
        generate_text += new_text
        count += 1
        # 计数32次或者计算完结时更新和回调
        is_count_call_step_cb = count % 8 == 0
        is_finish = 1 if generate_text.endswith(end_token) else 0
        if is_finish or is_count_call_step_cb:
            session.latestChat().update_model_data(
                text=generate_text.rsplit(assistant_token, 1)[1].rstrip(end_token).strip())
            if step_cb:
                await step_cb(session, is_finish)

    return session
