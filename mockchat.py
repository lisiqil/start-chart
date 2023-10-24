import uuid

code = f'\n\
```javascript\n\
const helloWorld = "hello world";\n\
console.log(helloWorld);\n\
```\n\
'

def get_output_data():
    session_id = uuid.uuid4().hex
    print(session_id)
    session_chatbot = {}
    session_chatbot[session_id] = {
        "session_id": session_id, "count": 0, "chat_list": []}
    session_chatbot[session_id]["count"] += 1
    session_chatbot[session_id]["chat_list"].extend(
        [{"user_data": "你好", "model_data": code}])
    session_chatbot[session_id]["chat_list"].extend(
        [{"user_data": "你好", "model_data": code}])
    session_chatbot[session_id]["chat_list"].extend(
        [{"user_data": "你好", "model_data": code}])
    return session_chatbot[session_id]
