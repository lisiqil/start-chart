import time
import hashlib

def authenticate(starchat_auth="", auth_s_key="", expire=600) -> str:
    print(starchat_auth)
    if not starchat_auth or not auth_s_key:
        return "4000"

    auth_params = starchat_auth.split(";")
    if len(auth_params) == 0:
        return "4001"

    # 待校验参数
    code = auth_params[0]
    sign = auth_params[1]
    timestamp = auth_params[2]

    # 检查参数
    if not timestamp.isdigit():
        return "4002"

    check_timestamp = int(time.time())
    pass_time = check_timestamp - int(timestamp)
    if not (pass_time > 0 and pass_time <= expire):
        return "4002"

    check_code = hashlib.md5(
        f'tencentinnercall-{sign}-{auth_s_key}-{timestamp}'.encode('utf-8')).hexdigest()
    if check_code != code:
        return "4003"

    return "0"
