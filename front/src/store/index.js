import { createStore } from 'vuex';
import { request, wsRequest } from './util';

wsRequest.onmessage = (event) => {
  console.log('web socket onmessage:', event);
  const { store } = window;
  store.commit('setChatSendStatus', 2);
  store.dispatch('sendInputToStarChatWS', { type: 'response', body: JSON.parse(event.data) });
};

const DEFAULT_SYS_PROMPT = '';
const DEFAULT_G_CONF = {
  temperature: 0.2,
  top_k: 50,
  top_p: 0.95,
  max_new_tokens: 1024,
  repetition_penalty: 1.2,
};

const state = {
  chat: {
    list: [],
    status: 0,  // 对话状态，0空闲，1发起请求，2响应中，3响应成功，4响应失败
    form: {
      input: '',
      session_id: '',
      sysprompt: DEFAULT_SYS_PROMPT,
      generation_config: JSON.clone(DEFAULT_G_CONF),
    },
  },
};

const mutations = {
  setChatList(state, data = []) {
    state.chat.list = data;
  },
  setChatSessionID(state, data = '') {
    state.chat.form.session_id = data;
  },
  setChatInput(state, data = '') {
    state.chat.form.input = data;
  },
  clearChatList(state) {
    state.chat.form.session_id = '';
    state.chat.list = [];
  },
  setSysprompt(state, data = '') {
    state.chat.form.sysprompt = data;
  },
  setGenerationConfig(state, data = {}) {
    const { generation_config } = state.chat.form;
    state.chat.form.generation_config = Object.assign({}, generation_config, data);
  },
  resetDefaultConfig(state) {
    state.chat.form.sysprompt = DEFAULT_SYS_PROMPT;
    state.chat.form.generation_config = JSON.clone(DEFAULT_G_CONF);
  },
  setChatSendStatus(state, data = 0) {
    state.chat.status = data;
  },
};

const actions = {
  async sendInputToStarChatHttp({ getters, commit }) {
    if (!getters.chatForm.input.trim()) {
      console.warn('用户未输入文本');
      return false;
    }
    if (getters.chatSendStatus === 1) {
      console.warn('等待服务响应中，当前响应提交:', getters.chatForm);
      return false;
    }
    commit('setChatSendStatus', 1);

    const { data: resp } = await request.post('/chat/http',
      getters.chatForm,
      { headers: { 'starchat-auth': '' } }
    );

    const { success, code, msg, data } = resp;
    if (!success) {
      commit('setChatSendStatus', 4);
      console.error(`${code}: ${msg}`);
      alert(`${code}: ${msg}`);
      return false;
    }

    const { session_id, chat_list } = data;
    commit('setChatSessionID', session_id)
    commit('setChatList', chat_list);
    commit('setChatSendStatus', 3);
    return true;
  },
  async sendInputToStarChatWS({ getters, commit }, data = {}) {
    const { type = 'send', body = getters.chatForm } = data;

    if (type === 'send') {
      if (!body.input.trim()) {
        console.warn('用户未输入文本');
        return false;
      }

      if (getters.chatSendStatus === 1) {
        console.warn('等待服务响应中，当前响应提交:', body);
        return false;
      }
      commit('setChatSendStatus', 1);
      wsRequest.send(JSON.stringify(body));
      return true;
    }

    if (type === 'response') {
      const { success, code, msg, data, isFinish } = body;
      if (!success) {
        commit('setChatSendStatus', 4);
        console.error(`${code}: ${msg}`);
        alert(`${code}: ${msg}`);
        return false;
      }

      const { session_id, chat_list } = data;
      commit('setChatSessionID', session_id)
      commit('setChatList', chat_list);
      isFinish && commit('setChatSendStatus', 3);
      return true;
    }
  }
};

const getters = {
  chatList(state) {
    return state.chat.list;
  },
  chatSendStatus(state) {
    return state.chat.status;
  },
  chatForm(state) {
    return state.chat.form;
  },
  sessionID(state, getters) {
    return getters.chatForm.session_id;
  },
  sysprompt(state, getters) {
    return getters.chatForm.sysprompt;
  },
  generationConfig(state, getters) {
    return getters.chatForm.generation_config;
  },
};

const store = createStore({ state, mutations, actions, getters });
window.store = store;
export default store;