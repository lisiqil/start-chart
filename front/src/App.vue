<template>
  <div class="app-container">
    <t-space direction="vertical" class="chat-list-box" style="width: 100%">
      <div
        class="chat-list"
        v-for="(chat, index) in chatList"
        :key="`chat_${index}`"
        :id="`chat_${index}`"
      >
        <div
          class="chat-item user-data-content"
          v-if="'user_data' in chat && chat.user_data"
        >
          {{ chat.user_data }}
        </div>
        <div
          class="chat-item model-data-content"
          v-if="'model_data' in chat && chat.model_data"
        >
          <md-preview :modelValue="chat.model_data"></md-preview>
        </div>
      </div>
    </t-space>
    <t-space direction="vertical" class="form-box" style="width: 100%">
      <t-collapse :default-value="[1]">
        <t-collapse-panel header="sysprompt">
          <div class="sysprompt-params-item block">
            <t-input
              placeholder="你可以尝试下：你现在是一个javascript开发工程师"
              :value="sysprompt"
              @change="(v) => syspromptChangeHandler(v)"
            ></t-input>
          </div>
        </t-collapse-panel>
        <t-collapse-panel header="对话输入">
          <t-loading class="chat-submit-box" :loading="[1, 2].indexOf(chatSendStatus) !== -1">
            <div class="chat-params-item block">
              <t-textarea
                placeholder="请输入..."
                @change="(v) => chatInputHandler(v)"
              ></t-textarea>
            </div>
            <div class="chat-params-item block submit-btn-box">
              <t-button
                theme="default"
                variant="base"
                type="reset"
                @click="() => clearChatClickHandler()"
                >清空对话</t-button
              >
              <t-button
                theme="default"
                variant="base"
                type="reset"
                @click="() => resetConfigClickHandler()"
                >重置参数</t-button
              >
              <t-button
                theme="primary"
                type="submit"
                @click="() => submitClickHandler()"
                >提交</t-button
              >
            </div>
          </t-loading>
        </t-collapse-panel>
        <t-collapse-panel header="generate config">
          <div class="other-params-item block">
            <div class="lable block">Temperature</div>
            <div class="tips block">更高的值会产生更多样化的输出</div>
            <div class="block">
              <t-slider
                :value="generationConfig.temperature"
                :min="0"
                :max="1"
                :step="0.1"
                :input-number-props="{ theme: 'column', decimalPlaces: 1 }"
                @change="
                  (v) => generationConfigChangeHandler({ temperature: v })
                "
              />
            </div>
          </div>
          <div class="other-params-item block">
            <div class="lable block">Top-k(top tokens)</div>
            <div class="tips block">在top tokens中选取</div>
            <div class="block">
              <t-slider
                :value="generationConfig.top_k"
                :input-number-props="{ theme: 'column' }"
                @change="(v) => generationConfigChangeHandler({ top_k: v })"
              />
            </div>
          </div>
          <div class="other-params-item block">
            <div class="lable block">Top-p</div>
            <div class="tips block">
              值越高，采样的概率越低，该参数生效在top_k后生效。
            </div>
            <div class="block">
              <t-slider
                :value="generationConfig.top_p"
                :min="0"
                :max="1"
                :step="0.05"
                :input-number-props="{ theme: 'column', decimalPlaces: 2 }"
                @change="(v) => generationConfigChangeHandler({ top_p: v })"
              />
            </div>
          </div>
          <div class="other-params-item block">
            <div class="lable block">Max new tokens</div>
            <div class="tips block">下次最大token</div>
            <div class="block">
              <t-slider
                :value="generationConfig.max_new_tokens"
                :min="0"
                :max="1024"
                :step="4"
                :input-number-props="{ theme: 'column' }"
                @change="
                  (v) => generationConfigChangeHandler({ max_new_tokens: v })
                "
              />
            </div>
          </div>
          <div class="other-params-item block">
            <div class="lable block">Repetition Penalty</div>
            <div class="tips block">重复惩罚的参数。 1.0 表示没有处罚。</div>
            <div class="block">
              <t-slider
                :value="generationConfig.repetition_penalty"
                :min="0"
                :max="10"
                :step="0.1"
                :input-number-props="{ theme: 'column', decimalPlaces: 1 }"
                @change="
                  (v) =>
                    generationConfigChangeHandler({ repetition_penalty: v })
                "
              />
            </div>
          </div>
        </t-collapse-panel>
      </t-collapse>
    </t-space>
  </div>
</template>
<script>
import { mapGetters, mapMutations, mapActions } from "vuex";
export default {
  computed: {
    ...mapGetters([
      "sessionID",
      "chatSendStatus",
      "chatList",
      "sysprompt",
      "generationConfig",
    ]),
  },
  watch: {
    chatSendStatus(n) {
      if (n === 2) {
        setTimeout(() => {
          const index = this.chatList.length - 1;
          if (!index < 0) return;

          const el = document.getElementById(`chat_${index}`);
          el.scrollIntoView({
            behavior: "smooth",
            inline: "nearest",
          });
        }, 100);
      }
      return n;
    }
  },
  methods: {
    ...mapMutations([
      "setChatInput",
      "clearChatList",
      "setSysprompt",
      "setGenerationConfig",
      "resetDefaultConfig",
    ]),
    ...mapActions(["sendInputToStarChat", "sendInputToStarChatWS"]),
    chatInputHandler(v) {
      this.setChatInput(v);
    },
    syspromptChangeHandler(v) {
      this.setSysprompt(v);
    },
    generationConfigChangeHandler(v) {
      this.setGenerationConfig(v);
    },
    async submitClickHandler() {
      this.sendInputToStarChatWS();
    },
    resetConfigClickHandler() {
      this.resetDefaultConfig();
    },
    clearChatClickHandler() {
      this.clearChatList();
    },
  },
  mounted() {},
};
</script>

<style scoped>
.app-container {
  gap: 16px;
  display: flex;
  flex-direction: column;
  width: 750px;
  margin: auto;
}

/* 对话列表 */
.chat-list-box {
  width: 100%;
  height: calc(100vh - 300px);
  overflow: auto;
  padding: 16px;
  background: rgb(247, 252, 255);
  border: solid 1px #dff7ff;
  box-sizing: border-box;
}
.chat-list-box .chat-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.chat-list-box .chat-list .chat-item {
  position: relative;
  padding: 6px 14px;
  box-sizing: border-box;
  display: flex;
  width: calc(100% - 32px);
  height: auto;
  border-radius: 10px;
}
.chat-list-box .chat-list .chat-item.user-data-content {
  align-self: flex-end;
  border-bottom-right-radius: 0;
  border: solid 1px #aae5ff;
  background-color: #aae5ff;
}

.chat-list-box .chat-list .chat-item.model-data-content {
  align-self: flex-start;
  border-bottom-left-radius: 0;
  border: solid 1px #aae5ff;
  background-color: #fff;
}
.chat-list-box .chat-list .chat-item :deep().md-editor-preview-wrapper {
  padding: 0;
}
.chat-list-box .chat-list .chat-item :deep().default-theme p {
  line-height: 1.4;
  padding: 0;
}
/* 提交box */
.chat-submit-box {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.chat-submit-box .submit-btn-box {
  display: flex;
  flex-direction: row;
  gap: 5px;
  justify-content: flex-end;
}
</style>
