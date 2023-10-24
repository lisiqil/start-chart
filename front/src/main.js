import { createApp } from 'vue'
import TDesign from 'tdesign-vue-next';
import 'tdesign-vue-next/es/style/index.css';
import { MdEditor } from 'md-editor-v3';
import 'md-editor-v3/lib/style.css';

import App from './App.vue';
import store from './store';

const app = createApp(App);
app.use(store);
app.use(TDesign);
app.use(MdEditor);
app.mount('#app');