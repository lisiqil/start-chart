import axios from "axios";

{
  const { parse, stringify } = JSON;
  Object.assign(JSON, { clone: json => parse(stringify(json)) })
}

const { protocol, host } = window.location;
const targetHost = `${host.replace(/:\d+$/, '')}:8080`;

export const request = axios.create({
  baseURL: `${protocol}//${targetHost}/`,
  timeout: 120000,
});

const wsApi = `ws://${targetHost}/chat/ws`;
export const wsRequest = new WebSocket(wsApi);
