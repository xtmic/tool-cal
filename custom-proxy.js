import { createProxyServer } from '/home/mikhail/.local/share/llm-tool-proxy/dist/proxy.js';

const upstream = process.env.UPSTREAM_BASE_URL || 'https://api.titangpt.xyz/v1';
const apiKey = process.env.UPSTREAM_API_KEY || '';
const port = Number(process.env.PORT) || 8787;
const host = process.env.HOST || '127.0.0.1';

const RU_TEMPLATE = ({ renderedTools, toolCallTag }) => {
  return [
    'ИНСТРУМЕНТЫ:',
    renderedTools,
    '',
    'Вызов — КАЖДЫЙ тег на ОТДЕЛЬНОЙ строке:',
    '<имя_инструмента>',
    'параметры',
    '</имя_инструмента>',
    '',
    'Пример:',
    '<bash>',
    'lsblk',
    '</bash>',
    '',
    'После </имя> ОСТАНОВИСЬ.',
  ].join('\n');
};

const server = createProxyServer({
  upstreamBaseURL: upstream,
  upstreamApiKey: apiKey,
  includeExamples: false,
  template: RU_TEMPLATE,
  xmlToolCalls: true,
});

server.listen(port, host, () => {
  console.log(`llm-tool-proxy listening on http://${host}:${port}/v1`);
  console.log(`  -> upstream: ${upstream}`);
});

const shutdown = () => server.close(() => process.exit(0));
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
