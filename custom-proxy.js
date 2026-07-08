import { createProxyServer } from './dist/proxy.js';

const upstream = process.env.UPSTREAM_BASE_URL || 'https://api.titangpt.xyz/v1';
const apiKey = process.env.UPSTREAM_API_KEY || '';
const port = Number(process.env.PORT) || 8787;
const host = process.env.HOST || '127.0.0.1';

const EN_TEMPLATE = ({ renderedTools, toolCallTag }) => {
  return [
    '# Tool use',
    'You have access to tools. When a tool helps — call it. Do not guess.',
    '',
    '## How to call',
    'Emit a fenced block tagged `tool_call` with JSON:',
    '',
    '```tool_call',
    '{"name": "<name>", "arguments": { /* ... */ }}',
    '```',
    '',
    '## Rules',
    '- Valid JSON only. "arguments" is a JSON object.',
    '- After a tool-call block: STOP. No extra text.',
    '- NEVER write "Ok", "Let\'s", "Sure", "Done", "Let me". Just call the tool or answer.',
    '- Do not add extra steps the user did not ask for.',
    '- Do not create files not in the user\'s list.',
    '',
    '## What you do NOT have',
    'You do NOT have: search, webfetch, apply_patch, multi_tool_use, or',
    'any other tool not listed above. Do not invent tools. Use only the',
    'tools listed in "Available tools".',
    '',
    '## Available tools',
    renderedTools,
  ].join('\n');
};

const server = createProxyServer({
  upstreamBaseURL: upstream,
  upstreamApiKey: apiKey,
  includeExamples: false,
  template: EN_TEMPLATE,
  xmlToolCalls: false,
});

server.listen(port, host, () => {
  console.log(`llm-tool-proxy listening on http://${host}:${port}/v1`);
  console.log(`  -> upstream: ${upstream}`);
});

const shutdown = () => server.close(() => process.exit(0));
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
