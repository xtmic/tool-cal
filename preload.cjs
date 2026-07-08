'use strict';
try {
  const { setGlobalDispatcher, Agent } = require('undici');
  setGlobalDispatcher(new Agent({ allowH2: false }));
} catch (e) {
  // undici not available as require module, try alternative
  try {
    // Fallback: use process-level approach
    const { setGlobalDispatcher } = require('node:undici') || {};
  } catch (_) {}
}
