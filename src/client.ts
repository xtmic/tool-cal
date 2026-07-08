import type {
  ChatClientLike,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionChunkDelta,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionContentPart,
  ChatCompletionTool,
  ToolCapableClient,
} from "./types.js";
import {
  buildToolPrompt,
  buildToolChoiceInstruction,
  DEFAULT_TOOL_CALL_TAG,
  DEFAULT_TOOL_RESULT_TAG,
  type PromptOptions,
} from "./prompt.js";
import { parseToolCalls, tryParseJson, extractReasoning } from "./parser.js";
import { ToolCallStreamParser, ReasoningStreamParser } from "./stream-parser.js";
import { ToolCapabilityError } from "./errors.js";

const META_TALK_RE =
  /^(Let's\s+(call|do|send|try|actually|follow)|(Ok|Okay)\.?\s*$|Proceed\.?\s*$|No\s+more\.?\s*$|Reading\.+\s*$|→\w+|Stop\.\s*$|tool_call\b|Can't\s+wrap|We'll\s+follow|But\s+tool|Wait\.?\s*$|\?\s*$|т[ыЫ]\s+не\s+|Сейчас\s+\w+|Выполню|Достану|Проверю|Запрошу|Попробую|Сделаю|Посмотрю|Похоже,|%[A-Z]\w+)/i;

const PROMISE_RE =
  /^(Сейчас|Сейчас я|Я сейчас|Let me|I will|I'll|Сейчас быстр\w+)\s+\w+/i;

/**
 * Detects meta-narration ("Let's call tool...", "Ok.", etc.) and returns `null`
 * when the text should be discarded from the conversation.  Applied only when
 * no tool calls were produced, so actual tool invocations are never affected.
 */
function cleanMetaTalk(text: string): string | null {
  if (!text) return text;
  const lines = text.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  if (lines.length === 0) return text;

  // Rule 3 – entire content is one short sentence matching the meta-talk pattern.
  if (text.length < 100 && lines.length === 1 && META_TALK_RE.test(text.trim())) return null;

  // Rule 2 – same line repeated 3+ times consecutively.
  for (let i = 0; i <= lines.length - 3; i++) {
    if (lines[i] === lines[i + 1] && lines[i] === lines[i + 2]) return null;
  }

  // Rule 1 – more than half of the non-empty lines match the meta-talk pattern.
  let matched = 0;
  for (const line of lines) if (META_TALK_RE.test(line)) matched++;
  if (matched > lines.length * 0.35) return null;

  return text;
}

export interface WrapOptions extends PromptOptions {
  /** Generates the `id` for parsed tool calls (e.g. for deterministic tests). */
  generateId?: (index: number) => string;
  /** Accept ```json / untagged look-alike blocks as a fallback. Default `true`. */
  lenientFences?: boolean;
  /**
   * How tool instructions combine with an existing system message:
   * `"merge"` appends to the first system/developer message; `"prepend"` always
   * inserts a fresh system message at the front. Default: `"merge"`.
   */
  systemInjection?: "merge" | "prepend";
  /**
   * Split a model's `<think>…</think>` reasoning out of `content` into the
   * `reasoning_content` field (an upstream `reasoning_content` field is forwarded
   * too). Applies to tool-enabled requests. Default: `true`.
   */
  reasoning?: boolean;
  /** Tag whose contents are treated as reasoning. Default: `think`. */
  reasoningTag?: string;
  /**
   * Also parse native XML-style tool tags — an own-line `<toolName>…</toolName>`
   * block whose tag matches a passed-in tool — into tool calls. Off by default;
   * enable for models (e.g. some R1 variants) that emit calls as XML instead of
   * the ` ```tool_call ` fence. In streaming mode this is best-effort and does
   * not guard against tags inside foreign code fences. Default: `false`.
   */
  xmlToolCalls?: boolean;
}

/** Extracts plain text from string-or-content-part-array message content. */
function textFromContent(
  content: string | ChatCompletionContentPart[] | null | undefined,
): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => p && p.type === "text" && typeof (p as any).text === "string")
      .map((p) => (p as any).text as string)
      .join("\n");
  }
  return "";
}

/**
 * Collapses a content-part array to a plain string **iff every part is text**.
 * Many tool-less upstreams reject OpenAI's array content form (some tokenize it
 * as 0 tokens and reject the request). Arrays with non-text parts (images, etc.)
 * are returned unchanged so multimodal requests keep working.
 */
function collapseTextContent(
  content: string | ChatCompletionContentPart[] | null | undefined,
): string | ChatCompletionContentPart[] | null | undefined {
  if (!Array.isArray(content) || content.length === 0) return content;
  const allText = content.every(
    (p) => p && p.type === "text" && typeof (p as any).text === "string",
  );
  if (!allText) return content;
  return content.map((p) => (p as any).text as string).join("\n");
}

/** Returns the message with text-only array content collapsed to a string. */
function normalizeOutboundMessage(
  msg: ChatCompletionMessageParam,
): ChatCompletionMessageParam {
  if ("content" in msg && Array.isArray(msg.content)) {
    const collapsed = collapseTextContent(msg.content);
    if (collapsed !== msg.content) {
      return { ...msg, content: collapsed } as ChatCompletionMessageParam;
    }
  }
  return msg;
}

function renderToolCallBlock(tag: string, name: string, argsString: string): string {
  const parsedArgs = tryParseJson(argsString);
  const argsValue = parsedArgs === undefined ? argsString : parsedArgs;
  const payload = JSON.stringify({ name, arguments: argsValue });
  return "```" + tag + "\n" + payload + "\n```";
}

function renderToolResultBlock(
  tag: string,
  name: string,
  id: string,
  result: string,
): string {
  const parsedResult = tryParseJson(result);
  const resultValue = parsedResult === undefined ? result : parsedResult;
  const payload = JSON.stringify({ name, tool_call_id: id, result: resultValue });
  return "```" + tag + "\n" + payload + "\n```";
}

/**
 * Rewrites a message history that uses native tool roles (assistant
 * `tool_calls`, `role: "tool"`) into plain text that follows the prompt
 * contract — because the underlying model understands neither. Consecutive
 * tool results are merged into one user message.
 */

function dedupInterTurn(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[] {
  const indices: number[] = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (msg.role === "assistant" && "tool_calls" in msg && msg.tool_calls && msg.tool_calls.length > 0) {
      indices.push(i);
      if (indices.length === 2) break;
    }
  }
  if (indices.length < 2) return messages;

  const laterIdx = indices[0]!;
  const earlierIdx = indices[1]!;
  const earlier = messages[earlierIdx]! as ChatCompletionAssistantMessageParam;
  const later = messages[laterIdx]! as ChatCompletionAssistantMessageParam;

  const earlierKeys = new Set(
    earlier.tool_calls!.map((tc) => tc.function.name + "\0" + tc.function.arguments),
  );

  const deduped = later.tool_calls!.filter(
    (tc) => !earlierKeys.has(tc.function.name + "\0" + tc.function.arguments),
  );

  if (deduped.length === later.tool_calls!.length) return messages;

  const copy = messages.slice();
  copy[laterIdx] = { ...later, tool_calls: deduped.length > 0 ? deduped : undefined } as ChatCompletionMessageParam;
  return copy;
}

export function flattenMessages(
  messages: ChatCompletionMessageParam[],
  opts: { toolCallTag: string; toolResultTag: string },
): ChatCompletionMessageParam[] {
  const out: ChatCompletionMessageParam[] = [];
  const idToName = new Map<string, string>();
  let pendingResults: string[] = [];

  // Pre-pass: map every tool_call id -> name first, so a tool result that
  // appears before its assistant message still renders the real tool name.
  for (const msg of messages) {
    if (msg.role === "assistant" && msg.tool_calls) {
      for (const tc of msg.tool_calls) idToName.set(tc.id, tc.function.name);
    }
  }

  const flushResults = () => {
    if (pendingResults.length > 0) {
      out.push({ role: "user", content: pendingResults.join("\n\n") });
      pendingResults = [];
    }
  };

  for (const msg of messages) {
    if (msg.role === "tool") {
      const name = idToName.get(msg.tool_call_id) ?? "tool";
      pendingResults.push(
        renderToolResultBlock(
          opts.toolResultTag,
          name,
          msg.tool_call_id,
          textFromContent(msg.content),
        ),
      );
      continue;
    }

    if (msg.role === "function") {
      // Deprecated function-role result — render like a tool result.
      pendingResults.push(
        renderToolResultBlock(
          opts.toolResultTag,
          msg.name,
          msg.name,
          textFromContent(msg.content),
        ),
      );
      continue;
    }

    flushResults();

    if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
      const blocks: string[] = [];
      const text = textFromContent(msg.content);
      for (const tc of msg.tool_calls) {
        idToName.set(tc.id, tc.function.name);
        blocks.push(
          renderToolCallBlock(opts.toolCallTag, tc.function.name, tc.function.arguments),
        );
      }
      const content = [text, ...blocks].filter((s) => s.length > 0).join("\n\n");
      out.push({ role: "assistant", content });
      continue;
    }

    out.push(normalizeOutboundMessage(msg));
  }

  flushResults();
  return out;
}

/** Inserts the tool-instruction text as/into a system message. */
function injectSystemPrompt(
  messages: ChatCompletionMessageParam[],
  instruction: string,
  mode: "merge" | "prepend",
): ChatCompletionMessageParam[] {
  if (mode === "prepend") {
    return [{ role: "system", content: instruction }, ...messages];
  }
  const idx = messages.findIndex(
    (m) => m.role === "system" || m.role === "developer",
  );
  if (idx === -1) {
    return [{ role: "system", content: instruction }, ...messages];
  }
  const target = messages[idx]!;
  // Preserve multimodal array content (images etc.) by appending a text part
  // rather than flattening the whole message to a string.
  const newContent = Array.isArray(target.content)
    ? [...target.content, { type: "text", text: instruction } as ChatCompletionContentPart]
    : textFromContent(target.content) + "\n\n" + instruction;
  const merged = { ...target, content: newContent } as ChatCompletionMessageParam;
  const copy = messages.slice();
  copy[idx] = merged;
  return copy;
}

interface TransformOptions {
  toolCallTag: string;
  generateId?: (i: number) => string;
  lenientFences: boolean;
  reasoning: boolean;
  reasoningTag?: string;
  xmlToolCalls: boolean;
  tools?: ChatCompletionTool[];
}

/** Builds a synthetic non-streamed response from the model's raw text. */
function transformResponse(res: ChatCompletion, opts: TransformOptions): ChatCompletion {
  const choices = res.choices.map((choice) => {
    const raw = choice.message?.content;
    if (typeof raw !== "string") return choice;

    let working = raw;
    let reasoning: string | null = null;
    if (opts.reasoning) {
      const r = extractReasoning(working, { reasoningTag: opts.reasoningTag });
      reasoning = r.reasoning;
      working = r.content;
    }

    const { content, toolCalls } = parseToolCalls(working, {
      toolCallTag: opts.toolCallTag,
      generateId: opts.generateId,
      lenientFences: opts.lenientFences,
      xmlToolCalls: opts.xmlToolCalls,
      tools: opts.tools,
    });

    // Forward an upstream-provided reasoning_content field if we didn't extract one.
    const upstreamReasoning = (choice.message as { reasoning_content?: unknown })
      .reasoning_content;
    const finalReasoning =
      reasoning ?? (typeof upstreamReasoning === "string" ? upstreamReasoning : null);

    // Always clean meta-talk from content. When tool calls were found the
    // parsed `content` is the prose that remains after removing tool-call
    // spans — never fall back to `raw` (which would reintroduce the blocks).
    const finalContent = toolCalls.length > 0
      ? (content ? cleanMetaTalk(content) : null)
      : (() => {
          const text = content ?? raw;
          if (text && text.length < 200 && PROMISE_RE.test(text.trim())) return null;
          return cleanMetaTalk(text);
        })();
    const message: ChatCompletionMessage = {
      role: "assistant",
      content: finalContent ?? null,
      refusal: choice.message.refusal ?? null,
    };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    if (finalReasoning != null) message.reasoning_content = finalReasoning;
    return {
      ...choice,
      message,
      finish_reason: toolCalls.length > 0 ? ("tool_calls" as const) : choice.finish_reason,
    };
  });
  return { ...res, choices };
}

function makeChunk(
  template: ChatCompletionChunk | undefined,
  delta: ChatCompletionChunkDelta,
  finishReason: ChatCompletionChunk["choices"][number]["finish_reason"] = null,
  usage?: ChatCompletionChunk["usage"],
): ChatCompletionChunk {
  return {
    id: template?.id ?? "chatcmpl-tool-shim",
    object: "chat.completion.chunk",
    created: template?.created ?? 0,
    model: template?.model ?? "",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...(usage !== undefined ? { usage } : {}),
  };
}

async function* transformStream(
  source: AsyncIterable<ChatCompletionChunk>,
  opts: TransformOptions,
): AsyncIterable<ChatCompletionChunk> {
  const parser = new ToolCallStreamParser({
    toolCallTag: opts.toolCallTag,
    generateId: opts.generateId,
    xmlToolCalls: opts.xmlToolCalls,
    tools: opts.tools,
  });
  const reasoner = opts.reasoning
    ? new ReasoningStreamParser({ reasoningTag: opts.reasoningTag })
    : null;
  let template: ChatCompletionChunk | undefined;
  let started = false;

  // Buffer content deltas so we can filter meta-talk at the end.
  // Tool-call deltas are yielded immediately so the runner can process them.
  const contentChunks: Array<{ chunk: ChatCompletionChunk | undefined; delta: ChatCompletionChunkDelta }> = [];
  let contentAcc = "";

  for await (const chunk of source) {
    template = chunk;
    if (!started) {
      yield makeChunk(chunk, { role: "assistant" });
      started = true;
    }
    const delta = chunk.choices?.[0]?.delta;
    // Forward an upstream-provided separate reasoning_content field untouched.
    const upstreamReasoning = (delta as { reasoning_content?: unknown } | undefined)
      ?.reasoning_content;
    if (typeof upstreamReasoning === "string" && upstreamReasoning.length > 0) {
      yield makeChunk(chunk, { reasoning_content: upstreamReasoning });
    }
    const text = delta?.content;
    if (typeof text === "string" && text.length > 0) {
      if (reasoner) {
        const r = reasoner.push(text);
        if (r.reasoning.length > 0) yield makeChunk(chunk, { reasoning_content: r.reasoning });
        for (const d of parser.push(r.content)) {
          if (d.tool_calls) {
            yield makeChunk(chunk, d);
          } else {
            contentChunks.push({ chunk, delta: d });
            contentAcc += d.content ?? "";
          }
        }
      } else {
        for (const d of parser.push(text)) {
          if (d.tool_calls) {
            yield makeChunk(chunk, d);
          } else {
            contentChunks.push({ chunk, delta: d });
            contentAcc += d.content ?? "";
          }
        }
      }
    }
  }

  if (!started) {
    // Empty upstream — still emit a well-formed role + stop pair.
    yield makeChunk(template, { role: "assistant" });
  }
  if (reasoner) {
    const r = reasoner.flush();
    if (r.reasoning.length > 0) yield makeChunk(template, { reasoning_content: r.reasoning });
    for (const d of parser.push(r.content)) {
      if (d.tool_calls) {
        yield makeChunk(template, d);
      } else {
        contentChunks.push({ chunk: template, delta: d });
        contentAcc += d.content ?? "";
      }
    }
  }
  for (const d of parser.flush()) {
    if (d.tool_calls) {
      yield makeChunk(template, d);
    } else {
      contentChunks.push({ chunk: template, delta: d });
      contentAcc += d.content ?? "";
    }
  }

  // If no tool calls were emitted and content is pure meta-talk, discard it.
  if (cleanMetaTalk(contentAcc) === null) {
    contentChunks.length = 0;
    contentAcc = "";
  }
  // Promise without action: "Сейчас сделаю X" / "Let me do X" with no tool calls.
  if (parser.toolCallCount === 0 && contentAcc.length < 200 && PROMISE_RE.test(contentAcc.trim())) {
    contentChunks.length = 0;
    contentAcc = "";
  }
  for (const { chunk, delta } of contentChunks) yield makeChunk(chunk, delta);

  yield makeChunk(
    template,
    {},
    parser.toolCallCount > 0 ? "tool_calls" : "stop",
    template?.usage ?? undefined,
  );
}

/**
 * Wraps an OpenAI-compatible client so that `tools` / `tool_choice` work even
 * when the underlying model has no native function calling. The returned
 * client's `chat.completions.create` is a drop-in for OpenAI's: it accepts the
 * same params and returns OpenAI-shaped `tool_calls` (and streamed tool-call
 * deltas). When no `tools` are passed it is fully transparent.
 */
export function wrapToolSupport(
  client: ChatClientLike,
  options: WrapOptions = {},
): ToolCapableClient {
  const toolCallTag = options.toolCallTag ?? DEFAULT_TOOL_CALL_TAG;
  const toolResultTag = options.toolResultTag ?? DEFAULT_TOOL_RESULT_TAG;
  const lenientFences = options.lenientFences ?? true;
  const systemInjection = options.systemInjection ?? "merge";
  const reasoning = options.reasoning ?? true;
  const reasoningTag = options.reasoningTag;
  const xmlToolCalls = options.xmlToolCalls ?? false;

  const promptOptions: PromptOptions = {
    toolCallTag,
    toolResultTag,
    includeExamples: options.includeExamples,
    template: options.template,
  };

  function create(
    body: ChatCompletionCreateParamsNonStreaming,
    requestOptions?: unknown,
  ): Promise<ChatCompletion>;
  function create(
    body: ChatCompletionCreateParamsStreaming,
    requestOptions?: unknown,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  function create(
    body: ChatCompletionCreateParams,
    requestOptions?: unknown,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>;
  async function create(
    body: ChatCompletionCreateParams,
    requestOptions?: unknown,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    const { tools, tool_choice, parallel_tool_calls, messages, stream, ...rest } = body;

    // Always flatten native tool roles — the model can't read them otherwise.
    let outMessages = flattenMessages(dedupInterTurn(messages), { toolCallTag, toolResultTag });

    const useTools = !!tools && tools.length > 0 && tool_choice !== "none";
    if (useTools) {
      // Prompted tool calling parses a single completion; n>1 would silently
      // drop the alternatives (and OpenAI forbids n>1 with tools anyway).
      const n = (rest as { n?: unknown }).n;
      if (typeof n === "number" && n > 1) {
        throw new ToolCapabilityError("n > 1 is not supported together with tools.");
      }
    }
    if (useTools) {
      const instruction =
        buildToolPrompt(tools!, promptOptions) +
        buildToolChoiceInstruction(tool_choice);
      outMessages = injectSystemPrompt(outMessages, instruction, systemInjection);
    }

    const outBound: ChatCompletionCreateParams = {
      ...rest,
      messages: outMessages,
      ...(stream !== undefined ? { stream } : {}),
    };

    const transformOpts: TransformOptions = {
      toolCallTag,
      generateId: options.generateId,
      lenientFences,
      reasoning,
      reasoningTag,
      xmlToolCalls,
      tools,
    };

    if (stream) {
      const source = (await client.chat.completions.create(
        outBound,
        requestOptions,
      )) as AsyncIterable<ChatCompletionChunk>;
      if (!useTools) return source;
      return transformStream(source, transformOpts);
    }

    const res = (await client.chat.completions.create(
      outBound,
      requestOptions,
    )) as ChatCompletion;
    if (!useTools) return res;
    return transformResponse(res, transformOpts);
  }

  return { chat: { completions: { create } } };
}
