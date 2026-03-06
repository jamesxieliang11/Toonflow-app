import "../type";
import { pollTask } from "@/utils/ai/utils";
import u from "@/utils";
import axios from "axios";

/**
 * DashScope 万相文生图实现
 * 支持模型：wan2.6-t2i, wan2.5-t2i-preview, wan2.2-t2i-flash, wan2.2-t2i-plus, wanx2.1-t2i-turbo, wanx2.1-t2i-plus 等
 * 使用与 qwen-max 相同的 DashScope API Key
 */

const WAN26_MODELS = ["wan2.6-t2i"];

// qwen-image-2.0 系列使用与 wan2.6 相同的同步 multimodal-generation 接口
const QWEN_IMAGE_MODELS_PREFIX = ["qwen-image-2.0"];

function isMultimodalGenerationModel(model: string): boolean {
  return WAN26_MODELS.includes(model) || QWEN_IMAGE_MODELS_PREFIX.some((prefix) => model.startsWith(prefix));
}

const PROMPT_OPTIMIZE_MODEL = "qwen-max";
const PROMPT_LENGTH_THRESHOLD = 200;

/**
 * 使用 qwen-max 将复杂的中文长 prompt 精简翻译为适合文生图模型的英文短 prompt。
 * 万相系列对简短、直接的英文画面描述理解力更强，此预处理可显著提升出图相关性。
 */
async function optimizePromptForImageGen(rawPrompt: string, apiKey: string): Promise<string> {
  const systemInstruction = `You are an expert prompt engineer for text-to-image AI models (like Stable Diffusion, DALL-E, Midjourney).
Your task is to convert a detailed Chinese character/scene description into an optimized English image generation prompt.

Rules:
1. Output ONLY the final English prompt, nothing else.
2. Keep it under 500 words. Be concise but preserve all critical visual details.
3. Use comma-separated descriptive tags/phrases, which is the standard format for image generation models.
4. Prioritize: subject description, appearance, clothing, pose, art style, background, camera angle.
5. If the input describes a multi-view character sheet (e.g. front/side/back views), clearly specify "character reference sheet, multiple views" at the beginning.
6. Translate all visual details accurately. Do not add or invent details not in the original.
7. Put the most important visual elements (art style, subject type) at the beginning of the prompt.
8. Include negative-prompt-worthy exclusions as positive instructions (e.g. "no text, no weapons" → "clean background, no text overlay, no weapons").`;

  const chatUrl = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";

  try {
    const { data } = await axios.post(
      chatUrl,
      {
        model: PROMPT_OPTIMIZE_MODEL,
        messages: [
          { role: "system", content: systemInstruction },
          { role: "user", content: rawPrompt },
        ],
        temperature: 0.3,
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        timeout: 30000,
      },
    );

    const optimizedPrompt = data?.choices?.[0]?.message?.content?.trim();
    if (optimizedPrompt) {
      console.log("[qwen] prompt 预处理完成，原始长度:", rawPrompt.length, "优化后长度:", optimizedPrompt.length);
      return optimizedPrompt;
    }
  } catch (error) {
    console.warn("[qwen] prompt 预处理失败，将使用原始 prompt:", error instanceof Error ? error.message : error);
  }

  return rawPrompt;
}

// wan2.5+ 模型支持更大分辨率
const sizeMapLarge: Record<string, Record<string, string>> = {
  "1K": {
    "16:9": "1696*960",
    "9:16": "960*1696",
    "1:1": "1280*1280",
    "4:3": "1472*1104",
    "3:4": "1104*1472",
  },
  "2K": {
    "16:9": "1696*960",
    "9:16": "960*1696",
    "1:1": "1440*1440",
    "4:3": "1472*1104",
    "3:4": "1104*1472",
  },
  "4K": {
    "16:9": "1696*960",
    "9:16": "960*1696",
    "1:1": "1440*1440",
    "4:3": "1472*1104",
    "3:4": "1104*1472",
  },
};

// wan2.2 及以下模型（wanx2.1、wanx2.0 等）宽高限制在 512~1440
const sizeMapSmall: Record<string, Record<string, string>> = {
  "1K": {
    "16:9": "1280*720",
    "9:16": "720*1280",
    "1:1": "1024*1024",
    "4:3": "1152*864",
    "3:4": "864*1152",
  },
  "2K": {
    "16:9": "1440*816",
    "9:16": "816*1440",
    "1:1": "1024*1024",
    "4:3": "1152*864",
    "3:4": "864*1152",
  },
  "4K": {
    "16:9": "1440*816",
    "9:16": "816*1440",
    "1:1": "1440*1440",
    "4:3": "1152*864",
    "3:4": "864*1152",
  },
};

const LARGE_SIZE_MODELS = ["wan2.5", "wan2.6"];

function getSize(model: string, inputSize: string, aspectRatio: string): string {
  const isLargeModel = LARGE_SIZE_MODELS.some((prefix) => model.startsWith(prefix));
  const map = isLargeModel ? sizeMapLarge : sizeMapSmall;
  return map[inputSize]?.[aspectRatio] ?? "1024*1024";
}

function getBaseUrl(configBaseURL: string | undefined): string {
  if (configBaseURL && configBaseURL.trim()) {
    return configBaseURL.replace(/\/+$/, "");
  }
  return "https://dashscope.aliyuncs.com/api/v1";
}

/**
 * wan2.6 模型使用新版同步协议
 */
async function callWan26Sync(prompt: string, negativePrompt: string, size: string, apiKey: string, model: string, baseUrl: string): Promise<string> {
  const requestUrl = `${baseUrl}/services/aigc/multimodal-generation/generation`;

  const { data } = await axios.post(
    requestUrl,
    {
      model,
      input: {
        messages: [
          {
            role: "user",
            content: [{ text: prompt }],
          },
        ],
      },
      parameters: {
        negative_prompt: negativePrompt,
        size,
        n: 1,
        prompt_extend: false,
        watermark: false,
      },
    },
    {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      timeout: 120000,
    },
  );

  const imageUrl = data?.output?.choices?.[0]?.message?.content?.[0]?.image;
  if (!imageUrl) {
    throw new Error("wan2.6 图片生成失败：未返回图片URL");
  }
  return imageUrl;
}

/**
 * wan2.5 及以下版本使用异步协议（创建任务 -> 轮询获取）
 */
async function callWanAsync(prompt: string, negativePrompt: string, size: string, apiKey: string, model: string, baseUrl: string): Promise<string> {
  const createUrl = `${baseUrl}/services/aigc/text2image/image-synthesis`;
  const headers = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
    "X-DashScope-Async": "enable",
  };

  const { data } = await axios.post(
    createUrl,
    {
      model,
      input: {
        prompt,
        negative_prompt: negativePrompt,
      },
      parameters: {
        size,
        n: 1,
        prompt_extend: false,
        watermark: false,
      },
    },
    { headers, timeout: 30000 },
  );

  const taskId = data?.output?.task_id;
  if (!taskId) {
    throw new Error("DashScope 图片任务创建失败：未返回 task_id");
  }

  const queryUrl = `${baseUrl}/tasks/${taskId}`;

  return await pollTask(async () => {
    const { data: queryData } = await axios.get(queryUrl, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });

    const taskStatus = queryData?.output?.task_status;

    if (taskStatus === "FAILED") {
      const errorMessage = queryData?.output?.message || queryData?.output?.code || "图片生成失败";
      return { completed: false, error: errorMessage };
    }

    if (taskStatus === "SUCCEEDED") {
      // 新版协议（wan2.6异步）返回 choices 格式
      const choicesUrl = queryData?.output?.choices?.[0]?.message?.content?.[0]?.image;
      // 旧版协议（wan2.5及以下）返回 results 格式
      const resultsUrl = queryData?.output?.results?.[0]?.url;
      const imageUrl = choicesUrl || resultsUrl;
      if (!imageUrl) {
        return { completed: false, error: "图片生成成功但未返回图片URL" };
      }
      return { completed: true, url: imageUrl };
    }

    return { completed: false };
  });
}

export default async (input: ImageConfig, config: AIConfig): Promise<string> => {
  if (!config.model) throw new Error("缺少Model名称");
  if (!config.apiKey) throw new Error("缺少API Key");

  const apiKey = config.apiKey.replace("Bearer ", "");
  const baseUrl = getBaseUrl(config.baseURL);
  const rawPrompt = input.systemPrompt ? `${input.systemPrompt}\n\n${input.prompt}` : input.prompt;
  const size = getSize(config.model, input.size, input.aspectRatio);

  // 当 prompt 较长时，先用 qwen-max 精简翻译为英文短 prompt，提升万相模型出图相关性
  const finalPrompt = rawPrompt.length > PROMPT_LENGTH_THRESHOLD
    ? await optimizePromptForImageGen(rawPrompt, apiKey)
    : rawPrompt;

  try {
    if (isMultimodalGenerationModel(config.model)) {
      return await callWan26Sync(finalPrompt, "", size, apiKey, config.model, baseUrl);
    } else {
      return await callWanAsync(finalPrompt, "", size, apiKey, config.model, baseUrl);
    }
  } catch (error: any) {
    const msg = u.error(error).message || "DashScope 图片生成失败";
    throw new Error(msg);
  }
};
