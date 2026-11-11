
const FIREBASE_API_KEY = "AIzaSyD4oCtqtD8U1wnErSP10iu2sIr0XzaqrQs";
const TEXT_MODEL = "gemini-3.8-flash";
const BACKUP_TEXT_MODEL = "gemini-3.5-flash-lite";
const IMAGE_MODEL = "gemini-3.1-flash-image";
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

const SYSTEM_PROMPT = `
You are ORVEX AI, a helpful, accurate and respectful AI assistant.
Answer clearly and honestly.
If you are unsure, say so.
Never invent sources, facts or capabilities.
For translation requests, translate faithfully and preserve the original meaning and tone.
For web research, prioritize current, verifiable information and distinguish facts from uncertainty.
`;

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    }
  });
}

async function fetchWithTimeout(url, options, timeoutMs = 45000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The AI service took too long to respond. Please try again.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function verifyFirebaseToken(idToken) {
  const response = await fetchWithTimeout(
    `https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${FIREBASE_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idToken })
    },
    15000
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok || !Array.isArray(data.users) || !data.users[0]) {
    return null;
  }

  const user = data.users[0];
  if (user.disabled === true) return null;

  return {
    uid: user.localId,
    email: user.email || ""
  };
}

function parseImageData(imageData) {
  if (typeof imageData !== "string" || !imageData) return null;

  const match = imageData.match(
    /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$/
  );

  if (!match) {
    throw new Error("Unsupported image. Please use PNG, JPEG or WEBP.");
  }

  const base64 = match[2];
  const estimatedBytes = Math.floor((base64.length * 3) / 4);

  if (estimatedBytes > MAX_IMAGE_BYTES) {
    throw new Error("Image is too large. Please upload an image under 5 MB.");
  }

  return {
    mimeType: match[1],
    data: base64
  };
}

function buildInteractionInput(history, message, image) {
  const safeHistory = Array.isArray(history) ? history.slice(-12) : [];

  const messages = safeHistory.filter(item =>
    item &&
    (item.role === "user" || item.role === "assistant") &&
    typeof item.content === "string"
  );

  if (!messages.length || messages[messages.length - 1].role !== "user") {
    messages.push({ role: "user", content: message });
  } else {
    messages[messages.length - 1].content = message;
  }

  const input = messages.map(item => ({
    type: item.role === "assistant" ? "model_output" : "user_input",
    content: [{
      type: "text",
      text: item.content.slice(0, 12000)
    }]
  }));

  if (image) {
    input[input.length - 1].content.push({
      type: "image",
      mime_type: image.mimeType,
      data: image.data
    });
  }

  return input;
}

function extractInteractionText(data) {
  const steps = Array.isArray(data?.steps) ? data.steps : [];
  const textParts = [];

  for (const step of steps) {
    if (step.type !== "model_output") continue;

    for (const item of step.content || []) {
      if (item.type === "text" && typeof item.text === "string") {
        textParts.push(item.text);
      }
    }
  }

  return textParts.join("\n").trim();
}

function extractInteractionSources(data) {
  const steps = Array.isArray(data?.steps) ? data.steps : [];
  const sources = [];
  const seen = new Set();

  for (const step of steps) {
    if (step.type !== "model_output") continue;

    for (const item of step.content || []) {
      for (const annotation of item.annotations || []) {
        const url = annotation.url || annotation.uri;
        const title = annotation.title || url;

        if (
          typeof url === "string" &&
          /^https?:\/\//i.test(url) &&
          !seen.has(url)
        ) {
          seen.add(url);
          sources.push({ title, url });
        }
      }
    }
  }

  return sources.slice(0, 8);
}

// Automatically retry temporary overload using the backup model.
async function callTextModelWithFallback(apiKey, requestBody) {
  const endpoint =
    "https://generativelanguage.googleapis.com/v1beta/interactions";

  async function send(payload) {
    const response = await fetchWithTimeout(
      endpoint,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey
        },
        body: JSON.stringify(payload)
      },
      22000
    );

    const data = await response.json().catch(() => ({}));

    return { response, data };
  }

  let result = await send(requestBody);

  const errorMessage = String(
    result.data?.error?.message || ""
  ).toLowerCase();

  const errorStatus = String(
    result.data?.error?.status || ""
  ).toLowerCase();

  const temporaryOverload =
    result.response.status === 503 ||
    result.response.status === 429 ||
    /high demand|overload|temporarily unavailable|resource exhausted|try again later/i.test(errorMessage) ||
    /resource_exhausted|unavailable/i.test(errorStatus);

  const quotaProblem =
    /quota exceeded|billing|insufficient quota/i.test(errorMessage);

  if (
    !result.response.ok &&
    temporaryOverload &&
    !quotaProblem
  ) {
    result = await send({
      ...requestBody,
      model: BACKUP_TEXT_MODEL
    });
  }

  return result;
}

async function runImageModel(apiKey, prompt, image) {
  const input = [];

  if (image) {
    input.push({
      type: "image",
      mime_type: image.mimeType,
      data: image.data
    });
  }

  input.push({
    type: "text",
    text: prompt
  });

  const response = await fetchWithTimeout(
    "https://generativelanguage.googleapis.com/v1beta/interactions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey
      },
      body: JSON.stringify({
        model: IMAGE_MODEL,
        input,
        response_format: {
          type: "image",
          mime_type: "image/png",
          aspect_ratio: "1:1"
        }
      })
    },
    55000
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data?.error?.message || "The image service request failed."
    );
  }

  let imageBlock = data.output_image || null;
  let text = "";

  if (!imageBlock && Array.isArray(data.steps)) {
    for (const step of data.steps) {
      if (step.type !== "model_output") continue;

      for (const item of step.content || []) {
        if (item.type === "image" && item.data) {
          imageBlock = item;
        }

        if (item.type === "text" && item.text) {
          text += item.text + "\n";
        }
      }
    }
  }

  if (!imageBlock?.data) {
    throw new Error(
      "The image model did not return an image. Please try again."
    );
  }

  const mimeType =
    imageBlock.mime_type || imageBlock.mimeType || "image/png";

  return {
    reply: text.trim() || "Your image is ready.",
    imageUrl: `data:${mimeType};base64,${imageBlock.data}`
  };
}

export async function onRequestPost({ request, env }) {
  try {
    if (!env.ORVEX_GEMINI_KEY) {
      return jsonResponse({
        error: "ORVEX AI is not configured. Please contact support."
      }, 503);
    }

    const authorization = request.headers.get("Authorization") || "";
    const tokenMatch = authorization.match(/^Bearer\s+(.+)$/i);

    if (!tokenMatch) {
      return jsonResponse({
        error: "Please log in to use ORVEX AI."
      }, 401);
    }

    const user = await verifyFirebaseToken(tokenMatch[1]);

    if (!user) {
      return jsonResponse({
        error: "Your session is invalid or expired. Please log in again."
      }, 401);
    }

    const body = await request.json().catch(() => null);

    if (!body || typeof body !== "object") {
      return jsonResponse({ error: "Invalid request body." }, 400);
    }

    const allowedModes = [
      "chat",
      "web_search",
      "translate",
      "analyze_image",
      "generate_image",
      "edit_image"
    ];

    const mode = allowedModes.includes(body.mode)
      ? body.mode
      : "chat";

    const message = typeof body.message === "string"
      ? body.message.trim()
      : "";

    if (!message) {
      return jsonResponse({ error: "Please enter a message." }, 400);
    }

    if (message.length > 12000) {
      return jsonResponse({
        error: "Your message is too long. Please shorten it."
      }, 400);
    }

    let image = null;

    try {
      image = parseImageData(body.imageData);
    } catch (error) {
      return jsonResponse({ error: error.message }, 400);
    }

    if (
      (mode === "analyze_image" || mode === "edit_image") &&
      !image
    ) {
      return jsonResponse({
        error: "Please attach an image first."
      }, 400);
    }

    if (mode === "generate_image" || mode === "edit_image") {
      const prompt = mode === "edit_image"
        ? `Edit the supplied image according to these instructions. Preserve details not mentioned. User instructions: ${message}`
        : `Generate a high-quality image based on this description: ${message}`;

      const result = await runImageModel(
        env.ORVEX_GEMINI_KEY,
        prompt,
        mode === "edit_image" ? image : null
      );

      return jsonResponse(result);
    }

    let instruction = SYSTEM_PROMPT;

    if (mode === "web_search") {
      instruction +=
        " Research the user's question using Google Search. Include source citations where available.";
    }

    if (mode === "translate") {
      instruction +=
        " Translate faithfully into the target language specified by the user. If no target language is specified, ask which language they want.";
    }

    if (mode === "analyze_image") {
      instruction +=
        " Analyze the attached image carefully. Describe visible details and distinguish observation from inference.";
    }

    const requestBody = {
      model: TEXT_MODEL,
      store: false,
      input: buildInteractionInput(body.history, message, image),
      system_instruction: instruction,
      generation_config: {
        thinking_level: "low",
        max_output_tokens: 1800
      }
    };

    if (mode === "web_search") {
      requestBody.tools = [{ type: "google_search" }];
    }

    const { response, data } = await callTextModelWithFallback(
      env.ORVEX_GEMINI_KEY,
      requestBody
    );

    if (!response.ok) {
      return jsonResponse({
        error: data?.error?.message ||
          `Gemini request failed with status ${response.status}. Both text models may be unavailable.`
      }, response.status >= 500 ? 502 : 400);
    }

    if (
      data.status === "failed" ||
      data.status === "cancelled"
    ) {
      return jsonResponse({
        error: "The AI service could not complete this request. Please try again."
      }, 502);
    }

    const reply = extractInteractionText(data);

    if (!reply) {
      return jsonResponse({
        error: `ORVEX received no text response. Interaction status: ${data.status || "unknown"}.`
      }, 502);
    }

    const sources = mode === "web_search"
      ? extractInteractionSources(data)
      : [];

    const replyWithSources = sources.length
      ? reply + "\n\nSources:\n" +
        sources.map((source, index) =>
          `${index + 1}. ${source.title}\n${source.url}`
        ).join("\n")
      : reply;

    return jsonResponse({
      reply: replyWithSources
    });

  } catch (error) {
    const isTimeout =
      error?.name === "AbortError" ||
      /took too long/i.test(error?.message || "");

    return jsonResponse({
      error: isTimeout
        ? "The AI service timed out. Please try again."
        : error?.message || "An unexpected server error occurred."
    }, isTimeout ? 504 : 500);
  }
      }
          
