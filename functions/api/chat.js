
const TEXT_MODEL = "gemini-3.5-flash-lite";
const BACKUP_TEXT_MODEL = "gemini-3.6-flash";
const IMAGE_ANALYSIS_MODEL = "gemini-3.6-flash";
const IMAGE_MODEL = "gemini-3.1-flash-image";

const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

const SYSTEM_PROMPT = `
You are ORVEX AI, a helpful, intelligent, accurate and respectful AI assistant.
Answer clearly, naturally and directly.
Prioritize fast, useful answers without unnecessary explanations.
Give detailed answers when the user asks for detail.
If you are unsure, say so honestly.
Never invent facts, sources or capabilities.
Translate faithfully and preserve meaning and tone.
For web research, use Google Search and provide sources where available.
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

async function fetchWithTimeout(url, options, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The AI service timed out.");
    }

    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function verifyFirebaseToken(idToken, apiKey) {
  if (!apiKey || !idToken) {
    return null;
  }

  const response = await fetchWithTimeout(
    `https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ idToken })
    },
    12000
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok || !Array.isArray(data.users) || !data.users[0]) {
    return null;
  }

  const user = data.users[0];

  if (user.disabled === true) {
    return null;
  }

  return {
    uid: user.localId,
    email: user.email || ""
  };
}

function parseImageData(imageData) {
  if (typeof imageData !== "string" || !imageData) {
    return null;
  }

  const match = imageData.match(
    /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$/
  );

  if (!match) {
    throw new Error(
      "Unsupported image. Please use PNG, JPEG or WEBP."
    );
  }

  const base64 = match[2];
  const estimatedBytes = Math.floor((base64.length * 3) / 4);

  if (estimatedBytes > MAX_IMAGE_BYTES) {
    throw new Error(
      "Image is too large. Please upload an image under 5 MB."
    );
  }

  return {
    mimeType: match[1],
    data: base64
  };
}

function buildInteractionInput(history, message, image) {
  const safeHistory = Array.isArray(history)
    ? history.slice(-8)
    : [];

  const messages = safeHistory.filter(item =>
    item &&
    (item.role === "user" || item.role === "assistant") &&
    typeof item.content === "string"
  );

  if (
    !messages.length ||
    messages[messages.length - 1].role !== "user"
  ) {
    messages.push({
      role: "user",
      content: message
    });
  } else {
    messages[messages.length - 1].content = message;
  }

  const input = messages.map(item => ({
    type: item.role === "assistant"
      ? "model_output"
      : "user_input",
    content: [{
      type: "text",
      text: item.content.slice(0, 6000)
    }]
  }));

  if (image && input.length) {
    input[input.length - 1].content.push({
      type: "image",
      mime_type: image.mimeType,
      data: image.data
    });
  }

  return input;
}

function extractInteractionText(data) {
  const steps = Array.isArray(data?.steps)
    ? data.steps
    : [];

  const textParts = [];

  for (const step of steps) {
    if (step.type !== "model_output") continue;

    for (const item of step.content || []) {
      if (
        item.type === "text" &&
        typeof item.text === "string"
      ) {
        textParts.push(item.text);
      }
    }
  }

  return textParts.join("\n").trim();
}

function extractInteractionSources(data) {
  const steps = Array.isArray(data?.steps)
    ? data.steps
    : [];

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

// Calls models in order and switches to a backup when necessary.
async function callTextModelWithFallback(
  apiKey,
  requestBody,
  models = [TEXT_MODEL, BACKUP_TEXT_MODEL]
) {
  const endpoint =
    "https://generativelanguage.googleapis.com/v1beta/interactions";

  let lastResult = {
    response: null,
    data: {
      error: {
        message: "All AI models failed to respond."
      }
    }
  };

  for (const model of models) {
    try {
      const response = await fetchWithTimeout(
        endpoint,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-goog-api-key": apiKey
          },
          body: JSON.stringify({
            ...requestBody,
            model
          })
        },
        15000
      );

      const data = await response.json().catch(() => ({}));
      const reply = extractInteractionText(data);

      if (
        response.ok &&
        data.status !== "failed" &&
        data.status !== "cancelled" &&
        reply
      ) {
        return {
          response,
          data
        };
      }

      const errorMessage =
        data?.error?.message ||
        (
          !response.ok
            ? `Model ${model} failed with HTTP ${response.status}.`
            : data.status === "failed" ||
              data.status === "cancelled"
              ? `Model ${model} could not complete the request.`
              : `Model ${model} returned no text response.`
        );

      lastResult = {
        response,
        data: {
          ...data,
          error: {
            ...data.error,
            message: errorMessage
          }
        }
      };

      if (
        response.status === 400 ||
        response.status === 401 ||
        response.status === 403
      ) {
        return lastResult;
      }

    } catch (error) {
      lastResult = {
        response: null,
        data: {
          error: {
            message: error.message ||
              "Network connection failed."
          }
        }
      };
    }
  }

  return lastResult;
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
      data?.error?.message ||
      "The image service request failed."
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
    imageBlock.mime_type ||
    imageBlock.mimeType ||
    "image/png";

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

    if (!env.FIREBASE_API_KEY) {
      return jsonResponse({
        error: "Firebase authentication is not configured. Please contact support."
      }, 503);
    }

    const authorization =
      request.headers.get("Authorization") || "";

    const tokenMatch =
      authorization.match(/^Bearer\s+(.+)$/i);

    if (!tokenMatch) {
      return jsonResponse({
        error: "Please log in to use ORVEX AI."
      }, 401);
    }

    const user = await verifyFirebaseToken(
      tokenMatch[1],
      env.FIREBASE_API_KEY
    );

    if (!user) {
      return jsonResponse({
        error: "Your session is invalid or expired. Please log in again."
      }, 401);
    }

    const body = await request.json().catch(() => null);

    if (!body || typeof body !== "object") {
      return jsonResponse({
        error: "Invalid request body."
      }, 400);
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
      return jsonResponse({
        error: "Please enter a message."
      }, 400);
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
      return jsonResponse({
        error: error.message
      }, 400);
    }

    if (
      (mode === "analyze_image" || mode === "edit_image") &&
      !image
    ) {
      return jsonResponse({
        error: "Please attach an image first."
      }, 400);
    }

    // Image generation and editing use the dedicated image model.
    if (
      mode === "generate_image" ||
      mode === "edit_image"
    ) {
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

    // Web search is handled separately from the AI text models.
    if (mode === "web_search") {
      if (!env.TAVILY_API_KEY) {
        return jsonResponse({
          error: "Web search is not configured. Please check the Tavily API secret."
        }, 500);
      }

      const searchResponse = await fetchWithTimeout(
        "https://api.tavily.com/search",
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${env.TAVILY_API_KEY}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            query: message,
            search_depth: "fast",
            topic: "general",
            max_results: 5,
            include_answer: false
          })
        },
        15000
      );

      const searchData =
        await searchResponse.json().catch(() => ({}));

      if (!searchResponse.ok) {
        return jsonResponse({
          error: searchData?.detail?.error ||
            "Web search provider could not complete the search."
        }, searchResponse.status);
      }

      const sources = (searchData.results || [])
        .filter(source => source.title && source.url)
        .slice(0, 5);

      if (!sources.length) {
        return jsonResponse({
          error: "No search results were found. Try another search."
        }, 404);
      }

      const searchReply =
        "## Web Search Results\n\n" +
        sources.map((source, index) =>
          `### ${index + 1}. ${source.title}\n\n` +
          `${source.content || "Open the source to read more."}\n\n` +
          `Source: ${source.url}`
        ).join("\n\n");

      return jsonResponse({
        reply: searchReply
      });
    }

    let instruction = SYSTEM_PROMPT;

    if (mode === "translate") {
      instruction +=
        " Translate faithfully into the target language specified by the user. If no target language is specified, ask which language they want.";
    }

    if (mode === "analyze_image") {
      instruction +=
        " Analyze the attached image carefully. Describe visible details, objects, text and relevant features. Distinguish direct observation from inference. Do not claim to see details that are unclear.";
    }

    const requestBody = {
      model: TEXT_MODEL,
      store: false,
      input: buildInteractionInput(
        body.history,
        message,
        image
      ),
      system_instruction: instruction,
      generation_config: {
        thinking_level: "low",
        max_output_tokens: 1200
      }
    };

    // Image analysis starts with a dedicated multimodal model.
    // Normal chat starts with the faster Flash-Lite model.
    const modelsToTry = mode === "analyze_image"
      ? [
          IMAGE_ANALYSIS_MODEL,
          TEXT_MODEL,
          BACKUP_TEXT_MODEL
        ]
      : [
          TEXT_MODEL,
          BACKUP_TEXT_MODEL
        ];

    const result = await callTextModelWithFallback(
      env.ORVEX_GEMINI_KEY,
      requestBody,
      modelsToTry
    );

    if (!result.response) {
      return jsonResponse({
        error: result.data?.error?.message ||
          "The AI models failed to respond. Please try again."
      }, 504);
    }

    if (!result.response.ok) {
      return jsonResponse({
        error: result.data?.error?.message ||
          `Gemini request failed with status ${result.response.status}.`
      }, result.response.status >= 500 ? 502 : 400);
    }

    const data = result.data;

    if (
      data.status === "failed" ||
      data.status === "cancelled"
    ) {
      return jsonResponse({
        error: data?.error?.message ||
          "The AI service could not complete this request. Please try again."
      }, 502);
    }

    const reply = extractInteractionText(data);

    if (!reply) {
      return jsonResponse({
        error: "ORVEX received no text response. Please try again."
      }, 502);
    }

    return jsonResponse({
      reply
    });

  } catch (error) {
    const isTimeout =
      error?.name === "AbortError" ||
      error?.name === "TimeoutError" ||
      /timed out|took too long/i.test(error?.message || "");

    return jsonResponse({
      error: isTimeout
        ? "The AI service timed out. Please try again."
        : error?.message || "An unexpected server error occurred."
    }, isTimeout ? 504 : 500);
  }
}
  
