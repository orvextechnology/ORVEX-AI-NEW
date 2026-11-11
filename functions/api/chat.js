
const FIREBASE_API_KEY = "AIzaSyD4oCtqtD8U1wnErSP10iu2sIr0XzaqrQs";
const TEXT_MODEL = "gemini-3.8-flash";
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

async function verifyFirebaseToken(idToken) {
  const response = await fetch(
    `https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${FIREBASE_API_KEY}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ idToken })
    }
  );

  const data = await response.json().catch(() => ({}));

  if (
    !response.ok ||
    !Array.isArray(data.users) ||
    !data.users[0]
  ) {
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
  const estimatedBytes = Math.floor(
    (base64.length * 3) / 4
  );

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

function buildContents(history, message, image) {
  const safeHistory = Array.isArray(history)
    ? history.slice(-12)
    : [];

  const contents = safeHistory
    .filter(item =>
      item &&
      (item.role === "user" || item.role === "assistant") &&
      typeof item.content === "string"
    )
    .map(item => ({
      role: item.role === "assistant" ? "model" : "user",
      parts: [
        {
          text: item.content.slice(0, 12000)
        }
      ]
    }));

  if (
    !contents.length ||
    contents[contents.length - 1].role !== "user"
  ) {
    contents.push({
      role: "user",
      parts: [{ text: message }]
    });
  } else {
    contents[contents.length - 1].parts[0].text = message;
  }

  if (image) {
    contents[contents.length - 1].parts.push({
      inlineData: {
        mimeType: image.mimeType,
        data: image.data
      }
    });
  }

  return contents;
}

function extractText(data) {
  const parts =
    data?.candidates?.[0]?.content?.parts || [];

  return parts
    .map(part => part.text || "")
    .join("\n")
    .trim();
}

function extractSources(data) {
  const chunks =
    data?.candidates?.[0]?.groundingMetadata?.groundingChunks || [];

  const seen = new Set();

  return chunks
    .map(chunk => chunk.web)
    .filter(web => web && web.uri && web.title)
    .filter(web => {
      if (seen.has(web.uri)) return false;
      seen.add(web.uri);
      return true;
    })
    .slice(0, 8)
    .map(web => ({
      title: web.title,
      url: web.uri
    }));
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

  const response = await fetch(
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
    }
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data?.error?.message ||
      "Image service request failed."
    );
  }

  let imageBlock = data?.output_image || null;
  let text = "";

  if (!imageBlock && Array.isArray(data.steps)) {
    for (const step of data.steps) {
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
        error: "ORVEX AI is not configured yet. Please contact support."
      }, 503);
    }

    const authorization =
      request.headers.get("Authorization") || "";

    const tokenMatch = authorization.match(
      /^Bearer\s+(.+)$/i
    );

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

    const message =
      typeof body.message === "string"
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

    /* IMAGE GENERATION AND EDITING */

    if (
      mode === "generate_image" ||
      mode === "edit_image"
    ) {
      const prompt = mode === "edit_image"
        ? `Edit the supplied image according to these instructions. Preserve details not mentioned in the request. User instructions: ${message}`
        : `Generate a high-quality image based on this description: ${message}`;

      const result = await runImageModel(
        env.ORVEX_GEMINI_KEY,
        prompt,
        mode === "edit_image" ? image : null
      );

      return jsonResponse(result);
    }

    /* CHAT, SEARCH, TRANSLATION AND IMAGE ANALYSIS */

    let instruction = SYSTEM_PROMPT;

    if (mode === "web_search") {
      instruction +=
        " Use Google Search grounding to research the user's question. Cite sources in the answer when available.";
    }

    if (mode === "translate") {
      instruction +=
        " This is translation mode. Translate the user's text into the language they specify. If they did not specify a target language, ask which language they want.";
    }

    if (mode === "analyze_image") {
      instruction +=
        " Analyze the attached image carefully. Describe visible details and distinguish observation from inference.";
    }

    const requestBody = {
      contents: buildContents(
        body.history,
        message,
        image
      ),
      systemInstruction: {
        parts: [{ text: instruction }]
      },
      generationConfig: {
        temperature: mode === "translate" ? 0.2 : 0.6,
        maxOutputTokens: 1800
      }
    };

    if (mode === "web_search") {
      requestBody.tools = [
        {
          google_search: {}
        }
      ];
    }

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${TEXT_MODEL}:generateContent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": env.ORVEX_GEMINI_KEY
        },
        body: JSON.stringify(requestBody)
      }
    );

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const apiMessage =
        data?.error?.message ||
        "The AI service request failed.";

      return jsonResponse({
        error: apiMessage
      }, response.status >= 500 ? 502 : 400);
    }

    const reply = extractText(data);

    if (!reply) {
      return jsonResponse({
        error: "ORVEX received an empty answer. Please try again."
      }, 502);
    }

    const sources = mode === "web_search"
      ? extractSources(data)
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
    return jsonResponse({
      error:
        error?.message ||
        "An unexpected server error occurred. Please try again."
    }, 500);
  }
      }
      
