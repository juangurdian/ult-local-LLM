const API_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE) ||
  "http://localhost:8001/api";

export type ImageGenRequest = {
  prompt: string;
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  seed?: number;
  negative_prompt?: string;
};

export type ImageGenResponse = {
  success: boolean;
  message: string;
  image_base64: string | null;
  seed: number | null;
  prompt: string | null;
};

export async function generateImage(
  request: ImageGenRequest
): Promise<ImageGenResponse> {
  const response = await fetch(`${API_BASE}/images/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Image generation failed (${response.status})`);
  }

  return response.json();
}

export async function checkComfyUIHealth(): Promise<{
  comfyui_available: boolean;
  server_address: string;
}> {
  const response = await fetch(`${API_BASE}/images/health`);
  if (!response.ok) {
    return { comfyui_available: false, server_address: "127.0.0.1:8188" };
  }
  return response.json();
}

