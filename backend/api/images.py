from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from ..image_gen.comfyui_client import ComfyUIClient

router = APIRouter(prefix="/images", tags=["images"])
logger = logging.getLogger(__name__)

# Initialize ComfyUI client
comfy_client = ComfyUIClient()


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    seed: Optional[int] = None
    negative_prompt: Optional[str] = "blurry, low quality, distorted, watermark"


class ImageResponse(BaseModel):
    success: bool
    message: str
    image_base64: Optional[str] = None
    seed: Optional[int] = None
    prompt: Optional[str] = None


@router.get("/health")
async def check_comfyui_health():
    """Check if ComfyUI server is accessible."""
    is_connected = comfy_client.check_connection()
    return {
        "comfyui_available": is_connected,
        "server_address": comfy_client.server_address
    }


@router.post("/generate", response_model=ImageResponse)
async def generate_image(body: ImageRequest):
    """Generate an image from a text prompt using ComfyUI."""
    try:
        # Check connection first
        if not comfy_client.check_connection():
            raise HTTPException(
                status_code=503,
                detail="ComfyUI server is not accessible. Make sure ComfyUI is running on 127.0.0.1:8188"
            )

        logger.info(f"Generating image: prompt='{body.prompt[:50]}...', size={body.width}x{body.height}")

        # Generate image
        image_base64 = comfy_client.generate_image_base64(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            cfg=body.cfg,
            seed=body.seed
        )

        logger.info("Image generated successfully")

        return ImageResponse(
            success=True,
            message="Image generated successfully",
            image_base64=image_base64,
            seed=body.seed,
            prompt=body.prompt
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Image generation failed: {exc}", exc_info=True)
        return ImageResponse(
            success=False,
            message=f"Image generation failed: {str(exc)}",
            image_base64=None,
            seed=body.seed,
            prompt=body.prompt
        )

