from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..image_gen.comfyui_client import ComfyUIClient

router = APIRouter(prefix="/images", tags=["images"])
comfy_client = ComfyUIClient()


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    negative_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    success: bool
    message: str
    image_base64: Optional[str] = None


@router.post("/generate", response_model=ImageResponse)
async def generate_image(body: ImageRequest):
    """Image generation endpoint (currently stubbed to ComfyUI)."""
    try:
        # Placeholder: actual implementation will call comfy_client.generate_image(...)
        return ImageResponse(
            success=True,
            message="Image generation stub (wire ComfyUI)",
            image_base64=None,
        )
    except Exception as exc:
        return ImageResponse(success=False, message=str(exc), image_base64=None)

