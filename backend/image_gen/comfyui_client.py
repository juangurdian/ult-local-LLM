"""Placeholder ComfyUI client. Wire up actual ComfyUI calls in Phase 4."""

from typing import Optional


class ComfyUIClient:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, steps: int = 20) -> Optional[bytes]:
        # TODO: Implement actual ComfyUI API calls
        return None

