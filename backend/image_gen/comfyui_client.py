"""
ComfyUI client for image generation via API and WebSocket.
"""

import json
import uuid
import urllib.request
import urllib.parse
import websocket
import time
import logging
from typing import Optional, Dict, Any
import base64

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for interacting with ComfyUI API and WebSocket."""

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def _get_base_url(self) -> str:
        """Get base URL for HTTP requests."""
        return f"http://{self.server_address}"

    def _get_ws_url(self) -> str:
        """Get WebSocket URL."""
        return f"ws://{self.server_address}/ws?clientId={self.client_id}"

    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow and return the prompt ID."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self._get_base_url()}/prompt",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            prompt_id = result.get('prompt_id')
            if not prompt_id:
                raise ValueError(f"Failed to queue prompt: {result}")
            logger.info(f"Queued prompt: {prompt_id}")
            return prompt_id
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise Exception(f"HTTP error {e.code}: {error_body}")
        except Exception as e:
            raise Exception(f"Failed to queue prompt: {e}")

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Retrieve generated image from ComfyUI."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        })
        url = f"{self._get_base_url()}/view?{params}"
        try:
            with urllib.request.urlopen(url) as response:
                return response.read()
        except Exception as e:
            raise Exception(f"Failed to retrieve image: {e}")

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get generation history for a prompt."""
        url = f"{self._get_base_url()}/history/{prompt_id}"
        try:
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read())
        except Exception as e:
            raise Exception(f"Failed to get history: {e}")

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for prompt completion via WebSocket."""
        ws_url = self._get_ws_url()
        start_time = time.time()

        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=10)
            logger.info(f"Connected to ComfyUI WebSocket, waiting for prompt {prompt_id}")

            while time.time() - start_time < timeout:
                message = ws.recv()
                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get('type')

                    if msg_type == 'executing':
                        node_id = data.get('data', {}).get('node')
                        if node_id is None:
                            # Execution finished
                            logger.info(f"Prompt {prompt_id} completed")
                            ws.close()
                            return self.get_history(prompt_id)
                    elif msg_type == 'progress':
                        # Log progress
                        progress = data.get('data', {})
                        logger.debug(f"Progress: {progress.get('value', 0)}/{progress.get('max', 100)}")
                    elif msg_type == 'execution_error':
                        error = data.get('data', {})
                        ws.close()
                        raise Exception(f"Execution error: {error}")

            ws.close()
            raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")

        except websocket.WebSocketException as e:
            raise Exception(f"WebSocket error: {e}")
        except Exception as e:
            raise Exception(f"Error waiting for completion: {e}")

    def _build_sdxl_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build SDXL workflow JSON."""
        if seed is None:
            seed = uuid.uuid4().int % (2**32)

        # SDXL workflow structure
        workflow = {
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt or "blurry, low quality, distorted",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "beast_gen",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

        return workflow

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        timeout: int = 300
    ) -> bytes:
        """
        Generate an image and return bytes.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt
            width: Image width (default 1024)
            height: Image height (default 1024)
            steps: Number of sampling steps (default 20)
            cfg: CFG scale (default 7.0)
            seed: Random seed (None for random)
            timeout: Timeout in seconds (default 300)
        
        Returns:
            Image bytes (PNG format)
        """
        try:
            # Build workflow
            workflow = self._build_sdxl_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed
            )

            # Queue the prompt
            prompt_id = self.queue_prompt(workflow)

            # Wait for completion
            history = self.wait_for_completion(prompt_id, timeout=timeout)

            # Extract image from history
            if prompt_id not in history:
                raise Exception(f"Prompt {prompt_id} not found in history")

            outputs = history[prompt_id].get('outputs', {})
            images = []

            # Find SaveImage node output
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    for image_info in node_output['images']:
                        image_data = self.get_image(
                            filename=image_info['filename'],
                            subfolder=image_info.get('subfolder', ''),
                            folder_type=image_info.get('type', 'output')
                        )
                        images.append(image_data)

            if not images:
                raise Exception("No images generated")

            logger.info(f"Generated image successfully ({len(images[0])} bytes)")
            return images[0]

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    def generate_image_base64(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None
    ) -> str:
        """Generate image and return as base64 string."""
        image_bytes = self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        return base64.b64encode(image_bytes).decode('utf-8')

    def check_connection(self) -> bool:
        """Check if ComfyUI server is accessible."""
        try:
            url = f"{self._get_base_url()}/system_stats"
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
