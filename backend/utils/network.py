import httpx


async def is_online(test_url: str = "https://www.google.com", timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            await client.get(test_url)
        return True
    except Exception:
        return False

