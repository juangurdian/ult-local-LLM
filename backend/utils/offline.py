from .network import is_online


async def offline_safe_feature() -> dict:
    """Simple helper to illustrate offline gating."""
    online = await is_online()
    return {"online": online, "message": "Feature available" if online else "Offline mode: limited features"}

