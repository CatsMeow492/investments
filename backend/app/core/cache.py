import redis.asyncio as redis
import pickle
from datetime import datetime, timezone
from typing import Any, Optional
from .config import settings

# Initialize Redis client
redis_client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=False)

async def get_cached_data(key: str, cache_type: str) -> Optional[dict]:
    """Get data from cache if available"""
    try:
        cached_data = await redis_client.get(key)
        if cached_data:
            data = pickle.loads(cached_data)
            # Check if the cached data includes a timestamp and is still valid
            if isinstance(data, dict) and "timestamp" in data:
                age = (datetime.now(timezone.utc) - data["timestamp"]).total_seconds()
                ttl = getattr(settings, f"CACHE_TTL_{cache_type.upper()}")
                if age < ttl:
                    print(f"Cache hit for {key}")
                    return data["data"]
            await redis_client.delete(key)  # Delete expired cache
    except Exception as e:
        print(f"Cache error for {key}: {str(e)}")
    return None

async def set_cached_data(key: str, data: Any, cache_type: str):
    """Store data in cache with timestamp"""
    try:
        cache_data = {
            "timestamp": datetime.now(timezone.utc),
            "data": data
        }
        ttl = getattr(settings, f"CACHE_TTL_{cache_type.upper()}")
        await redis_client.setex(
            key,
            ttl,
            pickle.dumps(cache_data)
        )
    except Exception as e:
        print(f"Cache set error for {key}: {str(e)}")

async def clear_cache(pattern: str = "*"):
    """Clear cache entries matching the given pattern"""
    try:
        keys = await redis_client.keys(pattern)
        if keys:
            await redis_client.delete(*keys)
            print(f"Cleared {len(keys)} cache entries matching pattern: {pattern}")
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")

# Startup and shutdown events
async def connect_to_redis():
    """Initialize Redis connection"""
    await redis_client.ping()
    print("Connected to Redis")

async def close_redis():
    """Close Redis connection"""
    await redis_client.close()
    print("Closed Redis connection")
