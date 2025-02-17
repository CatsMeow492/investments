import httpx
import json
from typing import Dict, Any, AsyncGenerator, List, Optional
from ..core.config import settings
from ..models.portfolio import ModelInfo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModel:
    OPENAI = "openai"
    OPENROUTER_CLAUDE = "openrouter_claude"
    OPENROUTER_MISTRAL = "openrouter_mistral"
    OPENROUTER_DEEPSEEK = "openrouter_deepseek"
    ANTHROPIC = "anthropic"

    @staticmethod
    def get_available_models() -> List[ModelInfo]:
        """Get list of available models with their status"""
        models = [
            ModelInfo(
                id=AIModel.OPENAI,
                name="GPT-4 Turbo",
                description="OpenAI's most capable model, best for complex analysis",
                is_available=bool(settings.OPENAI_API_KEY)
            ),
            ModelInfo(
                id=AIModel.OPENROUTER_CLAUDE,
                name="Claude 3 Opus",
                description="Anthropic's most capable model, excellent for detailed analysis and reasoning",
                is_available=bool(settings.OPENROUTER_API_KEY)
            ),
            ModelInfo(
                id=AIModel.OPENROUTER_MISTRAL,
                name="Mixtral 8x7B",
                description="Fast and efficient model with strong reasoning capabilities",
                is_available=bool(settings.OPENROUTER_API_KEY)
            ),
            ModelInfo(
                id=AIModel.OPENROUTER_DEEPSEEK,
                name="DeepSeek R1",
                description="Specialized model with strong coding and analysis capabilities",
                is_available=bool(settings.OPENROUTER_API_KEY)
            ),
            ModelInfo(
                id=AIModel.ANTHROPIC,
                name="Claude 3 Opus (Direct)",
                description="Direct access to Claude 3 Opus via Anthropic's API",
                is_available=bool(settings.ANTHROPIC_API_KEY)
            )
        ]
        available_models = [model for model in models if model.is_available]
        logger.info(f"Available AI models: {[model.name for model in available_models]}")
        return models

# OpenRouter base configuration
OPENROUTER_BASE_CONFIG = {
    "headers": {
        "HTTP-Referer": "https://github.com/CatsMeow492/investments",
        "X-Title": "Investment Portfolio Tracker",
        "Content-Type": "application/json"
    },
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Model-specific configurations
MODEL_CONFIGS = {
    AIModel.OPENAI: {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4-turbo-preview",
        "max_tokens": 4000,
    },
    AIModel.OPENROUTER_CLAUDE: {
        **OPENROUTER_BASE_CONFIG,
        "model": "anthropic/claude-3-opus",
        "name": "Claude 3 Opus",
        "description": "Anthropic's most capable model, excellent for detailed analysis and reasoning",
    },
    AIModel.OPENROUTER_MISTRAL: {
        **OPENROUTER_BASE_CONFIG,
        "model": "mistralai/mixtral-8x7b",
        "name": "Mixtral 8x7B",
        "description": "Fast and efficient model with strong reasoning capabilities",
    },
    AIModel.OPENROUTER_DEEPSEEK: {
        **OPENROUTER_BASE_CONFIG,
        "model": "deepseek/deepseek-r1",
        "name": "DeepSeek R1",
        "description": "Specialized model with strong coding and analysis capabilities",
    },
    AIModel.ANTHROPIC: {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-opus-20240229",
        "max_tokens": 4000,
    }
}

# Default model configuration - prefer OpenRouter if available
DEFAULT_MODEL = AIModel.OPENROUTER_CLAUDE if settings.OPENROUTER_API_KEY else (
    AIModel.OPENAI if settings.OPENAI_API_KEY else AIModel.ANTHROPIC if settings.ANTHROPIC_API_KEY else None
)

if not DEFAULT_MODEL:
    print("Warning: No API keys configured for any AI model")

async def query_ai_model(prompt: str, context: Dict[str, Any], model: str = None) -> Dict[str, Any]:
    """Query AI model with prompt and context"""
    if not model:
        model = DEFAULT_MODEL
        if not model:
            logger.error("No AI model API keys configured")
            raise ValueError("No AI model API keys configured")

    if model not in MODEL_CONFIGS:
        logger.error(f"Unsupported model: {model}")
        raise ValueError(f"Unsupported model: {model}")

    logger.info(f"Querying {model} for analysis")
    config = MODEL_CONFIGS[model]
    
    # Handle API key validation based on model type
    api_key = None
    if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
        api_key = settings.OPENROUTER_API_KEY
    else:
        api_key = getattr(settings, f"{model.upper()}_API_KEY")
    
    if not api_key:
        logger.error(f"{model} API key not configured")
        raise ValueError(f"{model} API key not configured")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Model-specific header configuration
    if model == AIModel.OPENAI:
        headers["Authorization"] = f"Bearer {api_key}"
    elif model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
        headers["Authorization"] = f"Bearer {api_key}"
        headers.update(config.get("headers", {}))
    elif model == AIModel.ANTHROPIC:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2024-01-01"

    logger.info(f"Preparing request for {model}")
    system_message = """You are an AI investment research assistant. Analyze the provided portfolio and market data to give informed insights and recommendations. Base your analysis on:
1. Portfolio composition and performance
2. Market conditions and trends
3. Risk factors and opportunities
4. Latest news and analyst opinions
Always provide specific, data-backed insights and clear reasoning for your recommendations."""

    try:
        async with httpx.AsyncClient(timeout=settings.MODEL_TIMEOUT) as client:
            # Prepare request based on model
            if model in [AIModel.OPENAI, AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context: {json.dumps(context, indent=2)}\n\nQuery: {prompt}"}
                ]
                request_data = {
                    "model": config["model"],
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": config["max_tokens"],
                }
            elif model == AIModel.ANTHROPIC:
                request_data = {
                    "model": config["model"],
                    "messages": [{
                        "role": "user",
                        "content": f"{system_message}\n\nContext: {json.dumps(context, indent=2)}\n\nQuery: {prompt}"
                    }],
                    "max_tokens": config["max_tokens"]
                }

            logger.info(f"Sending request to {model}")
            response = await client.post(
                config["api_url"],
                headers=headers,
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received successful response from {model}")
                
                # Extract content based on model
                if model in [AIModel.OPENAI, AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
                    content = result["choices"][0]["message"]["content"]
                elif model == AIModel.ANTHROPIC:
                    content = result["messages"][0]["content"][0]["text"]
                
                logger.info(f"Successfully extracted content from {model} response")
                return {
                    "answer": content,
                    "sources": context,
                    "model_used": config["model"]
                }
            else:
                error_message = response.text
                logger.error(f"{model} API error: {error_message}")
                raise ValueError(f"{model} API error: {error_message}")
                
    except Exception as e:
        logger.error(f"Error querying {model}: {str(e)}", exc_info=True)
        raise ValueError(f"Error querying {model}: {str(e)}")

async def stream_ai_response(prompt: str, context: Dict[str, Any], model: str = None) -> AsyncGenerator[str, None]:
    """Stream AI model response"""
    try:
        if not model:
            model = DEFAULT_MODEL
            if not model:
                raise ValueError("No AI model API keys configured")

        config = MODEL_CONFIGS[model]
        
        # Handle API key validation
        api_key = None
        if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
            api_key = settings.OPENROUTER_API_KEY
        else:
            api_key = getattr(settings, f"{model.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"{model} API key not configured")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
            headers.update(config.get("headers", {}))
        elif model == AIModel.ANTHROPIC:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2024-01-01"

        system_message = """You are an AI investment research assistant. Analyze the provided portfolio and market data to give informed insights and recommendations. Base your analysis on:
1. Portfolio composition and performance
2. Market conditions and trends
3. Risk factors and opportunities
4. Latest news and analyst opinions
Always provide specific, data-backed insights and clear reasoning for your recommendations."""

        request_data = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context: {json.dumps(context, indent=2)}\n\nQuery: {prompt}"}
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": config["max_tokens"],
        }

        async with httpx.AsyncClient(timeout=settings.MODEL_TIMEOUT) as client:
            async with client.stream(
                "POST",
                config["api_url"],
                headers=headers,
                json=request_data,
                timeout=settings.MODEL_TIMEOUT
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.aiter_bytes():
                    chunk_str = chunk.decode()
                    
                    # Handle SSE format
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]
                    
                    try:
                        data = json.loads(chunk_str)
                        content = None
                        
                        if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
                            if "choices" in data and data["choices"]:
                                content = data["choices"][0].get("delta", {}).get("content", "")
                        elif model == AIModel.OPENAI:
                            if "choices" in data and data["choices"]:
                                content = data["choices"][0].get("delta", {}).get("content", "")
                        elif model == AIModel.ANTHROPIC:
                            if "type" in data and data["type"] == "content_block_delta":
                                content = data.get("delta", {}).get("text", "")
                        
                        if content:
                            yield json.dumps({
                                "answer": content,
                                "model_used": config["model"]
                            }) + "\n"
                            
                    except json.JSONDecodeError:
                        buffer += chunk_str
                        if "\n" in buffer:
                            lines = buffer.split("\n")
                            buffer = lines[-1]
                            
                            for line in lines[:-1]:
                                if line.strip():
                                    try:
                                        if line.startswith("data: "):
                                            line = line[6:]
                                        data = json.loads(line)
                                        content = None
                                        if "choices" in data and data["choices"]:
                                            content = data["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            yield json.dumps({
                                                "answer": content,
                                                "model_used": config["model"]
                                            }) + "\n"
                                    except json.JSONDecodeError:
                                        continue

    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"
