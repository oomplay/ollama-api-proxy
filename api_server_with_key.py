import os
import uvicorn
import requests
import time
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
from pydantic import BaseModel
from typing import List, Literal
load_dotenv()

OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"

SERVER_API_KEY = os.getenv("API")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

class ResponseChatMessage(BaseModel):
    role: Literal["assistant"]
    content: str

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = f"chatcmpl-{uuid.uuid4()}"
    object: str = "chat.completion"
    created: int = int(time.time())
    model: str
    choices: List[ChatCompletionChoice]

bearer_scheme = HTTPBearer()

def check_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Check Bearer Token from Request"""
    if credentials.scheme != "Bearer" or credentials.credentials != SERVER_API_KEY:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

app = FastAPI(
    title="Professional AI Chat API",
    description="An OpenAI-compatible API layer for Ollama",
    version="2.0.0"
)

@app.post(
    "/chat/completions",
    response_model=ChatCompletionResponse, 
    tags=["Chat Completions"]
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(check_api_key)
):
    """
    Receive Chat Completion requests, forward them to Ollama, and standardize the results.
    """
    print(f"✅ [Authorized] Received request for model: {request.model}")
    

    ollama_payload = {
        "model": request.model,
        "messages": [msg.dict() for msg in request.messages],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=ollama_payload, timeout=300)
        response.raise_for_status()
        ollama_data = response.json()
        
        assistant_message_content = ollama_data.get("message", {}).get("content", "")
        actual_model_used = ollama_data.get("model", request.model)

        final_response = ChatCompletionResponse(
            model=actual_model_used,
            choices=[
                ChatCompletionChoice(
                    message=ResponseChatMessage(
                        role="assistant",
                        content=assistant_message_content
                    )
                )
            ]
        )
        return final_response

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to Ollama: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Professional-Grade API Server...")
    print(f"🔑 Required API Key for 'Authorization: Bearer <key>': {SERVER_API_KEY}")
    print("=" * 60)
    
    uvicorn.run(
        "api_server_with_key:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
