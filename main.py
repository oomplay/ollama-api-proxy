import os
import uvicorn
import requests
import time
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
from pydantic import BaseModel, Field
from typing import List, Literal

# โหลดข้อมูลจาก .env
load_dotenv()

OLLAMA_CHAT_ENDPOINT =  os.getenv("OLLAMA_ENDPOINT",http://127.0.0.1:11434/api/chat)
SERVER_API_KEY = os.getenv("API")

# ตรวจสอบว่า API Key ถูกโหลดจาก .env หรือไม่
if not SERVER_API_KEY:
    raise ValueError("API key is missing. Please check your .env file.")

# แบบจำลองสำหรับข้อความแชท
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

# แบบจำลองสำหรับคำขอการสร้างแชท
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

# แบบจำลองสำหรับข้อความตอบกลับ
class ResponseChatMessage(BaseModel):
    role: Literal["assistant"]
    content: str

# แบบจำลองสำหรับตัวเลือกการตอบกลับแชท
class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseChatMessage
    finish_reason: str = "stop"

# แบบจำลองสำหรับการตอบกลับแชท
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]

# ระบบตรวจสอบ Bearer Token
bearer_scheme = HTTPBearer()

def check_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Check Bearer Token from Request"""
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Authorization scheme must be Bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != SERVER_API_KEY:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# การตั้งค่า FastAPI
app = FastAPI(
    title="Professional AI Chat API",
    description="An OpenAI-compatible API layer for Ollama",
    version="2.0.1"
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
    รับคำขอ Chat Completion และส่งต่อไปยัง Ollama
    """
    print(f"✅ [Authorized] Received request for model: {request.model}")

    ollama_payload = {
        "model": request.model,
        "messages": [msg.dict() for msg in request.messages],
        "stream": False
    }

    try:
        # ส่งคำขอไปยัง Ollama API
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=ollama_payload, timeout=30)  # ลดเวลา timeout
        response.raise_for_status()  # ตรวจสอบสถานะของการตอบกลับ

        # ตรวจสอบข้อมูลที่ได้จาก Ollama
        ollama_data = response.json()
        assistant_message_content = ollama_data.get("message", {}).get("content", "")
        actual_model_used = ollama_data.get("model", request.model)

        # สร้างผลลัพธ์การตอบกลับ
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

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to Ollama API timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to Ollama: {e}")
    except ValueError:
        raise HTTPException(status_code=500, detail="Error parsing response from Ollama API")

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🚀 Starting API Server...")
        print(f"🔑 Required API Key for 'Authorization: Bearer <key>': {SERVER_API_KEY}")
        print("=" * 60)
        
        # เริ่มต้นเซิร์ฟเวอร์ด้วย uvicorn
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True
        )
    except Exception as e:
        print(f"❌ Error starting the server: {e}")
