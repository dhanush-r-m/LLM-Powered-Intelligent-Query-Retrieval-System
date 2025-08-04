from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import os
import tempfile
import aiohttp
import asyncio

from intelligent_query_system import IntelligentQuerySystem
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

query_system = IntelligentQuerySystem(groq_api_key=groq_api_key, cache_dir="./query_system_cache")

app = FastAPI()

class HackRxInput(BaseModel):
    documents: str  # URL to a PDF document
    questions: List[str]

@app.post("/hackrx/run")
async def handle_hackrx(input_data: HackRxInput, request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    # Download document to temp file
    document_url = input_data.documents
    temp_file_path = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(document_url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download document.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(await resp.read())
                    temp_file_path = temp_file.name

        await query_system.build_knowledge_base([temp_file_path], save_index=False)

        responses = await query_system.batch_query(input_data.questions)
        answers = [res.answer["response"] for res in responses]

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)