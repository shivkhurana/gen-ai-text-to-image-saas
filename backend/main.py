from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model import ImageGenerator

app = FastAPI(title="Text-to-Image API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML model (loads into memory once on startup)
generator = ImageGenerator()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
async def generate(request: PromptRequest):
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
            
        base64_image = generator.generate_image_base64(request.prompt)
        return {"status": "success", "image_data": base64_image}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))