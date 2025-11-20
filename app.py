from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import sys
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from pydantic import BaseModel
from textSummarizer.pipeline.prediction import PredictionPipeline


# Request model for the API
class TextInput(BaseModel):
    text: str

app = FastAPI(title="AI Text Summarizer", description="Advanced NLP-powered text summarization API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the modern frontend interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs-redirect", tags=["documentation"])
async def docs_redirect():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    """Train the model (unchanged logic)"""
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(input_data: TextInput):
    """Generate text summary (enhanced with proper request handling)"""
    try:
        obj = PredictionPipeline()
        summary = obj.predict(input_data.text)
        return {"input_text": input_data.text, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
    

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
