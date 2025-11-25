import os
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Remote Diarization Service")

# Global variable to hold the pipeline
pipeline = None
loaded_token = None

class DiarizationSegment(BaseModel):
    start: float
    end: float
    speaker: float  # Using speaker label/id

@app.post("/diarize")
async def diarize_audio(
    file: UploadFile = File(...),
    hf_token: str = Form(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    global pipeline, loaded_token

    # 1. Load Pipeline if needed (Singleton pattern per token)
    if pipeline is None or loaded_token != hf_token:
        print(f"Loading Pyannote pipeline with new token...")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Send to GPU if available
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                print("Pipeline sent to CUDA (GPU).")
            else:
                print("CUDA not available. Running on CPU.")
                
            loaded_token = hf_token
        except Exception as e:
            return JSONResponse(
                status_code=500, 
                content={"error": f"Failed to load pipeline. Check HF Token. Error: {str(e)}"}
            )

    # 2. Save uploaded file temporarily
    temp_filename = f"/tmp/uploads/{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to save file: {str(e)}"})

    # 3. Run Diarization
    try:
        # Construct parameters
        params = {}
        if min_speakers: params["min_speakers"] = min_speakers
        if max_speakers: params["max_speakers"] = max_speakers

        # Run inference
        diarization = pipeline(temp_filename, **params)

        # 4. Format Output
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker
            })

        # Cleanup
        os.remove(temp_filename)

        return JSONResponse(content={"segments": segments})

    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return JSONResponse(status_code=500, content={"error": f"Diarization failed: {str(e)}"})

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}

