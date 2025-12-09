"""
Serveur d'inférence CSM Streaming - API uniquement
FastAPI server pour génération audio sans chat/RAG
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
os.environ["PYTORCH_DISABLE_CUDA_GRAPHS"] = "1"

import asyncio
import logging
import tempfile
import time
import io
import base64
import json
import numpy as np
from typing import Optional, List
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generator import load_csm_1b, generate_streaming_audio, Segment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CSM Streaming Inference API",
    description="API pour génération audio avec CSM (Conversational Speech Model)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
generator = None

# Pydantic models
class TextToSpeechRequest(BaseModel):
    text: str
    speaker: int = 0
    output_format: str = "wav"  # wav, base64
    context_text: Optional[str] = None
    context_audio_base64: Optional[str] = None

class TextToSpeechResponse(BaseModel):
    success: bool
    message: str
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    gpu_name: Optional[str] = None
    memory_free_mb: Optional[int] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Charger le modèle CSM au démarrage"""
    global generator
    logger.info("Démarrage du serveur d'inférence CSM...")
    
    try:
        logger.info("Chargement du modèle CSM-1B...")
        generator = load_csm_1b("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Modèle CSM chargé avec succès !")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        generator = None

# Routes
@app.get("/", response_class=JSONResponse)
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "CSM Streaming Inference API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "stream": "/stream",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérification de santé du service"""
    gpu_available = torch.cuda.is_available()
    model_loaded = generator is not None
    gpu_name = None
    memory_free_mb = None
    
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            memory_free_mb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        except:
            pass
    
    status = "healthy" if (gpu_available and model_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        gpu_name=gpu_name,
        memory_free_mb=memory_free_mb
    )

@app.post("/generate", response_model=TextToSpeechResponse)
async def generate_audio(request: TextToSpeechRequest):
    """Générer de l'audio à partir de texte"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Modèle CSM non disponible")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
    
    start_time = time.time()
    
    try:
        logger.info(f"Génération audio pour: '{request.text[:50]}...'")
        
        # Préparer le contexte si fourni
        context = []
        if request.context_text and request.context_audio_base64:
            try:
                # Décoder l'audio de base64
                audio_bytes = base64.b64decode(request.context_audio_base64)
                audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
                
                # Resample si nécessaire
                if sr != generator.sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor.squeeze(0), sr, generator.sample_rate
                    )
                
                context = [Segment(
                    text=request.context_text,
                    speaker=request.speaker,
                    audio=audio_tensor
                )]
                logger.info("Contexte audio ajouté")
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du contexte: {e}")
        
        # Générer l'audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        generate_streaming_audio(
            generator=generator,
            text=request.text,
            speaker=request.speaker,
            context=context,
            output_file=temp_path,
            play_audio=False
        )
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Audio généré en {duration_ms:.0f}ms")
        
        # Retourner selon le format demandé
        if request.output_format == "base64":
            # Encoder en base64
            with open(temp_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode()
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_path)
            
            return TextToSpeechResponse(
                success=True,
                message="Audio généré avec succès",
                audio_base64=audio_base64,
                duration_ms=duration_ms
            )
        else:
            # Retourner l'URL du fichier
            return TextToSpeechResponse(
                success=True,
                message="Audio généré avec succès",
                audio_url=f"/audio/{Path(temp_path).name}",
                duration_ms=duration_ms
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.websocket("/stream")
async def websocket_stream_audio(websocket: WebSocket):
    """Streamer l'audio en chunks via WebSocket"""
    await websocket.accept()
    logger.info("Nouvelle connexion WebSocket pour streaming audio")
    
    try:
        # Recevoir la requête via WebSocket
        request_data = await websocket.receive_text()
        request_json = json.loads(request_data)
        
        text = request_json.get("text", "")
        speaker = request_json.get("speaker", 0)
        context_text = request_json.get("context_text")
        context_audio_base64 = request_json.get("context_audio_base64")
        
        if not text.strip():
            await websocket.send_text(json.dumps({"error": "Texte vide"}))
            return
            
        if generator is None:
            await websocket.send_text(json.dumps({"error": "Modèle CSM non disponible"}))
            return
        
        logger.info(f"Streaming audio pour: '{text[:50]}...'")
        
        # Préparer le contexte si fourni
        context = []
        if context_text and context_audio_base64:
            try:
                # Décoder l'audio de base64
                audio_bytes = base64.b64decode(context_audio_base64)
                audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
                
                # Resample si nécessaire
                if sr != generator.sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor.squeeze(0), sr, generator.sample_rate
                    )
                
                context = [Segment(
                    text=context_text,
                    speaker=speaker,
                    audio=audio_tensor
                )]
                logger.info("Contexte audio ajouté pour streaming")
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du contexte: {e}")
        
        # Fonction callback pour envoyer chaque chunk
        chunk_count = 0
        start_time = time.time()
        
        async def send_chunk(chunk):
            nonlocal chunk_count
            try:
                # Convertir le chunk en numpy puis en bytes
                chunk_np = chunk.cpu().numpy()
                chunk_bytes = (chunk_np * 32767).astype(np.int16).tobytes()
                chunk_b64 = base64.b64encode(chunk_bytes).decode()
                
                chunk_data = {
                    "type": "audio_chunk",
                    "chunk_id": chunk_count,
                    "data": chunk_b64,
                    "sample_rate": generator.sample_rate,
                    "duration_ms": len(chunk) / generator.sample_rate * 1000
                }
                
                await websocket.send_text(json.dumps(chunk_data))
                chunk_count += 1
                
                # Log de progression tous les 5 chunks
                if chunk_count % 5 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Chunk {chunk_count} envoyé après {elapsed:.2f}s")
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'envoi du chunk: {e}")
        
        # Générer et streamer l'audio
        try:
            for chunk in generator.generate_stream(
                text=text,
                speaker=speaker,
                context=context,
                max_audio_length_ms=90_000,
                temperature=0.7,
                topk=30
            ):
                await send_chunk(chunk)
            
            # Signal de fin
            total_time = time.time() - start_time
            end_data = {
                "type": "end",
                "total_chunks": chunk_count,
                "total_duration_ms": total_time * 1000,
                "message": "Streaming terminé avec succès"
            }
            await websocket.send_text(json.dumps(end_data))
            logger.info(f"Streaming terminé - {chunk_count} chunks envoyés en {total_time:.2f}s")
            
        except Exception as e:
            error_data = {
                "type": "error",
                "message": f"Erreur lors de la génération: {str(e)}"
            }
            await websocket.send_text(json.dumps(error_data))
            logger.error(f"Erreur lors du streaming: {e}")
        
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnecté")
    except json.JSONDecodeError:
        try:
            await websocket.send_text(json.dumps({"error": "Format JSON invalide"}))
        except:
            pass
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Servir les fichiers audio générés"""
    file_path = f"/tmp/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier audio introuvable")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Gestion d'erreur globale
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Erreur interne du serveur", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)