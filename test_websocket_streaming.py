#!/usr/bin/env python3
"""
Script de test pour le streaming audio via WebSocket
"""
import asyncio
import json
import base64
import wave
import numpy as np
import websockets

async def test_websocket_streaming():
    """Test du streaming audio via WebSocket"""
    uri = "ws://localhost:8000/stream"
    
    print("🚀 Connexion au WebSocket streaming...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connecté au serveur WebSocket")
            
            # Préparer la requête
            request_data = {
                "text": "Hi, this is a relatime test for cesame text to speech streming engine.",
                "speaker": 0
            }
            
            print(f"📝 Envoi de la requête: '{request_data['text']}'")
            await websocket.send(json.dumps(request_data))
            
            # Collecter les chunks audio
            audio_chunks = []
            chunk_count = 0
            sample_rate = None
            
            print("🎵 Réception des chunks audio...")
            
            while True:
                try:
                    # Recevoir le message
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "audio_chunk":
                        chunk_count += 1
                        
                        # Décoder le chunk audio
                        chunk_bytes = base64.b64decode(data["data"])
                        chunk_audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                        audio_chunks.append(chunk_audio)
                        
                        # Stocker le sample rate
                        if sample_rate is None:
                            sample_rate = data["sample_rate"]
                        
                        duration_ms = data.get("duration_ms", 0)
                        print(f"  📦 Chunk {chunk_count}: {len(chunk_audio)} samples ({duration_ms:.1f}ms)")
                        
                    elif data.get("type") == "end":
                        total_chunks = data.get("total_chunks", chunk_count)
                        total_duration = data.get("total_duration_ms", 0)
                        print(f"✅ Streaming terminé!")
                        print(f"   Total chunks reçus: {total_chunks}")
                        print(f"   Durée totale: {total_duration:.0f}ms")
                        break
                        
                    elif data.get("type") == "error":
                        print(f"❌ Erreur: {data.get('message', 'Erreur inconnue')}")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Connexion fermée par le serveur")
                    break
                except json.JSONDecodeError as e:
                    print(f"❌ Erreur de décodage JSON: {e}")
                    break
            
            # Sauvegarder l'audio complet
            if audio_chunks and sample_rate:
                print("💾 Sauvegarde de l'audio...")
                
                # Concaténer tous les chunks
                full_audio = np.concatenate(audio_chunks)
                
                # Sauvegarder en WAV
                output_file = "websocket_streaming_output.wav"
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(full_audio.tobytes())
                
                duration_sec = len(full_audio) / sample_rate
                print(f"✅ Audio sauvegardé: {output_file}")
                print(f"   Durée: {duration_sec:.2f}s")
                print(f"   Échantillons: {len(full_audio)}")
                print(f"   Sample rate: {sample_rate}Hz")
            else:
                print("❌ Aucun chunk audio reçu")
    
    except websockets.exceptions.ConnectionRefused:
        print("❌ Impossible de se connecter au serveur WebSocket")
        print("   Assurez-vous que le serveur CSM est démarré sur localhost:8000")
    except Exception as e:
        print(f"❌ Erreur: {e}")

def test_with_context():
    """Test avec contexte audio de référence"""
    print("\n" + "="*50)
    print("TEST AVEC CONTEXTE AUDIO")
    print("="*50)
    
    # Pour tester avec un contexte, vous devez avoir un fichier audio de référence
    # Décommentez et adaptez le code ci-dessous si vous avez un fichier de référence
    
    """
    import torchaudio
    
    # Charger l'audio de référence
    ref_audio_path = "reference.wav"  # Remplacez par votre fichier
    if os.path.exists(ref_audio_path):
        audio_tensor, sr = torchaudio.load(ref_audio_path)
        
        # Convertir en base64
        audio_np = audio_tensor.numpy()
        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        request_data = {
            "text": "Test avec contexte audio de référence",
            "speaker": 0,
            "context_text": "Texte correspondant à l'audio de référence",
            "context_audio_base64": audio_b64
        }
        
        return asyncio.run(test_websocket_with_context(request_data))
    """
    
    print("ℹ️  Pour tester avec contexte, ajoutez un fichier audio de référence")
    print("   et décommentez la section correspondante dans le code")

async def test_websocket_with_context(request_data):
    """Test WebSocket avec contexte"""
    uri = "ws://localhost:8000/stream"
    
    async with websockets.connect(uri) as websocket:
        print("📤 Envoi de la requête avec contexte...")
        await websocket.send(json.dumps(request_data))
        
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "audio_chunk":
                print(f"  📦 Chunk reçu: {data.get('duration_ms', 0):.1f}ms")
            elif data.get("type") == "end":
                print("✅ Streaming avec contexte terminé!")
                break
            elif data.get("type") == "error":
                print(f"❌ Erreur: {data.get('message')}")
                break

def main():
    """Fonction principale"""
    print("🎵 Test du streaming audio WebSocket CSM")
    print("="*50)
    
    # Test de base
    asyncio.run(test_websocket_streaming())
    
    # Test avec contexte (optionnel)
    test_with_context()
    
    print("\n✨ Tests terminés!")
    print("📁 Fichier généré: websocket_streaming_output.wav")

if __name__ == "__main__":
    main()