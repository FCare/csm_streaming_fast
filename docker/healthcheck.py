#!/usr/bin/env python3
"""
Script de vérification de santé pour le conteneur CSM Streaming
Vérifie que l'application répond correctement sur le port 8000
"""

import sys
import requests
import time

def check_health():
    """Vérifie la santé de l'application"""
    try:
        # Tenter de se connecter à l'application
        response = requests.get('http://localhost:8000/', timeout=5)
        
        if response.status_code == 200:
            print("✅ Application CSM Streaming opérationnelle")
            return True
        else:
            print(f"❌ Application répond mais code d'erreur: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'application sur le port 8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout lors de la connexion à l'application")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)  # Succès
    else:
        sys.exit(1)  # Échec