#!/bin/bash

# Script de démarrage rapide pour CSM Streaming Docker
set -e

echo "🚀 CSM Streaming - Démarrage rapide Docker"
echo "=========================================="

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

# Vérifier Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi

# Vérifier le support GPU
echo "🔍 Vérification du support GPU..."
if docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ Support GPU NVIDIA détecté"
else
    echo "⚠️  Aucun GPU NVIDIA détecté ou nvidia-docker2 non installé"
    echo "   L'application peut ne pas fonctionner correctement"
fi

# Vérifier si .env existe
if [ ! -f .env ]; then
    echo "📝 Création du fichier .env..."
    cp .env.example .env
    echo "⚠️  Pensez à éditer .env avec votre token Hugging Face :"
    echo "   HF_TOKEN=votre_token_ici"
    echo ""
    read -p "Appuyez sur Entrée pour continuer..."
fi

# Créer les dossiers nécessaires
echo "📁 Création des dossiers..."
mkdir -p config

# Construire et démarrer
echo "🏗️  Construction de l'image Docker..."
docker-compose build

echo "🚀 Démarrage des services..."
docker-compose up -d

echo ""
echo "✅ CSM Streaming est en cours de démarrage !"
echo ""
echo "📋 Informations utiles :"
echo "   • Interface web : http://localhost:8000"
echo "   • Configuration : http://localhost:8000/setup" 
echo "   • Logs : docker-compose logs -f csm-streaming"
echo "   • Arrêt : docker-compose down"
echo ""
echo "⏳ Patientez quelques minutes pour le téléchargement des modèles..."

# Afficher les logs pendant 30 secondes
echo "📊 Logs de démarrage (30s) :"
timeout 30 docker-compose logs -f csm-streaming || true

echo ""
echo "🎉 Installation terminée ! Rendez-vous sur http://localhost:8000"