from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ajouter le répertoire racine à sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.routes.chatbot_routes import chatbot_router

app = FastAPI()

# CORS pour React (localhost:3000 par ex.)
app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router, prefix="/chatbot")
