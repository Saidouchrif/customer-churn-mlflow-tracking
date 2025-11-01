"""
API FastAPI pour la prédiction du churn bancaire
Auteur: Système de ML
Date: 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import uvicorn

# Configuration de l'application
app = FastAPI(
    title="API de Prédiction du Churn Bancaire",
    description="API pour prédire le risque de churn des clients bancaires",
    version="1.0.0"
)

# Chemins des fichiers
MODEL_PATH = Path("../models/best_model.pkl")
METADATA_PATH = Path("../models/best_model_metadata.json")

# Variables globales pour le modèle
model = None
model_metadata = None

# Modèles Pydantic pour la validation des données
class CustomerData(BaseModel):
    """Modèle de données pour un client"""
    CreditScore: int = Field(..., ge=300, le=850, description="Score de crédit (300-850)")
    Geography: str = Field(..., description="Pays (France, Spain, Germany)")
    Gender: str = Field(..., description="Genre (Male, Female)")
    Age: int = Field(..., ge=18, le=100, description="Âge du client (18-100)")
    Tenure: int = Field(..., ge=0, le=10, description="Ancienneté en années (0-10)")
    Balance: float = Field(..., ge=0, description="Solde du compte")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Nombre de produits (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Possède une carte de crédit (0/1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Membre actif (0/1)")
    EstimatedSalary: float = Field(..., ge=0, description="Salaire estimé")

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 3,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0
            }
        }

class PredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions"""
    prediction: int = Field(..., description="Prédiction (0=Non-Churn, 1=Churn)")
    probability_no_churn: float = Field(..., description="Probabilité de non-churn")
    probability_churn: float = Field(..., description="Probabilité de churn")
    confidence_level: str = Field(..., description="Niveau de confiance")
    risk_category: str = Field(..., description="Catégorie de risque")

class BatchPredictionRequest(BaseModel):
    """Modèle pour les prédictions en lot"""
    customers: List[CustomerData] = Field(..., description="Liste des clients à prédire")

# Fonctions utilitaires
def load_model():
    """Charge le modèle et ses métadonnées"""
    global model, model_metadata
    
    try:
        # Chargement du modèle
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"Modele charge depuis: {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Modele non trouve: {MODEL_PATH}")
        
        # Chargement des métadonnées
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            print(f"Metadonnees chargees depuis: {METADATA_PATH}")
        else:
            print("Metadonnees non trouvees, utilisation des valeurs par defaut")
            model_metadata = {
                "model_name": "Unknown",
                "strategy": "Unknown",
                "f1_score": 0.0,
                "accuracy": 0.0,
                "roc_auc": 0.0
            }
            
    except Exception as e:
        print(f"Erreur lors du chargement du modele: {str(e)}")
        raise e

def get_confidence_level(probability: float) -> str:
    """Détermine le niveau de confiance basé sur la probabilité"""
    if probability > 0.8:
        return "Très confiant"
    elif probability > 0.6:
        return "Modérément confiant"
    else:
        return "Peu confiant"

def get_risk_category(churn_probability: float) -> str:
    """Détermine la catégorie de risque"""
    if churn_probability > 0.7:
        return "Risque élevé"
    elif churn_probability > 0.4:
        return "Risque modéré"
    else:
        return "Risque faible"

def prepare_data(customer_data: CustomerData) -> pd.DataFrame:
    """Prépare les données pour la prédiction"""
    # Conversion en DataFrame
    data_dict = customer_data.dict()
    df = pd.DataFrame([data_dict])
    
    return df

# Événements de démarrage
@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    print("Demarrage de l'API de prediction du churn...")
    load_model()
    print("API prete a recevoir des requetes!")

# Routes de l'API

@app.get("/", tags=["Santé"])
async def root():
    """
    Route de vérification du bon fonctionnement de l'API
    """
    return {
        "message": "API de Prediction du Churn Bancaire",
        "status": "Fonctionnelle",
        "version": "1.0.0",
        "description": "API pour predire le risque de churn des clients bancaires",
        "endpoints": {
            "/": "Vérification du statut",
            "/health": "Santé détaillée de l'API",
            "/predict": "Prédiction pour un client",
            "/predict/batch": "Prédictions en lot",
            "/model/info": "Informations sur le modèle"
        }
    }

@app.get("/health", tags=["Santé"])
async def health_check():
    """
    Vérification détaillée de la santé de l'API
    """
    model_loaded = model is not None
    metadata_loaded = model_metadata is not None
    
    return {
        "api_status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "metadata_loaded": metadata_loaded,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "metadata_path": str(METADATA_PATH),
        "metadata_exists": METADATA_PATH.exists(),
        "model_info": model_metadata if metadata_loaded else None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_churn(customer: CustomerData):
    """
    Prédit le risque de churn pour un client
    
    - **customer**: Données du client à analyser
    - **return**: Prédiction avec probabilités et niveau de confiance
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Préparation des données
        df = prepare_data(customer)
        
        # Prédiction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        prob_no_churn = float(probabilities[0])
        prob_churn = float(probabilities[1])
        
        # Calcul du niveau de confiance
        max_prob = max(prob_no_churn, prob_churn)
        confidence_level = get_confidence_level(max_prob)
        
        # Catégorie de risque
        risk_category = get_risk_category(prob_churn)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability_no_churn=prob_no_churn,
            probability_churn=prob_churn,
            confidence_level=confidence_level,
            risk_category=risk_category
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict/batch", tags=["Prédiction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Prédit le risque de churn pour plusieurs clients
    
    - **request**: Liste des clients à analyser
    - **return**: Liste des prédictions
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        results = []
        
        for i, customer in enumerate(request.customers):
            # Préparation des données
            df = prepare_data(customer)
            
            # Prédiction
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            
            prob_no_churn = float(probabilities[0])
            prob_churn = float(probabilities[1])
            
            # Calcul du niveau de confiance
            max_prob = max(prob_no_churn, prob_churn)
            confidence_level = get_confidence_level(max_prob)
            
            # Catégorie de risque
            risk_category = get_risk_category(prob_churn)
            
            results.append({
                "customer_index": i,
                "prediction": int(prediction),
                "probability_no_churn": prob_no_churn,
                "probability_churn": prob_churn,
                "confidence_level": confidence_level,
                "risk_category": risk_category
            })
        
        return {
            "total_customers": len(request.customers),
            "predictions": results,
            "summary": {
                "high_risk_count": sum(1 for r in results if r["risk_category"] == "Risque élevé"),
                "moderate_risk_count": sum(1 for r in results if r["risk_category"] == "Risque modéré"),
                "low_risk_count": sum(1 for r in results if r["risk_category"] == "Risque faible")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors des prédictions en lot: {str(e)}")

@app.get("/model/info", tags=["Modèle"])
async def get_model_info():
    """
    Retourne les informations sur le modèle chargé
    """
    if model_metadata is None:
        raise HTTPException(status_code=500, detail="Métadonnées du modèle non disponibles")
    
    return {
        "model_metadata": model_metadata,
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None,
        "features_expected": model_metadata.get("features", []) if model_metadata else []
    }

@app.get("/test/sample", tags=["Test"])
async def test_with_sample():
    """
    Test l'API avec des données d'exemple
    """
    sample_customer = CustomerData(
        CreditScore=650,
        Geography="France",
        Gender="Male",
        Age=35,
        Tenure=3,
        Balance=50000.0,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=75000.0
    )
    
    # Utilise la fonction de prédiction
    result = await predict_churn(sample_customer)
    
    return {
        "message": "Test avec données d'exemple",
        "sample_data": sample_customer.dict(),
        "prediction_result": result.dict()
    }

# Point d'entrée pour lancer l'API
if __name__ == "__main__":
    print("Lancement de l'API de prediction du churn...")
    uvicorn.run(
        "churn_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )