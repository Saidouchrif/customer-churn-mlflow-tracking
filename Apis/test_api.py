"""
Script de test pour l'API de prédiction du churn
"""

import requests
import json

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def test_root_endpoint():
    """Test de la route racine"""
    print("🧪 Test de la route GET /")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ Route / fonctionne!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
    print("-" * 50)

def test_health_endpoint():
    """Test de la route de santé"""
    print("🧪 Test de la route GET /health")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Route /health fonctionne!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
    print("-" * 50)

def test_prediction_endpoint():
    """Test de la route de prédiction"""
    print("🧪 Test de la route POST /predict")
    
    # Données d'exemple
    customer_data = {
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
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("✅ Route /predict fonctionne!")
            print("📊 Données envoyées:")
            print(json.dumps(customer_data, indent=2, ensure_ascii=False))
            print("📈 Résultat de la prédiction:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
    print("-" * 50)

def test_sample_endpoint():
    """Test de la route d'exemple"""
    print("🧪 Test de la route GET /test/sample")
    try:
        response = requests.get(f"{API_BASE_URL}/test/sample")
        if response.status_code == 200:
            print("✅ Route /test/sample fonctionne!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
    print("-" * 50)

def test_model_info_endpoint():
    """Test de la route d'informations du modèle"""
    print("🧪 Test de la route GET /model/info")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            print("✅ Route /model/info fonctionne!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
    print("-" * 50)

def main():
    """Fonction principale de test"""
    print("🚀 TESTS DE L'API DE PRÉDICTION DU CHURN")
    print("=" * 60)
    
    # Tests des différentes routes
    test_root_endpoint()
    test_health_endpoint()
    test_model_info_endpoint()
    test_prediction_endpoint()
    test_sample_endpoint()
    
    print("🎉 Tests terminés!")

if __name__ == "__main__":
    main()
