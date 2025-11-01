"""
Script de test pour l'API Docker de prédiction du churn
"""

import requests
import time
import json
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_RETRIES = 30
RETRY_DELAY = 2

def wait_for_api():
    """Attend que l'API soit disponible"""
    print("🔄 Attente du démarrage de l'API...")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API disponible!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"   Tentative {attempt + 1}/{MAX_RETRIES}...")
        time.sleep(RETRY_DELAY)
    
    print("❌ Timeout: L'API n'est pas disponible")
    return False

def test_health():
    """Test de santé de l'API"""
    print("\n🧪 Test de santé...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Santé OK")
            print(f"   Modèle chargé: {data.get('model_loaded', False)}")
            print(f"   Statut API: {data.get('api_status', 'unknown')}")
            return True
        else:
            print(f"❌ Erreur santé: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur santé: {e}")
        return False

def test_root():
    """Test de la route racine"""
    print("\n🧪 Test de la route racine...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Route racine OK")
            print(f"   Message: {data.get('message', '')}")
            print(f"   Version: {data.get('version', '')}")
            return True
        else:
            print(f"❌ Erreur route racine: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur route racine: {e}")
        return False

def test_prediction():
    """Test de prédiction"""
    print("\n🧪 Test de prédiction...")
    
    # Données de test
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
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prédiction OK")
            print(f"   Prédiction: {data.get('prediction', 'N/A')}")
            print(f"   Probabilité churn: {data.get('probability_churn', 0):.3f}")
            print(f"   Catégorie risque: {data.get('risk_category', 'N/A')}")
            print(f"   Confiance: {data.get('confidence_level', 'N/A')}")
            return True
        else:
            print(f"❌ Erreur prédiction: {response.status_code}")
            print(f"   Réponse: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        return False

def test_model_info():
    """Test des informations du modèle"""
    print("\n🧪 Test des informations du modèle...")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Informations modèle OK")
            metadata = data.get('model_metadata', {})
            print(f"   Nom du modèle: {metadata.get('model_name', 'N/A')}")
            print(f"   Stratégie: {metadata.get('strategy', 'N/A')}")
            print(f"   F1-Score: {metadata.get('f1_score', 0):.3f}")
            print(f"   Accuracy: {metadata.get('accuracy', 0):.3f}")
            return True
        else:
            print(f"❌ Erreur info modèle: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur info modèle: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🐳 TESTS DE L'API DOCKER - PRÉDICTION DU CHURN")
    print("=" * 60)
    
    # Attendre que l'API soit disponible
    if not wait_for_api():
        sys.exit(1)
    
    # Tests
    tests = [
        ("Santé", test_health),
        ("Route racine", test_root),
        ("Informations modèle", test_model_info),
        ("Prédiction", test_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1
    
    print(f"\nTests réussis: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 Tous les tests sont passés! L'API Docker fonctionne correctement.")
        sys.exit(0)
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les logs Docker.")
        sys.exit(1)

if __name__ == "__main__":
    main()
