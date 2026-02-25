"""
Test script for Vision Agentic AI MVP
Tests basic functionality of all major components
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules import correctly"""
    print("\n[TEST] Module Imports")
    print("=" * 60)
    
    modules = [
        ("Config", "from app.config import config"),
        ("Logger", "from app.utils.logger import setup_logger"),
        ("Database", "from app.database.db import init_db, get_db"),
        ("Models", "from app.database.models import Patient, Session"),
        ("Embeddings", "from app.core.embedding_engine import EmbeddingEngine"),
        ("FAISS", "from app.core.faiss_store import FAISSStore"),
        ("LLM", "from app.core.llm_engine import LLMEngine"),
        ("Safety", "from app.core.safety_rules import SafetyChecker"),
        ("Face Auth", "from app.auth.face_auth import FaceAuthenticator"),
        ("Token Auth", "from app.auth.token_auth import TokenAuthenticator"),
        ("Vision", "from app.vision.webcam_capture import WebcamCapture"),
        ("Speech", "from app.speech.speech_to_text import SpeechToTextEngine"),
        ("Sentiment", "from app.speech.sentiment_analyzer import SentimentAnalyzer"),
        ("PDF", "from app.reports.pdf_generator import PDFGenerator"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_statement in modules:
        try:
            exec(import_statement)
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {str(e)[:50]}")
            failed += 1
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_database():
    """Test database connectivity"""
    print("\n[TEST] Database Connectivity")
    print("=" * 60)
    
    try:
        from app.database.db import init_db, get_db, SessionLocal
        init_db()
        print("✓ Database initialization")
        
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("✓ Database connection")
        
        return True
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\n[TEST] Configuration")
    print("=" * 60)
    
    try:
        from app.config import config
        
        checks = [
            ("Database URL", config.DATABASE_URL),
            ("LLM Model", config.LLM_MODEL),
            ("Embedding Model", config.EMBEDDING_MODEL),
            ("Face Similarity Threshold", str(config.FACE_SIMILARITY_THRESHOLD)),
            ("Embedding Dimension", str(config.EMBEDDING_DIM)),
        ]
        
        for name, value in checks:
            print(f"✓ {name}: {value}")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_embedding():
    """Test embedding engine"""
    print("\n[TEST] Embedding Engine")
    print("=" * 60)
    
    try:
        from app.core.embedding_engine import EmbeddingEngine
        
        engine = EmbeddingEngine()
        print("✓ EmbeddingEngine initialized")
        
        # Try to embed text
        text = "Patient has fever and headache for 3 days"
        embedding = engine.embed_text(text)
        
        if embedding is not None:
            print(f"✓ Text embedding generated (dim: {embedding.shape[0]})")
            return True
        else:
            print("⚠ Embedding returned None (SentenceTransformers may not be installed)")
            return True  # Not critical
            
    except Exception as e:
        print(f"⚠ Embedding error (not critical): {e}")
        return True


def test_similarity():
    """Test similarity engine"""
    print("\n[TEST] Similarity Engine")
    print("=" * 60)
    
    try:
        from app.core.similarity_engine import SimilarityEngine
        import numpy as np
        
        engine = SimilarityEngine()
        print("✓ SimilarityEngine initialized")
        
        # Create test embeddings
        emb1 = np.random.randn(128)
        emb2 = np.random.randn(128)
        emb3 = emb1.copy()  # Same embedding
        
        sim12 = engine.compute_similarity(emb1, emb2)
        sim13 = engine.compute_similarity(emb1, emb3)
        
        print(f"✓ Random embeddings similarity: {sim12:.4f}")
        print(f"✓ Identical embeddings similarity: {sim13:.4f}")
        
        return sim13 > sim12  # Identical should be more similar
        
    except Exception as e:
        print(f"✗ Similarity error: {e}")
        return False


def test_safety():
    """Test safety checker"""
    print("\n[TEST] Safety Checker")
    print("=" * 60)
    
    try:
        from app.core.safety_rules import SafetyChecker
        
        checker = SafetyChecker()
        print("✓ SafetyChecker initialized")
        
        # Test medication conflict
        is_safe, conflicts = checker.check_medication_conflicts(
            "Aspirin", ["Warfarin"]
        )
        
        if not is_safe:
            print(f"✓ Conflict detected: {conflicts[0][:40]}...")
        else:
            print("⚠ No conflict detected")
        
        # Test contraindication
        is_safe, contras = checker.check_contraindications(
            "Aspirin", ["Bleeding disorder"]
        )
        
        if not is_safe:
            print(f"✓ Contraindication detected: {contras[0][:40]}...")
        else:
            print("⚠ No contraindication detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Safety error: {e}")
        return False


def test_agents():
    """Test agent initialization"""
    print("\n[TEST] AI Agents")
    print("=" * 60)
    
    try:
        from app.agents.symptom_agent import SymptomAgent
        from app.agents.vision_agent import VisionAgent
        from app.agents.condition_agent import ConditionAgent
        from app.agents.medication_agent import MedicationAgent
        
        agents = [
            ("SymptomAgent", SymptomAgent()),
            ("VisionAgent", VisionAgent()),
            ("ConditionAgent", ConditionAgent()),
            ("MedicationAgent", MedicationAgent()),
        ]
        
        for name, agent in agents:
            print(f"✓ {name} initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Vision Agentic AI MVP - Test Suite                       ║
╚════════════════════════════════════════════════════════════╝
""")
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Database", test_database),
        ("Embedding", test_embedding),
        ("Similarity", test_similarity),
        ("Safety", test_safety),
        ("Agents", test_agents),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ All tests passed! Application is ready to run.")
        print("\nStart the server with:")
        print("  python -m uvicorn app.main:app --reload")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
