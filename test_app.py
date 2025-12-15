"""
Simple test to validate Flask API endpoints
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all imports work"""
    try:
        # Mock tensorflow to avoid heavy installation
        import sys
        from unittest.mock import MagicMock
        sys.modules['tensorflow'] = MagicMock()
        
        import app
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_flask_app():
    """Test that Flask app is created"""
    try:
        import sys
        from unittest.mock import MagicMock
        sys.modules['tensorflow'] = MagicMock()
        
        from app import app as flask_app
        print(f"✓ Flask app created: {flask_app}")
        return True
    except Exception as e:
        print(f"✗ Flask app creation failed: {e}")
        return False

def test_routes():
    """Test that routes are registered"""
    try:
        import sys
        from unittest.mock import MagicMock
        sys.modules['tensorflow'] = MagicMock()
        
        from app import app as flask_app
        routes = [str(rule) for rule in flask_app.url_map.iter_rules()]
        print(f"✓ Registered routes: {routes}")
        
        # Check for required routes
        assert any('/health' in route for route in routes), "/health route not found"
        assert any('/predict' in route for route in routes), "/predict route not found"
        print("✓ All required routes present")
        return True
    except Exception as e:
        print(f"✗ Route test failed: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint without model"""
    try:
        import sys
        from unittest.mock import MagicMock
        sys.modules['tensorflow'] = MagicMock()
        
        from app import app as flask_app
        with flask_app.test_client() as client:
            response = client.get('/health')
            print(f"✓ Health endpoint response: {response.status_code}")
            print(f"  Response data: {response.get_json()}")
            assert response.status_code == 200, "Health endpoint should return 200"
            return True
    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False

if __name__ == '__main__':
    print("Running basic Flask API tests...\n")
    
    tests = [
        test_imports,
        test_flask_app,
        test_routes,
        test_health_endpoint
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "="*50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
