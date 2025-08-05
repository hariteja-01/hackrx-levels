import asyncio
from main import app
from fastapi.testclient import TestClient
import requests

# Test with FastAPI TestClient (for unit testing)
def test_api_with_testclient():
    client = TestClient(app)
    
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "questions": [
            "What is the sum insured for the policy?",
            "What is the policy term?",
            "What are the exclusions in this policy?"
        ],
        "language": "en"
    }
    
    response = client.post("/hackrx/run", json=test_data)
    print(response.json())

# Test with actual HTTP requests (for end-to-end testing)
async def test_api_with_http_requests():
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "questions": [
            "What is the sum insured for the policy?",
            "What is the policy term?",
            "What are the exclusions in this policy?"
        ],
        "language": "en"
    }
    
    async with requests.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/hackrx/run",
            json=test_data
        )
        print(response.json())

if __name__ == "__main__":
    # Run TestClient test
    print("Running TestClient test...")
    test_api_with_testclient()
    
    # Run HTTP test (requires server to be running)
    print("\nRunning HTTP test...")
    asyncio.run(test_api_with_http_requests())