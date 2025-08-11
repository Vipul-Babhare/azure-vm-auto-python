from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import jwt
import httpx
import os
from typing import Optional, Any, Dict

# FastAPI app
app = FastAPI(
    title="JWT Token Validator & ML Proxy",
    description="Validates JWT tokens and forwards requests to TensorFlow Serving",
    version="2.0.0"
)

# Configuration
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
MODEL_SERVING_URL = os.getenv('MODEL_SERVING_URL', 'http://localhost:8083')

# Security
security = HTTPBearer()

# Pydantic models
class ErrorResponse(BaseModel):
    """Standardized error response format"""
    token: str = ""
    expires_in: str = ""
    token_type: str = ""
    message: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str

# JWT Token validation dependency
async def validate_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Validate JWT token from Authorization header"""
    try:
        # Decode and validate token
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=['HS256'])
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": "Token has expired"
            }
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": "Invalid token"
            }
        )

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info"""
    return {
        "service": "JWT Token Validator & ML Proxy",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/v1/models/{model_name}:predict (POST) - Validates token and makes prediction",
            "batch_predict": "/v1/models/{model_name}:batch_predict (POST) - For batch predictions",
            "docs": "/docs"
        },
        "tensorflow_serving": MODEL_SERVING_URL,
        "note": "All prediction endpoints validate JWT tokens automatically"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="token_validator_proxy"
    )

@app.post("/v1/models/{model_name}:predict")
async def predict_with_validation(
    model_name: str,
    request: Request,
    token_payload: dict = Depends(validate_jwt_token)
):
    """
    MAIN ENDPOINT - Validates JWT token and makes prediction in single request
    
    Usage:
    1. Client includes JWT token in Authorization header: "Bearer <token>"
    2. This endpoint validates the token automatically
    3. If valid, forwards prediction request to TensorFlow Serving
    4. Returns prediction results or authentication error
    
    URL: POST /v1/models/rainfall_model:predict
    Headers: Authorization: Bearer <your-jwt-token>
    Body: { "instances": [...] }
    """
    
    try:
        # Build the TensorFlow Serving URL
        tf_serving_url = f"{MODEL_SERVING_URL}/v1/models/{model_name}:predict"
        
        print(f"✅ JWT Token validated successfully")
        print(f"👤 User: {token_payload.get('purpose', 'Unknown')}")
        print(f"🔄 Forwarding to: {tf_serving_url}")
        
        # Get the prediction request data
        request_data = await request.json()
        
        # Forward to TensorFlow Serving
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                tf_serving_url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
        
        print(f"📊 TensorFlow Response: {response.status_code}")
        
        # Return TensorFlow Serving response
        if response.status_code == 200:
            return JSONResponse(
                content=response.json(),
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "error": "Model prediction failed",
                    "details": response.text,
                    "tf_status": response.status_code
                },
                status_code=response.status_code
            )
        
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": f"Cannot connect to TensorFlow Serving at {tf_serving_url}"
            }
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": "Model prediction request timed out"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": f"Prediction error: {str(e)}"
            }
        )

@app.post("/v1/models/{model_name}:batch_predict")
async def batch_predict_with_validation(
    model_name: str,
    prediction_data: Dict[str, Any],
    token_payload: dict = Depends(validate_jwt_token)
):
    """
    Batch prediction endpoint with JWT validation
    
    For processing multiple predictions in a single request
    """
    try:
        tf_serving_url = f"{MODEL_SERVING_URL}/v1/models/{model_name}:predict"
        
        print(f"✅ JWT Token validated for batch prediction")
        print(f"👤 User: {token_payload.get('purpose', 'Unknown')}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                tf_serving_url,
                json=prediction_data,
                headers={"Content-Type": "application/json"}
            )
        
        if response.status_code == 200:
            return JSONResponse(content=response.json(), status_code=200)
        else:
            return JSONResponse(
                content={
                    "error": "Batch prediction failed",
                    "details": response.text
                },
                status_code=response.status_code
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": f"Batch prediction error: {str(e)}"
            }
        )

# Optional: Quick token validation endpoint (for debugging)
@app.get("/token/info")
async def get_token_info(token_payload: dict = Depends(validate_jwt_token)):
    """Get information about the provided JWT token"""
    return {
        "valid": True,
        "payload": token_payload,
        "message": "Token is valid and active"
    }

# Run with: uvicorn app:app --host 0.0.0.0 --port 8083
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)