from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt
import datetime
import os
from typing import Optional

# FastAPI app
app = FastAPI(
    title="JWT Token Generator",
    description="Generates JWT tokens for ML model authentication",
    version="1.0.0"
)

# Configuration
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
TOKEN_EXPIRATION_HOURS = int(os.getenv('TOKEN_EXPIRATION_HOURS', '24'))

# Pydantic models
class TokenRequest(BaseModel):
    """Optional request body for token generation"""
    purpose: Optional[str] = "model_prediction"
    custom_data: Optional[dict] = None

class TokenResponse(BaseModel):
    """Token generation response"""
    token: str
    expires_in: int
    token_type: str
    message: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info"""
    return {
        "service": "JWT Token Generator",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_token": "/generate_token (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="token_generator"
    )

@app.post("/generate_token", response_model=TokenResponse)
async def generate_token(request: Optional[TokenRequest] = None):
    """Generate JWT token - NO AUTHENTICATION REQUIRED"""
    try:
        # Default request if none provided
        if request is None:
            request = TokenRequest()
        
        # Create token payload
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRATION_HOURS),
            'iat': datetime.datetime.utcnow(),
            'iss': 'ml_auth_system',
            'purpose': request.purpose,
        }
        
        # Add custom data if provided
        if request.custom_data:
            payload['custom_data'] = request.custom_data
        
        # Generate token
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        
        return TokenResponse(
            token=token,
            expires_in=TOKEN_EXPIRATION_HOURS * 3600,
            token_type="Bearer",
            message="Token generated successfully"
        )
        
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Token generation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "token": "",
                "expires_in": "",
                "token_type": "",
                "message": f"Token generation failed: {str(e)}"
            }
        )

# Run with: uvicorn app:app --host 0.0.0.0 --port 8082
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)