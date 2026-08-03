from fastapi import FastAPI

# Create FastAPI application instance
app = FastAPI(
    title="Credit Risk Knowledge Assistant API",
    description="Backend API for AI-powered Credit Risk Knowledge Assistant",
    version="1.0.0")


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Credit Risk Knowledge Assistant API is running"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "credit-risk-ai-assistant"
    }