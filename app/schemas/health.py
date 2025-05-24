# app/schemas/health.py
from datetime import datetime
from pydantic import BaseModel

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime
    system_stats: dict

    class Config:
        json_schema_extra = {
            "example": {
                "status": "OK",
                "model_loaded": True,
                "timestamp": "2024-05-22T14:30:45.123456",
                "system_stats": {
                    "memory_usage": 45.2,
                    "cpu_usage": 12.3
                }
            }
        }