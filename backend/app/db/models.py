from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class DefectLog(Base):
    __tablename__ = "defect_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    defect_type = Column(String, index=True)
    confidence = Column(Float)
    inference_latency_ms = Column(Float)
    image_snapshot_path = Column(String, nullable=True)
    severity = Column(String)  # e.g., Low, Medium, High
