from sqlalchemy import Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


# ─── Official 4-Class Taxonomy ───────────────────────────────────────────────
class DefectType(str, enum.Enum):
    MICRO_CRACK     = "micro-crack"       # Class 0 – Hairline or visible cracks on blades
    THERMAL_PITTING = "thermal-pitting"   # Class 1 – Small pits caused by heat damage
    BLADE_EROSION   = "blade-erosion"     # Class 2 – Material loss from the blade edge or surface
    CORROSION       = "corrosion"         # Class 3 – Oxidation or rust-like degradation


class SeverityLevel(str, enum.Enum):
    LOW      = "Low"
    MEDIUM   = "Medium"
    HIGH     = "High"
    CRITICAL = "Critical"


class DefectLog(Base):
    """
    Persistent audit record for every confirmed defect detection.

    Severity thresholds (example mapping based on confidence):
        confidence >= 0.90  → Critical
        confidence >= 0.75  → High
        confidence >= 0.50  → Medium
        confidence <  0.50  → Low
    """
    __tablename__ = "defect_logs"

    id                   = Column(Integer, primary_key=True, index=True)
    timestamp            = Column(DateTime(timezone=True), server_default=func.now())
    class_id             = Column(Integer, nullable=False)                         # 0-3
    defect_type          = Column(
                               Enum(DefectType),
                               index=True,
                               nullable=False,
                           )
    confidence           = Column(Float, nullable=False)
    inference_latency_ms = Column(Float)
    bbox_x1              = Column(Float)   # Bounding-box coords (pixel space, 640×640)
    bbox_y1              = Column(Float)
    bbox_x2              = Column(Float)
    bbox_y2              = Column(Float)
    severity             = Column(Enum(SeverityLevel))
    image_snapshot_path  = Column(String, nullable=True)
