from datetime import datetime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text

from .config import settings


connect_args = {"check_same_thread": False} if "sqlite" in settings.db_url else {} # Ensure the DB URL is valid for SQLite

engine = create_engine(settings.db_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    candidate_name = Column(String, index=True)
    jd_title = Column(String, index=True)
    overall_score = Column(Float)
    skills_score = Column(Float)
    experience_score = Column(Float)
    seniority_score = Column(Float)
    raw_assessment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)