from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base

class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    subscription_status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True) # Start nullable for migration, enforce later
    name = Column(String)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    profile_img = Column(String, nullable=True)
    # company_name field in User might be redundant if we have company_id, but keeping for now as it might be used in frontend
    company_name = Column(String, nullable=True) 
    role = Column(String, default="employee") # admin, employee
    permissions = Column(String, nullable=True)
    generations_count = Column(Integer, default=0)

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, nullable=True) # Added to fix potential schema mismatch if needed, though not in original
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    content = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_read = Column(Boolean, default=False)
    status = Column(String, default="sent") # sent, delivered, read

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String) # CREATED_BRAND, UPDATED_BRAND, GENERATED_CONTENT, CREATED_MEMBER, CREATED_EVENT, CREATED_TASK
    details = Column(String) # Description text
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class Brand(Base):
    __tablename__ = "brands"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    slug = Column(String, unique=True, index=True) # For URL friendly ID
    name = Column(String)
    description = Column(String)
    industry = Column(String)
    archetype = Column(String)
    status = Column(String)
    owner = Column(String)
    version = Column(String)
    core_values = Column(String) # Comma separated
    logomark_url = Column(String, nullable=True)
    wordmark_url = Column(String, nullable=True)
    
    # Colors
    primary_color = Column(String, nullable=True)
    primary_color_name = Column(String, nullable=True)
    primary_color_usage = Column(String, nullable=True)
    
    secondary_color = Column(String, nullable=True)
    secondary_color_name = Column(String, nullable=True)
    secondary_color_usage = Column(String, nullable=True)
    
    accent_color = Column(String, nullable=True)
    accent_color_name = Column(String, nullable=True)
    accent_color_usage = Column(String, nullable=True)
    
    last_update = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_archived = Column(Boolean, default=False)
    is_campaign = Column(Boolean, default=False)

class BrandAsset(Base):
    __tablename__ = "brand_assets"

    id = Column(Integer, primary_key=True, index=True)
    brand_id = Column(Integer, ForeignKey("brands.id"))
    name = Column(String)
    category = Column(String)
    description = Column(String, nullable=True)
    file_url = Column(String)
    file_size = Column(String, nullable=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    title = Column(String)
    brand = Column(String, nullable=True)
    date = Column(String) # YYYY-MM-DD
    start_time = Column(String, nullable=True)
    end_time = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    description = Column(String, nullable=True)
    type = Column(String) # campaign, meeting, task
    color = Column(String)
    
class EventParticipant(Base):
    __tablename__ = "event_participants"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
