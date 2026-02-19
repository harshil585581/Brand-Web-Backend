from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Query
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from .database import get_db, engine, Base
from . import models
from passlib.context import CryptContext
from dotenv import load_dotenv
import shutil
import uuid
import os

# Load .env manually from absolute path to handle CWD discrepancies
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Password hashing configuration
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Mount uploads directory to serve static files
os.makedirs("backend/uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/api/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    if not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    return {
        "message": "Login successful", 
        "user_id": db_user.id,
        "user": {
            "name": db_user.name,
            "username": db_user.username,
            "email": db_user.email,
            "role": db_user.role,
            "profile_img": db_user.profile_img
        }
    }

@app.post("/api/members")
async def create_member(
    name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    permissions: str = Form(None),
    profile_picture: UploadFile = File(None),
    profile_picture_url: str = Form(None),
    db: Session = Depends(get_db)
):
    # Check for duplicate email or username
    existing_user_email = db.query(models.User).filter(models.User.email == email).first()
    if existing_user_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    existing_user_username = db.query(models.User).filter(models.User.username == username).first()
    if existing_user_username:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Handle profile picture
    profile_img_path = None
    if profile_picture:
        file_extension = profile_picture.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"backend/uploads/{filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(profile_picture.file, buffer)
        # Store the URL path that the frontend can access
        profile_img_path = f"/uploads/{filename}" 
    elif profile_picture_url:
        profile_img_path = profile_picture_url

    # Hash password
    hashed_password = pwd_context.hash(password)

    new_user = models.User(
        name=name,
        username=username,
        email=email,
        password=hashed_password,
        role=role,
        permissions=permissions,
        profile_img=profile_img_path
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Log activity (System or Admin? We don't have current_user here easily without auth dependency change)
    # For now, let's assume it's a public registration or admin action. 
    # If we want to track WHO created it, we need authentication on this endpoint. 
    # Currently `create_member` seems open? No `Depends(get_current_user)`.
    # Let's log it with a system user ID or just handle it if it's an authenticated route later.
    # Actually, the user requirement says "if team member created it should show that".
    # We can log it as "New Team Member Joined" or similar.
    # Let's assign it to the new user themselves? "User X joined".
    log_activity(db, new_user.id, "MEMBER_JOINED", f"New team member {new_user.name} joined.")

    return {"message": "Member created successfully", "user_id": new_user.id}

from typing import Optional

class UserDisplay(BaseModel):
    id: int
    name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    profile_img: Optional[str] = None
    permissions: Optional[str] = None

    class Config:
        from_attributes = True

@app.get("/api/members", response_model=list[UserDisplay])
def get_members(db: Session = Depends(get_db)):
    users = db.query(models.User).order_by(models.User.id.asc()).all()
    return users

@app.get("/api/users/{user_id}")
def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "name": user.name,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "profile_img": user.profile_img,
        "permissions": user.permissions,
        "generations_count": user.generations_count or 0
    }

@app.delete("/api/members/{user_id}")
def delete_member(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

class RoleUpdate(BaseModel):
    role: str

@app.put("/api/members/{user_id}/role")
def update_role(user_id: int, role_update: RoleUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.role = role_update.role
    db.commit()
    return {"message": "Role updated successfully"}

# Real-time Chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            # Expecting data = {"receiver_id": int, "content": str}
            receiver_id = data.get("receiver_id")
            content = data.get("content")

            if receiver_id and content:
                # Save to DB
                new_message = models.Message(
                    sender_id=user_id,
                    receiver_id=receiver_id,
                    content=content,
                    status="sent"
                )
                db.add(new_message)
                db.commit()
                db.refresh(new_message)

                message_payload = {
                    "id": new_message.id,
                    "sender_id": user_id,
                    "receiver_id": receiver_id,
                    "content": content,
                    "timestamp": new_message.timestamp.isoformat(),
                    "is_read": False,
                    "status": "sent"
                }

                # Send to receiver
                await manager.send_personal_message(message_payload, receiver_id)
                # Send back to sender (so they see their own message confirmed/appended)
                await manager.send_personal_message(message_payload, user_id)

            # Handle Status Updates (Delivery / Read Receipts)
            message_type = data.get("type")
            if message_type in ["delivery_receipt", "read_receipt"]:
                message_id = data.get("message_id")
                if message_id:
                    msg = db.query(models.Message).filter(models.Message.id == message_id).first()
                    if msg:
                        # Update status
                        new_status = "delivered" if message_type == "delivery_receipt" else "read"
                        
                        # Only update if progressing (sent -> delivered -> read)
                        # Or if we want to ensure we don't regress (read -> delivered shouldn't happen)
                        current_status_priority = {"sent": 0, "delivered": 1, "read": 2}
                        if current_status_priority.get(new_status, 0) > current_status_priority.get(msg.status, 0):
                            msg.status = new_status
                            if new_status == "read":
                                msg.is_read = True
                            db.commit()
                            
                            # Broadcast update to sender so their UI updates
                            update_payload = {
                                "type": "status_update",
                                "message_id": message_id,
                                "status": new_status,
                                "receiver_id": msg.receiver_id,
                                "sender_id": msg.sender_id
                            }
                            await manager.send_personal_message(update_payload, msg.sender_id)
                            # Also reflect to receiver? Maybe not needed, but good for sync if multiple devices
                            await manager.send_personal_message(update_payload, msg.receiver_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            manager.disconnect(websocket, user_id)
        except:
            pass

from datetime import datetime

class MessageDisplay(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    content: str
    timestamp: datetime
    is_read: bool
    status: str

    class Config:
        from_attributes = True

@app.get("/api/messages/{other_user_id}", response_model=List[MessageDisplay])
def get_messages(other_user_id: int, current_user_id: int = Query(...), db: Session = Depends(get_db)):
    # Fetch messages between current_user and other_user
    messages = db.query(models.Message).filter(
        ((models.Message.sender_id == current_user_id) & (models.Message.receiver_id == other_user_id)) |
        ((models.Message.sender_id == other_user_id) & (models.Message.receiver_id == current_user_id))
    ).order_by(models.Message.timestamp.asc()).all()
    
    return messages

# Event APIs

class EventCreate(BaseModel):
    title: str
    brand: str = None
    date: str
    start_time: str = None
    end_time: str = None
    owner_id: int = None # If None, use current user
    description: str = None
    type: str # campaign, meeting, task
    color: str
    participants: List[int] = [] # List of user IDs

@app.post("/api/events")
def create_event(event: EventCreate, db: Session = Depends(get_db)):
    # Create Event
    new_event = models.Event(
        title=event.title,
        brand=event.brand,
        date=event.date,
        start_time=event.start_time,
        end_time=event.end_time,
        owner_id=event.owner_id, # Should be validated or set to current user if not provided
        description=event.description,
        type=event.type,
        color=event.color
    )
    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    # Add Participants
    for user_id in event.participants:
        participant = models.EventParticipant(event_id=new_event.id, user_id=user_id)
        db.add(participant)
    
    db.commit()
    
    # Log Activity
    # Determine type for readable log
    action_type = "CREATED_TASK" if event.type == 'task' else "CREATED_EVENT"
    log_activity(db, event.owner_id, action_type, f"Created {event.type}: {event.title}")

    return {"message": "Event created successfully", "event_id": new_event.id}

class EventDisplay(BaseModel):
    id: int
    title: str
    brand: Optional[str] = None
    date: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    owner: Optional[UserDisplay] = None # Return full owner object
    description: Optional[str] = None
    type: str
    color: str
    participants: List[UserDisplay] = []

    class Config:
        from_attributes = True

@app.get("/api/events", response_model=List[EventDisplay])
def get_events(user_id: int = Query(None), db: Session = Depends(get_db)):
    # If user_id is provided, fetch events where user is owner OR participant
    # This logic assumes we want to filter by user. If no user_id, maybe return all?
    # Let's enforce user_id for now or default to return all if None (admin view)
    
    if user_id:
        # Subquery for events where user is participant
        participant_events = db.query(models.EventParticipant.event_id).filter(models.EventParticipant.user_id == user_id).subquery()
        
        events = db.query(models.Event).filter(
            (models.Event.owner_id == user_id) | 
            (models.Event.id.in_(participant_events))
        ).order_by(models.Event.date.asc(), models.Event.start_time.asc()).all()
    else:
        events = db.query(models.Event).order_by(models.Event.date.asc(), models.Event.start_time.asc()).all()
    
    # We might want to filter by date >= today in backend too, but frontend can handle "Upcoming" vs "Past"
    # User asked for "nearest to today date in order".
    # But `get_events` might be used by Calendar which needs ALL events.
    # So let's keep it returning all, but sorted. Frontend Dashboard will filter >= today.
    
    # Format response to include participants details
    result = []
    for event in events:
        owner = db.query(models.User).filter(models.User.id == event.owner_id).first()
        
        participant_associations = db.query(models.EventParticipant).filter(models.EventParticipant.event_id == event.id).all()
        participant_ids = [p.user_id for p in participant_associations]
        participants = db.query(models.User).filter(models.User.id.in_(participant_ids)).all()
        
        result.append({
            "id": event.id,
            "title": event.title,
            "brand": event.brand,
            "date": event.date,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "owner": owner,
            "description": event.description,
            "type": event.type,
            "color": event.color,
            "participants": participants
        })
        
    return result

# Brand API Endpoints

class BrandCreate(BaseModel):
    name: str = Form(...)
    description: str = Form(...)
    industry: str = Form(...)
    archetype: str = Form(...)
    status: str = Form(...)
    owner: str = Form(...)
    version: str = Form(...)
    core_values: str = Form(...)
    
    # Colors
    primary_color: str = Form(...)
    primary_color_name: str = Form(...)
    primary_color_usage: str = Form(...)
    secondary_color: str = Form(...)
    secondary_color_name: str = Form(...)
    secondary_color_usage: str = Form(...)
    accent_color: str = Form(...)
    accent_color_name: str = Form(...)
    accent_color_usage: str = Form(...)

@app.post("/api/brands")
async def create_brand(
    name: str = Form(...),
    description: str = Form(...),
    industry: str = Form(...),
    archetype: str = Form(...),
    status: str = Form(...),
    owner: str = Form(...),
    version: str = Form(...),
    core_values: str = Form(...),
    primary_color: str = Form(...),
    primary_color_name: str = Form(...),
    primary_color_usage: str = Form(...),
    secondary_color: str = Form(...),
    secondary_color_name: str = Form(...),
    secondary_color_usage: str = Form(...),
    accent_color: str = Form(...),
    accent_color_name: str = Form(...),
    accent_color_usage: str = Form(...),
    logomark: UploadFile = File(None),
    wordmark: UploadFile = File(None),
    # Dynamic assets - Handling them might be tricky with Form(...), simplistic approach for now:
    # We will assume assets are uploaded separately or handled as a list of files with metadata json
    # For MVP, let's keep it simple: Just create the brand first.
    # Actually, let's handle assets in a separate endpoint or just basic file uploads if possible.
    # To support multiple assets in one go with metadata, we'd need a more complex multipart form parsing or JSON + Base64.
    # Strategy: API receives JSON for metadata and Files. 
    # But usually <form> is easiest for files. 
    # Let's start with Brand fields + Logo/Wordmark. Assets can be added later or we try to parse them.
    db: Session = Depends(get_db)
):
    # Generate slug
    slug = name.lower().replace(" ", "-")
    existing_brand = db.query(models.Brand).filter(models.Brand.slug == slug).first()
    if existing_brand:
        slug = f"{slug}-{uuid.uuid4().hex[:6]}"

    # Handle files
    logomark_path = None
    if logomark:
        ext = logomark.filename.split(".")[-1]
        fname = f"logo_{uuid.uuid4()}.{ext}"
        fpath = f"backend/uploads/{fname}"
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(logomark.file, buffer)
        logomark_path = f"/uploads/{fname}"

    wordmark_path = None
    if wordmark:
        ext = wordmark.filename.split(".")[-1]
        fname = f"wordmark_{uuid.uuid4()}.{ext}"
        fpath = f"backend/uploads/{fname}"
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(wordmark.file, buffer)
        wordmark_path = f"/uploads/{fname}"

    new_brand = models.Brand(
        slug=slug,
        name=name,
        description=description,
        industry=industry,
        archetype=archetype,
        status=status,
        owner=owner,
        version=version,
        core_values=core_values,
        logomark_url=logomark_path,
        wordmark_url=wordmark_path,
        primary_color=primary_color,
        primary_color_name=primary_color_name,
        primary_color_usage=primary_color_usage,
        secondary_color=secondary_color,
        secondary_color_name=secondary_color_name,
        secondary_color_usage=secondary_color_usage,
        accent_color=accent_color,
        accent_color_name=accent_color_name,
        accent_color_usage=accent_color_usage,
    )

    db.add(new_brand)
    db.commit()
    db.refresh(new_brand)
    
    # Log Activity
    # Try to find user ID by owner name
    user = db.query(models.User).filter(models.User.name == owner).first()
    user_id = user.id if user else 1 # Default to 1 if not found
    log_activity(db, user_id, "CREATED_BRAND", f"Created new brand: {new_brand.name}")

    return new_brand

@app.post("/api/brands/{brand_id}/assets")
async def upload_brand_asset(
    brand_id: int,
    file: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form(...),
    description: str = Form(None),
    db: Session = Depends(get_db)
):
    brand = db.query(models.Brand).filter(models.Brand.id == brand_id).first()
    if not brand:
         raise HTTPException(status_code=404, detail="Brand not found")

    ext = file.filename.split(".")[-1]
    fname = f"asset_{brand_id}_{uuid.uuid4()}.{ext}"
    fpath = f"backend/uploads/{fname}"
    with open(fpath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_url = f"/uploads/{fname}"
    
    # Get file size roughly
    file_size = f"{os.path.getsize(fpath) / 1024:.1f} KB"

    new_asset = models.BrandAsset(
        brand_id=brand_id,
        name=name,
        category=category,
        description=description,
        file_url=file_url,
        file_size=file_size
    )
    db.add(new_asset)
    db.commit()
    return new_asset

@app.get("/api/brands")
def get_brands(db: Session = Depends(get_db)):
    brands = db.query(models.Brand).all()
    # Populate asset count for listing?
    # For now return raw brands, frontend can handle.
    # Better: return a schema that includes asset count
    result = []
    for brand in brands:
        asset_count = db.query(models.BrandAsset).filter(models.BrandAsset.brand_id == brand.id).count()
        brand_data = brand.__dict__
        brand_data['assets_count'] = asset_count
        result.append(brand_data)
    return result

@app.get("/api/brands/{brand_id}")
def get_brand_details(brand_id: str, db: Session = Depends(get_db)):
    # Try ID first, then slug
    if brand_id.isdigit():
        brand = db.query(models.Brand).filter(models.Brand.id == int(brand_id)).first()
    else:
        brand = db.query(models.Brand).filter(models.Brand.slug == brand_id).first()
        
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    assets = db.query(models.BrandAsset).filter(models.BrandAsset.brand_id == brand.id).all()
    
    return {"brand": brand, "assets": assets}
    return {"brand": brand, "assets": assets}

@app.put("/api/brands/{brand_id}")
async def update_brand(
    brand_id: int,
    name: str = Form(...),
    description: str = Form(...),
    industry: str = Form(...),
    archetype: str = Form(...),
    status: str = Form(...),
    owner: str = Form(...),
    version: str = Form(...),
    core_values: str = Form(...),
    primary_color: str = Form(...),
    primary_color_name: str = Form(...),
    primary_color_usage: str = Form(...),
    secondary_color: str = Form(...),
    secondary_color_name: str = Form(...),
    secondary_color_usage: str = Form(...),
    accent_color: str = Form(...),
    accent_color_name: str = Form(...),
    accent_color_usage: str = Form(...),
    logomark: UploadFile = File(None),
    wordmark: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    brand = db.query(models.Brand).filter(models.Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    # Handle files if provided
    if logomark:
        ext = logomark.filename.split(".")[-1]
        fname = f"logo_{uuid.uuid4()}.{ext}"
        fpath = f"backend/uploads/{fname}"
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(logomark.file, buffer)
        brand.logomark_url = f"/uploads/{fname}"

    if wordmark:
        ext = wordmark.filename.split(".")[-1]
        fname = f"wordmark_{uuid.uuid4()}.{ext}"
        fpath = f"backend/uploads/{fname}"
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(wordmark.file, buffer)
        brand.wordmark_url = f"/uploads/{fname}"

    # Update fields
    brand.name = name
    # Update slug if name changes? Maybe keep slug stable or update it. 
    # Usually slugs are stable, but if name changes significantly, valid to update.
    # Let's keep slug stable for now to avoid breaking URLs, or maybe update if it serves as ID.
    # user didn't specify. let's update basic fields.
    brand.description = description
    brand.industry = industry
    brand.archetype = archetype
    brand.status = status
    brand.owner = owner
    brand.version = version
    brand.core_values = core_values
    brand.primary_color = primary_color
    brand.primary_color_name = primary_color_name
    brand.primary_color_usage = primary_color_usage
    brand.secondary_color = secondary_color
    brand.secondary_color_name = secondary_color_name
    brand.secondary_color_usage = secondary_color_usage
    brand.accent_color = accent_color
    brand.accent_color_name = accent_color_name
    brand.accent_color_usage = accent_color_usage
    
    brand.last_update = func.now()

    db.commit()
    db.refresh(brand)
    
    # Log Activity
    user = db.query(models.User).filter(models.User.name == owner).first()
    user_id = user.id if user else 1
    log_activity(db, user_id, "UPDATED_BRAND", f"Updated brand: {brand.name}")

    return brand

class ContentGenerationRequest(BaseModel):
    brand_id: int
    prompt: str
    format: str
    creativity: int # 0-100
    tone: list[str]
    user_id: int

@app.post("/api/generate-content")
async def generate_content(request: ContentGenerationRequest, db: Session = Depends(get_db)):
    brand = db.query(models.Brand).filter(models.Brand.id == request.brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    # Construct the system prompt based on brand identity
    system_prompt = f"""
You are an advanced Brand Content Intelligence Engine for '{brand.name}'.
Your goal is to generate high-quality, brand-aligned content.

BRAND IDENTITY:
- Name: {brand.name}
- Industry: {brand.industry}
- Archetype: {brand.archetype}
- Core Values: {brand.core_values}
- Description: {brand.description}
- Tone: {', '.join(request.tone)}

INSTRUCTIONS:
1. Analyze the user prompt and brand identity.
2. Generate content in the format: {request.format}.
3. Creativity Level: {request.creativity}/100 (Low=Safe, High=Wild).
4. Output ONLY the generated content. No conversational filler.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173", # Optional, for OpenRouter rankings
        "X-Title": "Brand Web AI Studio", # Optional, for OpenRouter rankings
    }

    # Map creativity (0-100) to temperature (0.0-1.0)
    temperature = request.creativity / 100.0

    payload = {
        "model": "mistralai/mistral-7b-instruct", # Removed :free suffix to use available endpoint
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt}
        ],
        "temperature": temperature,
    }

    import requests
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            
            # Increment user generations count
            user = db.query(models.User).filter(models.User.id == request.user_id).first()
            if user:
                user.generations_count = (user.generations_count or 0) + 1
                db.commit()
                
                # Log Activity
                log_activity(db, request.user_id, "GENERATED_CONTENT", f"Generated content for {brand.name}")

            return {"content": content}
        else:
            raise Exception("No content generated")

    except requests.exceptions.HTTPError as e:
        error_msg = f"OpenRouter API Error: {response.text}"
        if response.status_code == 402:
            raise HTTPException(status_code=402, detail="Insufficient API credits. Please add funds or use a free model.")
        
        print(error_msg)
        raise HTTPException(status_code=response.status_code, detail=error_msg)
    except Exception as e:
        print(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def log_activity(db: Session, user_id: int, action: str, details: str):
    try:
        log = models.ActivityLog(user_id=user_id, action=action, details=details)
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Failed to log activity: {e}")

@app.get("/api/logs")
def get_activity_logs(limit: int = 20, db: Session = Depends(get_db)):
    # Optimize query with join
    results = db.query(models.ActivityLog, models.User).outerjoin(models.User, models.ActivityLog.user_id == models.User.id).order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()
    
    response = []
    for log, user in results:
        response.append({
            "id": log.id,
            "user": user.name if user else "Unknown",
            "user_initials": user.name[:2].upper() if user and user.name else "??",
            "action": log.action,
            "details": log.details,
            "timestamp": log.timestamp.isoformat(),
            "time_ago": "just now"
        })
    return response

@app.get("/api/dashboard/stats/{user_id}")
def get_dashboard_stats(user_id: int, db: Session = Depends(get_db)):
    # Total Brands
    total_brands = db.query(models.Brand).count()
    
    # Active Campaigns: Brands with status='ACTIVE'
    # User requested to show total number of brand cards that are active.
    active_campaigns = db.query(models.Brand).filter(models.Brand.status == 'ACTIVE').count()
    
    # AI Generations
    user = db.query(models.User).filter(models.User.id == user_id).first()
    ai_generations = user.generations_count if user else 0
    
    # Team Members
    team_members = db.query(models.User).count()
    
    return {
        "total_brands": total_brands,
        "active_campaigns": active_campaigns,
        "ai_generations": ai_generations,
        "team_members": team_members
    }
