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
import traceback
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import joinedload

# Load .env manually from absolute path to handle CWD discrepancies
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

# Create tables if they don't exist
# Create tables if they don't exist
# models.Base.metadata.create_all(bind=engine)

# Run Alembic Migrations
from alembic.config import Config
from alembic import command
import os


# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
try:
    command.upgrade(alembic_cfg, "head")
except Exception as e:
    print(f"Migration failed: {e}")


app = FastAPI()

@app.get("/create-admin")
def create_admin(db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == "admin@gmail.com").first()
    if existing:
        return {"message": "Admin already exists"}

    # Create Default Company for Admin
    existing_company = db.query(models.Company).filter(models.Company.name == "Admin Company").first()
    if not existing_company:
        admin_company = models.Company(name="Admin Company")
        db.add(admin_company)
        db.commit()
        db.refresh(admin_company)
        company_id = admin_company.id
    else:
        company_id = existing_company.id

    hashed_password = pwd_context.hash("shubam123")

    admin_user = models.User(
        name="Admin User",
        username="shubam",
        email="shubam@gmail.com",
        password=hashed_password,
        role="admin",
        permissions="all",
        company_id=company_id,
        company_name="Admin Company"
    )

    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)

    return {"message": "Admin created successfully"}


from sqlalchemy import inspect

@app.get("/check-tables")
def check_tables():
    inspector = inspect(engine)
    return {"tables": inspector.get_table_names()}


# Password hashing configuration
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Mount uploads directory to serve static files
os.makedirs("backend/uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ], 
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
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email, "user_id": db_user.id, "company_id": db_user.company_id},
        expires_delta=access_token_expires
    )

    return {
        "message": "Login successful", 
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": db_user.id,
        "user": {
            "id": db_user.id,
            "name": db_user.name,
            "username": db_user.username,
            "email": db_user.email,
            "role": db_user.role,
            "profile_img": db_user.profile_img,
            "company_id": db_user.company_id,
            "company_name": db_user.company_name
        }
    }

@app.post("/api/members")
async def create_member(
    name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    company_name: str = Form(None),
    role: str = Form("employee"), # Default to employee if not provided
    permissions: str = Form(None),
    profile_picture: UploadFile = File(None),
    profile_picture_url: str = Form(None),
    db: Session = Depends(get_db)
):
    # Logic: 
    # If authenticated user creates member -> assign to same company
    # If public sign up (no auth) -> Create NEW Company based on company_name
    
    # We need to distinguish between "Admin creating employee" vs "New User Signup"
    # For now, let's assume if 'company_name' is provided and it's a signup, we check or create company.
    
    try:
        # Check duplicate email/username
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
            profile_img_path = f"/uploads/{filename}" 
        elif profile_picture_url:
            profile_img_path = profile_picture_url

        # Hash password
        hashed_password = pwd_context.hash(password)

        # Company Logic
        if company_name:
            # Check if company exists (by name? risky if duplicates allowed, but let's assume unique names for SaaS)
            # IMPROVEMENT: If logged in admin, use their company_id. 
            # But this endpoint is likely used for Signup Page too.
            # Let's try to lookup company.
            company = db.query(models.Company).filter(models.Company.name == company_name).first()
            if not company:
                # Create new company
                company = models.Company(name=company_name)
                db.add(company)
                db.commit()
                db.refresh(company)
            
            final_company_id = company.id
            final_company_name = company.name
        else:
            # If no company name provided, maybe fallback or error?
            # For an employee being added by admin, the admin should pass company_name field or we need auth here.
            # Let's require company_name for now if public signup.
            # If we had `current_user`, we could default to `current_user.company_id`.
            # Since this is "create_member" (could be signup), let's create a default "Personal Workspace" if missing?
            # Or error.
            raise HTTPException(status_code=400, detail="Company Name is required")

        new_user = models.User(
            name=name,
            username=username,
            email=email,
            password=hashed_password,
            company_name=final_company_name,
            company_id=final_company_id,
            role=role,
            permissions=permissions,
            profile_img=profile_img_path
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        log_activity(db, new_user.id, "MEMBER_JOINED", f"New team member {new_user.name} joined.")

        return {"message": "Member created successfully", "user_id": new_user.id}
    except Exception as e:
        print(f"Error creating member: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

from typing import Optional

class UserDisplay(BaseModel):
    id: int
    name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    company_name: Optional[str] = None
    role: Optional[str] = None
    profile_img: Optional[str] = None
    permissions: Optional[str] = None

    class Config:
        from_attributes = True

@app.get("/api/members", response_model=list[UserDisplay])
def get_members(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Filter by Company
    users = db.query(models.User).filter(models.User.company_id == current_user.company_id).order_by(models.User.id.asc()).all()
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
def delete_member(user_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id, models.User.company_id == current_user.company_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Handle Foreign Key Constraints
    # 1. Remove from Event Participants
    db.query(models.EventParticipant).filter(models.EventParticipant.user_id == user_id).delete(synchronize_session=False)

    # 2. Unlink Messages (Sender & Receiver)
    db.query(models.Message).filter(models.Message.sender_id == user_id).update({models.Message.sender_id: None}, synchronize_session=False)
    db.query(models.Message).filter(models.Message.receiver_id == user_id).update({models.Message.receiver_id: None}, synchronize_session=False)

    # 3. Unlink Events (Owner)
    db.query(models.Event).filter(models.Event.owner_id == user_id).update({models.Event.owner_id: None}, synchronize_session=False)

    # 4. Unlink Activity Logs
    db.query(models.ActivityLog).filter(models.ActivityLog.user_id == user_id).update({models.ActivityLog.user_id: None}, synchronize_session=False)

    # Now safe to delete user
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
                    # Ensure accessing message within company scope if needed, 
                    # but websockets effectively authenticated by connection. 
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
def create_event(event: EventCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Create Event
    new_event = models.Event(
        title=event.title,
        brand=event.brand,
        date=event.date,
        start_time=event.start_time,
        end_time=event.end_time,
        owner_id=current_user.id, # Force usage of logged in user
        company_id=current_user.company_id,
        description=event.description,
        type=event.type,
        color=event.color
    )
    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    # Add Participants (validate they belong to company?)
    for user_id in event.participants:
        # Check participant is in same company
        part_user = db.query(models.User).filter(models.User.id == user_id, models.User.company_id == current_user.company_id).first()
        if part_user:
            participant = models.EventParticipant(event_id=new_event.id, user_id=user_id)
            db.add(participant)
    
    db.commit()
    
    # Log Activity
    action_type = "CREATED_TASK" if event.type == 'task' else "CREATED_EVENT"
    log_activity(db, new_event.owner_id, action_type, f"Created {event.type}: {event.title}")

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
def get_events(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # Filter by Company
    events = db.query(models.Event).filter(models.Event.company_id == current_user.company_id).order_by(models.Event.date.asc(), models.Event.start_time.asc()).all()
    
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

class BrandAssetDisplay(BaseModel):
    id: int
    brand_id: int
    name: str
    category: str
    description: Optional[str] = None
    file_url: str
    file_size: Optional[str] = None
    upload_date: datetime

    class Config:
        from_attributes = True

class BrandDisplay(BaseModel):
    id: int
    company_id: Optional[int] = None
    slug: str
    name: str
    description: str
    industry: str
    archetype: str
    status: str
    owner: str
    version: str
    core_values: str
    logomark_url: Optional[str] = None
    wordmark_url: Optional[str] = None
    primary_color: Optional[str] = None
    primary_color_name: Optional[str] = None
    primary_color_usage: Optional[str] = None
    secondary_color: Optional[str] = None
    secondary_color_name: Optional[str] = None
    secondary_color_usage: Optional[str] = None
    accent_color: Optional[str] = None
    accent_color_name: Optional[str] = None
    accent_color_usage: Optional[str] = None
    last_update: Optional[datetime] = None
    is_archived: bool
    is_campaign: bool
    assets_count: Optional[int] = 0

    class Config:
        from_attributes = True

class BrandDetailDisplay(BaseModel):
    brand: BrandDisplay
    assets: List[BrandAssetDisplay]

@app.post("/api/brands", response_model=BrandDisplay)
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
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Generate slug
    slug = name.lower().replace(" ", "-")
    existing_brand = db.query(models.Brand).filter(models.Brand.slug == slug, models.Brand.company_id == current_user.company_id).first()
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
        company_id=current_user.company_id,
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
    
    log_activity(db, current_user.id, "CREATED_BRAND", f"Created new brand: {new_brand.name}")

    return new_brand



@app.post("/api/brands/{brand_id}/assets", response_model=BrandAssetDisplay)
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
def get_brands(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    brands = db.query(models.Brand).filter(models.Brand.company_id == current_user.company_id).all()
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

    return result



@app.get("/api/brands/{brand_id}", response_model=BrandDetailDisplay)
def get_brand_details(brand_id: str, db: Session = Depends(get_db)):
    try:
        # Try ID first, then slug
        if brand_id.isdigit():
            brand = db.query(models.Brand).filter(models.Brand.id == int(brand_id)).first()
        else:
            brand = db.query(models.Brand).filter(models.Brand.slug == brand_id).first()
            
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")
        
        assets = db.query(models.BrandAsset).filter(models.BrandAsset.brand_id == brand.id).all()
        
        return {"brand": brand, "assets": assets}
    except Exception as e:
        print(f"Error fetching brand details: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

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
        user = db.query(models.User).filter(models.User.id == user_id).first()
        company_id = user.company_id if user else None
        
        log = models.ActivityLog(user_id=user_id, company_id=company_id, action=action, details=details)
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Failed to log activity: {e}")

@app.get("/api/logs")
def get_activity_logs(limit: int = 20, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # Optimize query with join and filter by company
    results = db.query(models.ActivityLog, models.User).outerjoin(models.User, models.ActivityLog.user_id == models.User.id).filter(models.ActivityLog.company_id == current_user.company_id).order_by(models.ActivityLog.timestamp.desc()).limit(limit).all()
    
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

@app.get("/api/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # Total Brands (Company scoped)
    total_brands = db.query(models.Brand).filter(models.Brand.company_id == current_user.company_id).count()
    
    # Active Campaigns: Brands with status='ACTIVE' (Company scoped)
    active_campaigns = db.query(models.Brand).filter(models.Brand.status == 'ACTIVE', models.Brand.company_id == current_user.company_id).count()
    
    # AI Generations (User scoped? Or Company scoped?)
    # Usually stats are for the team. Let's make it company wide or keep it user specific?
    # User requirement: "all pages ... scoped only to their company"
    # "Dashboard ... should display only the data associated with that company."
    # AI generations count might be interesting as a company total.
    # But User model has `generations_count`.
    # Let's sum up generations for all users in the company.
    company_users = db.query(models.User).filter(models.User.company_id == current_user.company_id).all()
    ai_generations = sum([(u.generations_count or 0) for u in company_users])
    
    # Team Members (Company scoped)
    team_members = len(company_users)
    
    return {
        "total_brands": total_brands,
        "active_campaigns": active_campaigns,
        "ai_generations": ai_generations,
        "team_members": team_members
    }
