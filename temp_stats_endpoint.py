
@app.get("/api/dashboard/stats/{user_id}")
def get_dashboard_stats(user_id: int, db: Session = Depends(get_db)):
    # Total Brands
    total_brands = db.query(models.Brand).count()
    
    # Active Campaigns (Active Brands that are marked as campaigns)
    # Based on models.py: is_campaign = Column(Boolean, default=False) and status = Column(String)
    # Let's count active campaigns as brands with status='ACTIVE' and is_campaign=True
    # Or simply brands where status='ACTIVE' if that's what user meant by campaigns. 
    # User request said "active campaigns is total number od active brands cards" 
    # In Portfolio.jsx logic: cardBg = brand.archived ? 'bg-zinc-200' : 'bg-white'; statusColor = ... brand.campaign ? ...
    # Let's assume Active Campaigns = Brands with status='ACTIVE' AND is_campaign=True
    # Wait, user said "active campaigns is total number od active brands cards"
    # Maybe simpler: allow filtering by status='ACTIVE'.
    # Let's look at Brand model again.
    # status = Column(String)
    # is_campaign = Column(Boolean)
    
    # Let's count brands where status == 'ACTIVE' and is_campaign == True
    active_campaigns = db.query(models.Brand).filter(models.Brand.status == 'ACTIVE', models.Brand.is_campaign == True).count()
    
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
