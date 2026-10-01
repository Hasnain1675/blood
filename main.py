"""Smart Blood & Emergency Donor Network - single-file backend (FastAPI + SQLAlchemy + SQLite).
Run: uvicorn main:app --reload   then open http://localhost:8000
"""
import hashlib, math, os, random, secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

engine = create_engine(os.getenv("DB_URL", "sqlite:///blood.db"), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------- MODELS ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String); email = Column(String, unique=True); phone = Column(String)
    pw = Column(String); salt = Column(String)
    role = Column(String)  # requester | donor | coordinator | admin
    token = Column(String, index=True)
    blood_group = Column(String, nullable=True)
    lat = Column(Float, nullable=True); lng = Column(Float, nullable=True); city = Column(String, nullable=True)
    last_donation = Column(DateTime, nullable=True)
    available = Column(Boolean, default=True)
    total_donations = Column(Integer, default=0)
    active = Column(Boolean, default=True)

class BloodRequest(Base):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True)
    requester_id = Column(Integer, ForeignKey("users.id"))
    blood_group = Column(String); units_required = Column(Integer); units_arranged = Column(Integer, default=0)
    hospital = Column(String); city = Column(String); lat = Column(Float); lng = Column(Float)
    required_before = Column(DateTime); urgency = Column(String); description = Column(String, default="")
    summary = Column(String, default="")
    verification_status = Column(String, default="pending")  # pending | verified | rejected
    status = Column(String, default="Pending Verification")
    radius_km = Column(Float, default=10); wave = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow); completed_at = Column(DateTime, nullable=True)

class DonorResponse(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True)
    request_id = Column(Integer, ForeignKey("requests.id")); donor_id = Column(Integer, ForeignKey("users.id"))
    match_score = Column(Float); distance_km = Column(Float)
    response_status = Column(String, default="notified")  # notified | accepted | declined
    notified_at = Column(DateTime, default=datetime.utcnow); response_time = Column(DateTime, nullable=True)
    donation_confirmed = Column(Boolean, default=False)

Base.metadata.create_all(engine)

# ---------------- HELPERS ----------------
COMPAT = {  # recipient -> donors allowed
    "O-": ["O-"], "O+": ["O+", "O-"], "A-": ["A-", "O-"], "A+": ["A+", "A-", "O+", "O-"],
    "B-": ["B-", "O-"], "B+": ["B+", "B-", "O+", "O-"], "AB-": ["AB-", "A-", "B-", "O-"],
    "AB+": ["AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"]}
HOSPITALS = {  # approximate coordinates, edit freely
    "PIMS Islamabad": (33.7036, 73.0507, "Islamabad"), "Polyclinic Hospital": (33.7027, 73.0654, "Islamabad"),
    "Shifa International": (33.6844, 73.0843, "Islamabad"), "Holy Family Rawalpindi": (33.6425, 73.0679, "Rawalpindi"),
    "Civil Hospital": (24.8607, 67.0099, "Karachi")}
MIN_GAP_DAYS = 90
STATUS_ORDER = ["Pending Verification", "Active", "Donors Contacted", "Partially Fulfilled", "Fulfilled", "Completed"]

def hash_pw(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()

def haversine(a, b, c, d):
    R = 6371; p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def current_user(x_token: str = Header(None), db: Session = Depends(get_db)) -> User:
    u = db.query(User).filter(User.token == x_token, User.active == True).first() if x_token else None  # noqa
    if not u: raise HTTPException(401, "Login required")
    return u

def require(*roles):
    def dep(u: User = Depends(current_user)):
        if u.role not in roles: raise HTTPException(403, "Not allowed for your role")  # RBAC on backend
        return u
    return dep

# ---------------- AI-ish FEATURES (rule-based, swap with LLM if time) ----------------
def classify_priority(desc, required_before):
    hrs = (required_before - datetime.utcnow()).total_seconds() / 3600
    text = (desc or "").lower()
    if hrs <= 3 or any(k in text for k in ["accident", "surgery", "bleeding", "critical", "icu", "emergency"]): return "Critical"
    if hrs <= 12 or "urgent" in text: return "Urgent"
    return "Normal"

def summarize(r):  # replace with LLM call for "AI summarization"
    d = (r.description or "").strip().replace("\n", " ")
    short = d[:90] + ("..." if len(d) > 90 else "")
    return f"URGENT {r.blood_group}: {r.units_required} unit(s) at {r.hospital}. {short}".strip()

def is_duplicate(db, r):
    since = datetime.utcnow() - timedelta(hours=24)
    return db.query(BloodRequest).filter(BloodRequest.requester_id == r.requester_id, BloodRequest.hospital == r.hospital,
        BloodRequest.blood_group == r.blood_group, BloodRequest.created_at >= since,
        BloodRequest.status.notin_(["Cancelled", "Rejected", "Expired"])).first() is not None

# ---------------- SMART MATCHER ----------------
def rank_donors(db, req):
    now = datetime.utcnow(); out = []
    already = {x.donor_id for x in db.query(DonorResponse).filter_by(request_id=req.id)}
    for d in db.query(User).filter(User.role == "donor", User.active == True, User.available == True,  # noqa
                                   User.blood_group.in_(COMPAT[req.blood_group])):
        if d.id in already or d.lat is None: continue
        if d.last_donation and (now - d.last_donation).days < MIN_GAP_DAYS: continue   # hard eligibility filter
        dist = haversine(req.lat, req.lng, d.lat, d.lng)
        if dist > req.radius_km: continue
        notified = db.query(DonorResponse).filter_by(donor_id=d.id).count()
        accepted = db.query(DonorResponse).filter_by(donor_id=d.id, response_status="accepted").count()
        history = (accepted / notified) if notified else 0.5
        gap = min(1, ((now - d.last_donation).days if d.last_donation else 365) / 180)
        exact = 1 if d.blood_group == req.blood_group else 0.6
        urg = {"Critical": 1, "Urgent": 0.6, "Normal": 0.3}[req.urgency]
        score = 100 * (0.40 * max(0, 1 - dist / req.radius_km) + 0.20 * 1 + 0.15 * history + 0.15 * gap + 0.05 * exact + 0.05 * urg)
        out.append((round(score, 1), round(dist, 1), d))
    return sorted(out, key=lambda x: -x[0])

def dispatch(db, req):
    """Wave 0 -> top 5, wave 1 -> next 10, wave 2+ -> radius +10km and next 10. Stops when units are covered."""
    if req.units_arranged >= req.units_required: return 0
    if req.wave >= 2: req.radius_km += 10
    n = 5 if req.wave == 0 else 10
    picked = rank_donors(db, req)[:n]
    for score, dist, d in picked:
        db.add(DonorResponse(request_id=req.id, donor_id=d.id, match_score=score, distance_km=dist))
        print(f"[NOTIFY] -> {d.name}: {summarize(req)} (~{dist} km)")  # plug SMTP/Resend here
    req.wave += 1
    if picked and req.status in ("Active", "Donors Contacted"): req.status = "Donors Contacted"
    db.commit(); return len(picked)

# ---------------- API ----------------
app = FastAPI(title="Smart Blood & Emergency Donor Network")

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=60); email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone: str = Field(pattern=r"^[0-9+\- ]{7,15}$"); password: str = Field(min_length=6, max_length=64)
    role: str = Field(pattern="^(requester|donor)$")  # coordinator/admin can never self-register
    blood_group: Optional[str] = None; lat: Optional[float] = None; lng: Optional[float] = None; city: Optional[str] = None

class LoginIn(BaseModel): email: str; password: str

class RequestIn(BaseModel):
    blood_group: str = Field(pattern="^(A|B|AB|O)[+-]$"); units_required: int = Field(ge=1, le=10)
    hospital: str; hours_needed: float = Field(gt=0, le=240); description: str = Field("", max_length=500)

@app.post("/api/register")
def register(b: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter_by(email=b.email).first(): raise HTTPException(400, "Email already used")
    if b.role == "donor" and b.blood_group not in COMPAT: raise HTTPException(400, "Valid blood group required")
    salt = secrets.token_hex(8)
    u = User(**b.model_dump(exclude={"password"}), pw=hash_pw(b.password, salt), salt=salt, token=secrets.token_hex(24))
    db.add(u); db.commit(); return {"token": u.token, "role": u.role, "name": u.name}

@app.post("/api/login")
def login(b: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter_by(email=b.email).first()
    if not u or u.pw != hash_pw(b.password, u.salt): raise HTTPException(401, "Wrong email or password")
    u.token = secrets.token_hex(24); db.commit()
    return {"token": u.token, "role": u.role, "name": u.name}

@app.get("/api/hospitals")
def hospitals(): return list(HOSPITALS)

@app.post("/api/requests")
def create_request(b: RequestIn, u: User = Depends(require("requester", "admin")), db: Session = Depends(get_db)):
    if b.hospital not in HOSPITALS: raise HTTPException(400, "Unknown hospital")
    lat, lng, city = HOSPITALS[b.hospital]; due = datetime.utcnow() + timedelta(hours=b.hours_needed)
    r = BloodRequest(requester_id=u.id, blood_group=b.blood_group, units_required=b.units_required, hospital=b.hospital,
                     city=city, lat=lat, lng=lng, required_before=due, description=b.description,
                     urgency=classify_priority(b.description, due))
    if is_duplicate(db, r): raise HTTPException(409, "Duplicate request detected (same hospital + blood group in 24h)")
    r.summary = summarize(r); db.add(r); db.commit(); return {"id": r.id, "urgency": r.urgency, "summary": r.summary}

def ser(r): return {c.name: (getattr(r, c.name).isoformat() if isinstance(getattr(r, c.name), datetime) else getattr(r, c.name))
                    for c in r.__table__.columns if c.name not in ("lat", "lng")}

@app.get("/api/requests")
def list_requests(blood_group: str = None, city: str = None, status: str = None, urgency: str = None,
                  u: User = Depends(current_user), db: Session = Depends(get_db)):
    q = db.query(BloodRequest)
    if u.role == "requester": q = q.filter_by(requester_id=u.id)  # ownership check (anti-IDOR)
    elif u.role == "donor": raise HTTPException(403, "Donors use /api/inbox")
    for k, v in dict(blood_group=blood_group, city=city, status=status, urgency=urgency).items():
        if v: q = q.filter(getattr(BloodRequest, k) == v)
    out = []
    for r in q.order_by(BloodRequest.id.desc()).all():
        d = ser(r); d["verified"] = r.verification_status == "verified"
        resp = []
        for x in db.query(DonorResponse).filter_by(request_id=r.id).order_by(DonorResponse.match_score.desc()):
            dn = db.get(User, x.donor_id)
            item = {"id": x.id, "donor": dn.name, "blood_group": dn.blood_group, "score": x.match_score, "km": x.distance_km,
                    "status": x.response_status, "confirmed": x.donation_confirmed}
            if x.response_status == "accepted": item["phone"] = dn.phone  # PRIVACY: contact only after accept
            resp.append(item)
        d["responses"] = resp; out.append(d)
    return out

@app.post("/api/requests/{rid}/verify")
def verify(rid: int, approve: bool = True, u: User = Depends(require("coordinator", "admin")), db: Session = Depends(get_db)):
    r = db.get(BloodRequest, rid)
    if not r or r.status != "Pending Verification": raise HTTPException(400, "Request not pending")
    if not approve: r.verification_status, r.status = "rejected", "Rejected"; db.commit(); return {"status": r.status}
    r.verification_status, r.status = "verified", "Active"; db.commit()
    return {"status": r.status, "notified": dispatch(db, r)}

@app.post("/api/requests/{rid}/escalate")
def escalate(rid: int, u: User = Depends(require("coordinator", "admin")), db: Session = Depends(get_db)):
    r = db.get(BloodRequest, rid)
    if not r or r.status not in ("Donors Contacted", "Partially Fulfilled", "Active"): raise HTTPException(400, "Cannot escalate")
    return {"notified": dispatch(db, r), "radius_km": r.radius_km, "wave": r.wave}

@app.get("/api/inbox")
def inbox(u: User = Depends(require("donor")), db: Session = Depends(get_db)):
    out = []
    for x in db.query(DonorResponse).filter_by(donor_id=u.id).order_by(DonorResponse.id.desc()):
        r = db.get(BloodRequest, x.request_id)
        if r.status in ("Cancelled", "Rejected", "Expired", "Completed", "Fulfilled") and x.response_status == "notified": continue  # auto-stop
        out.append({"id": x.id, "summary": r.summary, "urgency": r.urgency, "hospital": r.hospital, "km": x.distance_km,
                    "score": x.match_score, "status": x.response_status, "confirmed": x.donation_confirmed})
    return out

@app.post("/api/responses/{xid}/respond")
def respond(xid: int, accept: bool, u: User = Depends(require("donor")), db: Session = Depends(get_db)):
    x = db.get(DonorResponse, xid)
    if not x or x.donor_id != u.id: raise HTTPException(404, "Not found")  # ownership check
    x.response_status = "accepted" if accept else "declined"; x.response_time = datetime.utcnow(); db.commit()
    return {"status": x.response_status}

@app.post("/api/responses/{xid}/confirm")
def confirm(xid: int, u: User = Depends(require("coordinator", "admin")), db: Session = Depends(get_db)):
    x = db.get(DonorResponse, xid)
    if not x or x.response_status != "accepted" or x.donation_confirmed: raise HTTPException(400, "Cannot confirm")
    r = db.get(BloodRequest, x.request_id); d = db.get(User, x.donor_id)
    x.donation_confirmed = True; r.units_arranged += 1; d.total_donations += 1; d.last_donation = datetime.utcnow()
    if r.units_arranged >= r.units_required:
        r.status, r.completed_at = "Fulfilled", datetime.utcnow()
        for o in db.query(DonorResponse).filter_by(request_id=r.id, response_status="notified"): o.response_status = "declined"  # stop extra notifications
    else: r.status = "Partially Fulfilled"
    db.commit(); return {"status": r.status, "units_arranged": r.units_arranged}

@app.post("/api/requests/{rid}/complete")
def complete(rid: int, u: User = Depends(current_user), db: Session = Depends(get_db)):
    r = db.get(BloodRequest, rid)
    if not r or (u.role == "requester" and r.requester_id != u.id) or u.role == "donor": raise HTTPException(403, "Not allowed")
    if r.status != "Fulfilled": raise HTTPException(400, "Only fulfilled requests can be completed")
    r.status = "Completed"; db.commit(); return {"status": r.status}

@app.get("/api/stats")
def stats(u: User = Depends(require("coordinator", "admin")), db: Session = Depends(get_db)):
    rs = db.query(BloodRequest).all(); done = [r for r in rs if r.completed_at]
    avg = sum((r.completed_at - r.created_at).total_seconds() for r in done) / len(done) / 60 if done else 0
    times = [(x.response_time - x.notified_at).total_seconds() for x in db.query(DonorResponse).filter(DonorResponse.response_time != None)]  # noqa
    by = {}
    for r in rs: by[r.blood_group] = by.get(r.blood_group, 0) + 1
    return {"total": len(rs), "active": sum(r.status in ("Active", "Donors Contacted", "Partially Fulfilled") for r in rs),
            "completed": sum(r.status in ("Fulfilled", "Completed") for r in rs), "critical": sum(r.urgency == "Critical" for r in rs),
            "donors": db.query(User).filter_by(role="donor").count(),
            "available_donors": db.query(User).filter_by(role="donor", available=True).count(),
            "donations": sum(x.donation_confirmed for x in db.query(DonorResponse)),
            "avg_fulfillment_min": round(avg, 1), "avg_response_sec": round(sum(times) / len(times), 1) if times else 0,
            "fulfillment_rate_pct": round(100 * len(done) / len(rs), 1) if rs else 0, "demand_by_group": by}

@app.post("/api/seed")
def seed(db: Session = Depends(get_db)):
    if db.query(User).count(): return {"msg": "already seeded"}
    def mk(name, email, role, **kw):
        s = secrets.token_hex(8); db.add(User(name=name, email=email, role=role, phone=kw.pop("phone", "0300" + str(random.randint(1000000, 9999999))),
                                              salt=s, pw=hash_pw("demo123", s), **kw))
    mk("Admin", "admin@demo.com", "admin"); mk("Coordinator", "coord@demo.com", "coordinator"); mk("Requester Ali", "req@demo.com", "requester")
    groups = list(COMPAT); names = ["Ahmed", "Sara", "Bilal", "Hina", "Usman", "Ayesha", "Zain", "Maryam", "Hamza", "Noor"]
    for i in range(50):
        mk(f"{random.choice(names)} {i}", f"donor{i}@demo.com", "donor", blood_group=random.choice(groups),
           lat=33.68 + random.uniform(-0.08, 0.08), lng=73.05 + random.uniform(-0.08, 0.08),
           city="Islamabad", available=random.random() > 0.15,
           last_donation=datetime.utcnow() - timedelta(days=random.choice([10, 30, 100, 150, 300])) if random.random() > 0.3 else None)
    db.commit(); return {"msg": "seeded", "logins": "admin@demo.com / coord@demo.com / req@demo.com / donor0@demo.com  (pw: demo123)"}

@app.on_event("startup")
def auto_seed():
    db = SessionLocal()
    try:
        if not db.query(User).count(): seed(db)
    finally: db.close()

@app.get("/")
def home(): return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")
