from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
# from app.routers.single_course import single_course_predict_router
from app.routers.students_classification import classify_student_router

app = FastAPI()
app.title = "Sofia VAK Classification API"
app.version = "0.0.1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify_student_router)

@app.get('/', tags = ['home'])
async def home():
    htmlBody = """
    <h1>Welcome to Sofia predictions API</h1>
    <h2>API Documentation</h2>
    <p>API documentation is available at <a href="/docs">/docs</a></p>
    <p>API documentation is available at <a href="/redoc">/redoc</a></p>
    """
    return HTMLResponse(content = htmlBody, status_code = 200)