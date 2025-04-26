from fastapi import FastAPI, Request, Form  # Import FastAPI core classes
from fastapi.responses import HTMLResponse  # Used to send back HTML pages
from fastapi.staticfiles import StaticFiles  # Serve static files like CSS
from fastapi.templating import Jinja2Templates  # Template engine for HTML
import joblib  # Used to load the trained ML model
import os  # To handle file paths
import pandas as pd  # For creating DataFrame to feed into model

# Initialize FastAPI app
app = FastAPI()

# Define base directory and path to the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../model/fake_news_classifier.pkl")

# Mount the static directory to serve CSS/JS/images under "/static"
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Set up Jinja2 template directory for HTML rendering
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load trained fake news classifier model
model = joblib.load(model_path)

# Route: GET request to homepage - render the input form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route: POST request for prediction
@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    title: str = Form(None),   # Get 'title' from form input
    content: str = Form(None)  # Get 'content' from form input
):
    # Check if both inputs are provided
    if not (title and content):
        result = "❌ Please provide both title and content!"
    else:
        # Format the input as a DataFrame for the model
        input_df = pd.DataFrame([{"title": title, "text": content}])
        prediction = model.predict(input_df)[0]  # Get prediction from model
        # Display corresponding result
        result = "🧠 IT'S REALLLLL!" if prediction == "true" else "🚨 IT'S FAKEEE!"

    # Return the page with the prediction result and original inputs
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "title": title,
        "content": content
    })
