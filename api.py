from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from environment import CustomerSupportEnv
from models import Action

# Initialize the FastAPI application
app = FastAPI(title="Customer Support OpenEnv API")

# Create a single, persistent instance of our environment
env_instance = CustomerSupportEnv()

@app.post("/reset")
def reset_environment():
    """Starts a new episode and returns the initial observation."""
    try:
        obs = env_instance.reset()
        # FastAPI automatically converts Pydantic models to JSON!
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    """Takes an action from the AI and advances the environment by one step."""
    # Prevent the AI from taking a step if the game hasn't started
    if env_instance.obs is None:
        raise HTTPException(status_code=400, detail="You must call /reset before calling /step.")
    
    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_current_state():
    """Returns the current observation without taking a step."""
    if env_instance.obs is None:
        raise HTTPException(status_code=400, detail="Environment has not been initialized. Call /reset.")
    
    return {"observation": env_instance.state()}

@app.get("/", response_class=HTMLResponse)
def home():
    """Provides a friendly homepage so visitors don't get a Not Found error."""
    return """
    <html>
        <head>
            <title>Support Buddy OpenEnv</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding-top: 50px; background-color: #f4f4f9; }
                .container { background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); display: inline-block; }
                h1 { color: #333; }
                a { display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #10a37f; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
                a:hover { background-color: #0d8a6a; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Customer Support OpenEnv API</h1>
                <p>Welcome to our Hackathon Submission! The server is running perfectly.</p>
                <p>Use the button below to view the interactive API endpoints.</p>
                <a href="/docs">View API Documentation</a>
            </div>
        </body>
    </html>
    """