"""Entry point for the dashboard: loads .env, then serves the app."""
from dotenv import load_dotenv
load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("fda_hazard.app:app", host="127.0.0.1", port=8000, reload=False)
