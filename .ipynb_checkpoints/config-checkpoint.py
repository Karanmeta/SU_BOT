import os
os.environ["GEMINI_API_KEY"] = "AIzaSyD31xGEwYEtV4V2kzXmrvC5qiix_vtzank"
os.environ["TAVILY_API_KEY"] = "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"

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
