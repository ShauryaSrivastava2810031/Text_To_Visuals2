import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(question, prompt):
    """
    Sends a natural language question to Gemini Pro and retrieves an SQL query response.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt[0], question])

        # Clean the response to remove undesired characters
        clean_response = response.text.replace("```", "").strip()
        return clean_response
    except Exception as e:
        return f"Error in Gemini API call: {str(e)}"