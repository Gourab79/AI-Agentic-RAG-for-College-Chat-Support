import os
import agentops
from groq import Groq

# Initialize AgentOps
ao_client = agentops.Client()
print(ao_client)
# @ao_client.record_action('query_groq')
def query_groq(prompt: str, model: str = "llama3-8b-8192") -> str:
    """Query Groq API with the given prompt"""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Simple Groq query interface"""
    print("Groq Query Interface (type 'exit' to quit)")
    
    while True:
        user_input = input("\nEnter your question: ")
        
        if user_input.lower() == 'exit':
            break
        
        response = query_groq(user_input)
        print("\nResponse:", response)

if __name__ == "__main__":
    try:
        main()
    finally:
        agentops.end_session('Success')