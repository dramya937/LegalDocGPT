import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def analyze_clauses(contract_text):
    prompt = f"Analyze this contract and extract key clauses with risk levels:\n\n{contract_text}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    analysis = response['choices'][0]['message']['content']
    return analysis
