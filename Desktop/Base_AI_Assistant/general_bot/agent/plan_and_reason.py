import openai

def generate_next_action(conversation_history: list) -> str:
    """
    Uses a generative language model (e.g., GPT-4) to determine the next action or response.
    """
    # Construct the prompt based on the conversation history
    prompt = "You are an AI agent in a phone call. Respond based on the following conversation:\n"
    for turn in conversation_history:
        if "user" in turn:
            prompt += f"User: {turn['user']}\n"
        elif "agent" in turn:
            prompt += f"AI: {turn['agent']}\n"
    
    prompt += "AI:"
    
    # Call GPT-4 or a similar model to generate the response
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()