import requests
import openai
import os
import json
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pathlib import Path
from datetime import datetime

class GeneralistAgent:
    def __init__(
        self,
        api_key: str,
        call_api_url: str,
        model_name="gpt-4-turbo",
        temperature=0.7,
        request_timeout=120,
        ckpt_dir="ckpt",
        resume=False
    ):
        # Initialize OpenAI API key, call tool endpoint, and conversation memory
        self.api_key = api_key
        self.call_api_url = call_api_url
        openai.api_key = api_key

        # LangChain model setup
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=2000
        )

        # Conversation memory directory setup
        self.ckpt_dir = Path(ckpt_dir) / "conversation"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\033[34mEnsured directory exists: {self.ckpt_dir}\033[0m")

        # Resume conversation history if enabled
        if resume:
            memory_file = self.ckpt_dir / "conversation_memory.json"
            if memory_file.exists():
                print(f"\033[32mLoading previous conversation memory from {memory_file}\033[0m")
                self.conversation_memory = self._load_json(memory_file)
            else:
                self.conversation_memory = []
                print(f"\033[31mNo previous conversation found. Initializing new conversation memory.\033[0m")
        else:
            self.conversation_memory = []

    def _load_json(self, file_path):
        """Load JSON from a file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _save_json(self, data, file_path):
        """Save JSON to a file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def make_call(self, user_phone: str, instruction: str) -> str:
        """
        Initiates a phone call by sending a POST request to the Call Tool API.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "userPhone": user_phone,
            "instruction": instruction
        }
        try:
            response = requests.post(self.call_api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return f"Call successfully initiated with {user_phone}: {response.json()}"
        except requests.exceptions.RequestException as e:
            return f"Failed to initiate call: {e}"

    def reason_and_generate_response(self, user_input: str) -> str:
        """
        Uses LangChain and OpenAI to generate the next response based on user input and conversation history.
        """
        # Add the user input to the conversation history
        self.conversation_memory.append({"user": user_input})

        # Prepare prompt with conversation history
        prompt = "You are an AI engaged in a phone conversation. Respond appropriately based on the context:\n"
        for entry in self.conversation_memory:
            if "user" in entry:
                prompt += f"User: {entry['user']}\n"
            elif "agent" in entry:
                prompt += f"AI: {entry['agent']}\n"

        # Generate AI response
        response = self.llm([HumanMessage(content=prompt)])
        ai_response = response.content.strip()

        # Add AI response to the conversation history
        self.conversation_memory.append({"agent": ai_response})

        # Save conversation memory to a checkpoint
        self._save_json(self.conversation_memory, self.ckpt_dir / "conversation_memory.json")

        return ai_response

    def start_phone_conversation(self, user_phone: str, instruction: str):
        """
        Starts a phone conversation, initiates the call, and handles the back-and-forth interaction.
        """
        # Initiate the phone call
        print(self.make_call(user_phone, instruction))

        # Simulate a phone conversation loop
        while True:
            user_input = input("User says: ")  # Replace with real-time transcriptions in production
            if user_input.lower() in ["exit", "quit"]:
                print("Conversation ended.")
                break

            # Generate AI response
            ai_response = self.reason_and_generate_response(user_input)

            # Simulate sending the response back via Text-to-Speech
            print(f"AI says: {ai_response}")


if __name__ == "__main__":
    # Load API key and endpoint from environment variables or replace with hardcoded values
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
    CALL_API_URL = "https://calltool2-684089940295.us-west2.run.app/api/create-call"

    agent = GeneralistAgent(
        api_key=OPENAI_API_KEY,
        call_api_url=CALL_API_URL,
        model_name="gpt-4",
        resume=True  # Resuming from saved conversation history if available
    )

    # Example: Engaging in a rap battle with the user
    agent.start_phone_conversation(user_phone="+15106128131", instruction="You are to engage in a rap battle with the user.")


# import requests
# import openai

# class GeneralistAgent:
#     def __init__(self, api_key: str, call_api_url: str):
#         # Initialize OpenAI API key and Call Tool API endpoint
#         self.api_key = api_key
#         self.call_api_url = call_api_url
#         openai.api_key = api_key

#     def make_call(self, user_phone: str, instruction: str) -> str:
#         """
#         Initiates a phone call by sending a POST request to the Call Tool API.
#         """
#         headers = {"Content-Type": "application/json"}
#         payload = {
#             "userPhone": user_phone,
#             "instruction": instruction
#         }
#         try:
#             response = requests.post(self.call_api_url, json=payload, headers=headers)
#             response.raise_for_status()  # Raise an exception for HTTP errors
#             return f"Call successfully initiated with {user_phone}: {response.json()}"
#         except requests.exceptions.RequestException as e:
#             return f"Failed to initiate call: {e}"

#     def transcribe_audio(self, audio_url: str) -> str:
#         """
#         Simulate transcription of audio using OpenAI Whisper or a placeholder logic.
#         Replace this with actual speech-to-text service calls.
#         """
#         # Placeholder for demonstration purposes
#         return "This is the recognized user input from the audio."

#     def reason_and_generate_response(self, user_input: str, conversation_history: list) -> str:
#         """
#         Uses GPT-based reasoning to generate the next response based on conversation history.
#         """
#         prompt = "You are an AI agent engaged in a phone conversation. Respond appropriately:\n"
#         for entry in conversation_history:
#             if "user" in entry:
#                 prompt += f"User: {entry['user']}\n"
#             elif "agent" in entry:
#                 prompt += f"AI: {entry['agent']}\n"

#         prompt += f"User: {user_input}\nAI:"
        
#         response = openai.Completion.create(
#             engine="gpt-4",
#             prompt=prompt,
#             max_tokens=150,
#             temperature=0.7
#         )
#         return response.choices[0].text.strip()

#     def simulate_phone_conversation(self, user_phone: str, instruction: str):
#         """
#         Simulates an interactive phone conversation with reasoning and responses.
#         """
#         # Step 1: Initiate the call using the Call Tool API
#         print(self.make_call(user_phone, instruction))

#         # Step 2: Start the conversation loop
#         conversation_history = []
#         while True:
#             user_input = input("User says: ")  # Simulating user input (replace with actual transcribed input)
#             if user_input.lower() in ["exit", "quit"]:
#                 print("Conversation ended.")
#                 break

#             # Append user input to conversation history
#             conversation_history.append({"user": user_input})

#             # Step 3: Generate the AI's response
#             ai_response = self.reason_and_generate_response(user_input, conversation_history)
#             conversation_history.append({"agent": ai_response})

#             # Step 4: Simulate sending the response back via Text-to-Speech
#             print(f"AI says: {ai_response}")

# if __name__ == "__main__":
#     # Replace with your actual API key and endpoint
#     OPENAI_API_KEY = "your_openai_api_key"
#     CALL_API_URL = "https://calltool2-684089940295.us-west2.run.app/create-call"
    
#     agent = GeneralistAgent(api_key=OPENAI_API_KEY, call_api_url=CALL_API_URL)

#     # Example: Engaging in a rap battle with the user
#     agent.simulate_phone_conversation(user_phone="+15106128131", instruction="You are to engage in a rap battle with the user.")
