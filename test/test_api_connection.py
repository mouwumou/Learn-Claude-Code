"""
This module contains tests for the API connection functionality of the Anthropic client. It verifies that the client can successfully connect to the API and handle responses correctly.

The function call will return the current working directory, which is expected to be the same as the one where this script is located. The test will print the API response and confirm that the connection is successful. If there are any issues with the connection, it will print an error message and fail the test.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

def test_api_connection(messages):
    """
    Test that the Anthropic client can successfully connect to the API and receive a response.
    """

    # Initialize the Anthropic client
    client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
    MODEL = os.environ["MODEL_ID"]


    SYSTEM = f"You are a coding agent at {os.getcwd()}."

    # Make a simple API call to verify connection
    try:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            max_tokens=1000,
        )
        assert response is not None, "API response should not be None"
        print("API connection test passed, received response:", response)
        
        messages.append({"role": "assistant", "content": response.content})

    except Exception as e:
        print(f"API connection test failed: {e}")
        assert False, f"API connection test failed: {e}"


if __name__ == "__main__":
    # Example message for testing
    history = []
    history.append({"role": "user", "content": "What is the current working directory?"})

    test_api_connection(history)    

    response_content = history[-1]["content"]
    if isinstance(response_content, list):
        for block in response_content:
            if hasattr(block, "text"):
                print(block.text)
    print()