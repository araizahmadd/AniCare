import streamlit as st
from groq import Groq
import json

# Initialize the Groq client (replace with your actual API key)
client = Groq(api_key="Enter your groq key")

# Function to call Groq LLaMA model and parse the prompt
def parse_prompt_with_groq(prompt):
    """
    Parse user prompt into actionable tasks using Groq's LLaMA model.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""{prompt}"""
            }
        ],
        model="llama3-8b-8192",  # Model to be used
        stream=False,
    )
    result = chat_completion.choices[0].message.content
    # print("Groq LLaMA Output:", result)
    return result
# Streamlit interface
st.title("Image Processing Workflow")

# User input for the prompt
user_prompt = st.text_input("Enter your prompt:")

# Process button
if st.button("Process"):
    if user_prompt:
        # Call the LLaMA model to parse the prompt and get segmentation and inpainting tasks
        parsed_tasks = parse_prompt_with_groq(user_prompt)
        
        if "error" in parsed_tasks:
            st.error(parsed_tasks["error"])
        else:
            # Display the JSON output
            st.text(parsed_tasks)
    else:
        st.error("Please provide a prompt!")
