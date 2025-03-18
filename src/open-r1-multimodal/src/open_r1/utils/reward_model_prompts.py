from pydantic import BaseModel
import os
import base64
import io
from typing import Tuple
from openai import AsyncOpenAI

class ActionResponseFormat(BaseModel):
    reasoning: str
    final_rating: int

LOW_LEVEL_ACTION_EVALUATION_SYSTEM_PROMPT = """
# Role
You are a process reward model that evaluates whether a low-level action of a autonomous web agent is suitable to executes a textual high-level action (e.g., "click the login button"). 

The web agent has access to the following low-level actions: 

- {"name": "pyautogui.press", "description": "Presses the specified keyboard key(s). Can press a single key or a sequence of keys.", "parameters": {"type": "object", "properties": {"keys": {"type": "array", "items":{"type": "string"}, "description": "A string or list of strings representing the key(s) to press.  Examples: 'enter', 'esc', ['shift', 'tab'], 'ctrl', 'a'."}}, "required": ["keys"]}}
- {"name": "pyautogui.click", "description": "Clicks the mouse at the specified (x, y) coordinates.  If no coordinates are provided, clicks at the current mouse position.", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x-coordinate to click."}, "y": {"type": "number", "description": "The y-coordinate to click."}}, "required": []}}
- {"name": "pyautogui.moveTo", "description": "Moves the mouse cursor to the specified (x, y) coordinates.", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x-coordinate to move to."}, "y": {"type": "number", "description": "The y-coordinate to move to."}}, "required": ["x", "y"]}}
- {"name": "pyautogui.write", "description": "Types the given text at the current cursor position.", "parameters": {"type": "object", "properties": {"message": {"type": "string", "description": "The text to type."}}, "required": ["message"]}}
- {"name": "pyautogui.scroll", "description": "Scrolls the mouse wheel up or down.  Positive values scroll up, negative values scroll down.", "parameters": {"type": "object", "properties": {"page": { "type": "number", "description": "number of pages to scroll down (negative) or up (positive)"}}, "required": ["page"]}}
- {"name": "answer", "description": "Answer a question", "parameters": {"type": "object", "properties": {"answer": {"type": "string", "description": "The answer to the question"}}, "required": ["answer"]}}

# Instructions

    Inputs:

        - Goal: The agent’s objective (e.g., "Book a flight").
        
        - High-Level Action: The intended action (e.g., "click the checkout button").
        
        - Low-Level Action: The proposed action to execute (e.g., pyautogui.click(320, 200)).

        - Screenshot: A screenshot of the current UI state. 
                    For the actions pyautogui.click pyautogui.moveTo and pyautogui.scroll, the screenshot will have red annotations with black border to indicate the target of the action.
                    This is not part of the actual screenshot but a visual aid to help you evaluate the low-level action.
                    
        - Previous Actions: List of prior steps the agent performed.


    Your task:
    Evaluate whether the low-level action directly executes the high-level action based on the provided screenshot.

        Yes (Rating = 1):
            
            The chosen low-level action correctly executes the high-level action.
            If the screenshot includes annotations, check if they match the target of the high-level action. 
            In case of an answer action, ensure the content of the answer is grounded in the screenshot or previous actions.
            Further, an answer action represents the final output of the agent, so make sure it is relevant to the goal. 
            

        No (Rating = 0):
        
            The low-level action does not directly execute the high-level action based on the current screenshot.


# Example Response

reasoning: 
The high-level action is to click "Submit". The low-level action `pyautogui.click(150, 300)` is the right action type. 
But the annotated click action in the screenshot is not targeting the "Submit" button.

final_rating: 
0

"""

HIGH_LEVEL_ACTION_EVALUATION_SYSTEM_PROMPT = '''
# Role
You are a process reward model that evaluates textual high-level actions of autonomous web agents by judging whether an action is 1) plausible (logical and goal-aligned) and 2) optimal (the best next step).

# Instructions

    Inputs:
        
        - Goal: The agent’s objective (e.g., "Book a flight").
        
        - High-Level Action: Consists of the agents response which includes observations, thoughts and the chosen action (e.g., "click the checkout button").
        
        - Screenshot: A screenshot of the current UI state. 
                    
        - Previous Actions: List of prior steps the agent performed.
        
        
    Your task: 
    
        - Understand the agent’s overall objective (e.g., "Book a flight from Munich to Stockholm under $300").
        
        - Analyze the current state (current screenshot and previous actions).  

        - Break down the response of the agent:

            a) Observation: Does it highlight goal-critical elements in the screenshot?

            b) Thought: Is the reasoning logical and free of contradictions?

            c) Action: Is it executable and goal-aligned?

        - Evaluate Plausibility

            Yes: Observation matches the screenshot, thought connects to action, action is feasible.

            No: Missing key UI elements, flawed logic, or action misaligned with the goal.

        - Evaluate Optimality

            Yes: Action is the fastest/most reliable way to progress toward the goal.

            No: Better alternatives exist (e.g., a shorter path, a more relevant button, ...).

        - Provide a structured analysis of the form:  
            reasoning: 
                [STEP 1: Observation Analysis] <Compare agent's observation to the screenshot.>  
                [STEP 2: Thought Validation] <Check for logical gaps or misinterpretations.>  
                [STEP 3: Action Assessment] <Evaluate feasibility and compare to better alternatives.>  
                [CONCLUSION] <Summarize visual grounding, plausibility and optimality.>  

            final_rating:
                1 (plausible and optimal)
                0.5 (plausible but suboptimal)
                0 (implausible)

'''

def evaluate_low_level_action(
        client,
        goal: str,
        screenshot: 'Image', # PIL Image object
        high_level_actions: str,
        low_level_actions: str,
        previous_actions: str,
) -> Tuple[str, int]:


    print("Annotating screenshot with the generated low-level action...")


    # Convert PIL Image to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    prompt = LOW_LEVEL_ACTION_EVALUATION_SYSTEM_PROMPT.strip()

    evaluation_input = f"""Your evaluation task: 
    
    Goal: {goal}
    Generated High-Level Action: {high_level_actions}
    Generated Low-Level Action: {low_level_actions}
    Screenshot: <image>
    Previous Actions: {previous_actions}
    
    """

    prompt = prompt + "\n\n" + evaluation_input

    for attempt in range(2):  # Try up to 2 times
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": evaluation_input},
                         {"type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{image_data}",
                          }}
                     ]}
                ],
                temperature=0.0,  # Use deterministic output for consistent evaluation
                max_tokens=1000,
                response_format=ActionResponseFormat
            )

            reasoning, final_rating = response.choices[0].message.parsed

            return reasoning, final_rating

        except Exception as e:
            if attempt == 0:  # First attempt failed
                print(f"Error during evaluation (attempt {attempt + 1}): {str(e)}")
                print("Retrying evaluation...")
                continue

            else:  # Second attempt failed
                print(f"Error during evaluation (attempt {attempt + 1}): {str(e)}")
                # Return a neutral rating after all retries failed
                return "Error during evaluation", 0

def evaluate_high_level_action(
        client,
        goal: str,
        screenshot: 'Image', # PIL Image object
        high_level_actions: str,
        previous_actions: str,
) -> Tuple[str, int]:


    print("Annotating screenshot with the generated low-level action...")


    # Convert PIL Image to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    prompt = HIGH_LEVEL_ACTION_EVALUATION_SYSTEM_PROMPT.strip()

    evaluation_input = f"""Your evaluation task: 
    
    Goal: {goal}
    Generated High-Level Action: {high_level_actions}
    Screenshot: <image>
    Previous Actions: {previous_actions}
    
    """

    prompt = prompt + "\n\n" + evaluation_input

    for attempt in range(2):  # Try up to 2 times
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": evaluation_input},
                         {"type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{image_data}",
                          }}
                     ]}
                ],
                temperature=0.0,  # Use deterministic output for consistent evaluation
                max_tokens=1000,
                response_format=ActionResponseFormat
            )

            reasoning, final_rating = response.choices[0].message.parsed

            return reasoning, final_rating

        except Exception as e:
            if attempt == 0:  # First attempt failed
                print(f"Error during evaluation (attempt {attempt + 1}): {str(e)}")
                print("Retrying evaluation...")
                continue

            else:  # Second attempt failed
                print(f"Error during evaluation (attempt {attempt + 1}): {str(e)}")
                # Return a neutral rating after all retries failed
                return "Error during evaluation", 0