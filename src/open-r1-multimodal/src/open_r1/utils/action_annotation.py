"""
Annotate the screenshot with the generated low-level action.

Available actions:
- {"name": "pyautogui.press", "description": "Presses the specified keyboard key(s). Can press a single key or a sequence of keys.", "parameters": {"type": "object", "properties": {"keys": {"type": "array", "items":{"type": "string"}, "description": "A string or list of strings representing the key(s) to press.  Examples: 'enter', 'esc', ['shift', 'tab'], 'ctrl', 'a'."}}, "required": ["keys"]}}
- {"name": "pyautogui.click", "description": "Clicks the mouse at the specified (x, y) coordinates.  If no coordinates are provided, clicks at the current mouse position.", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x-coordinate to click."}, "y": {"type": "number", "description": "The y-coordinate to click."}}, "required": []}}
- {"name": "pyautogui.moveTo", "description": "Moves the mouse cursor to the specified (x, y) coordinates.", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x-coordinate to move to."}, "y": {"type": "number", "description": "The y-coordinate to move to."}}, "required": ["x", "y"]}}
- {"name": "pyautogui.write", "description": "Types the given text at the current cursor position.", "parameters": {"type": "object", "properties": {"message": {"type": "string", "description": "The text to type."}}, "required": ["message"]}}
- {"name": "pyautogui.scroll", "description": "Scrolls the mouse wheel up or down.  Positive values scroll up, negative values scroll down.", "parameters": {"type": "object", "properties": {"page": { "type": "number", "description": "number of pages to scroll down (negative) or up (positive)"}}, "required": ["page"]}}
- {"name": "answer", "description": "Answer a question", "parameters": {"type": "object", "properties": {"answer": {"type": "string", "description": "The answer to the question"}}, "required": ["answer"]}}

Available annotations:
- Click: Draws a red circle at the specified coordinates.
- Move: Draws a blue circle at the specified coordinates.
- Scroll: Draws an arrow at the specified coordinates.

Author: @korbinian-hoermann
"""

from PIL.Image import Image
from PIL.ImageDraw import ImageDraw, Draw
from PIL import Image as I
from typing import Tuple
import re
import math


def convert_scroll(action):
    """
    Convert the scroll action of the form: pyautogui.scroll(page=-0.28) to x and y coordinates.
    """
    # Extract the page value
    match = re.search(r"page=(-?\d+\.\d+)", action)
    if match is None:
        return None, None

    page = float(match.group(1)) * 720

    # Positive page value means scroll up (negative y)
    # Negative page value means scroll down (positive y)
    x = 0
    y = -page  # Invert the direction

    return x, y



def annotate_action(actions: str, screenshot: Image) -> Image:
    """
    Annotate the screenshot with the generated low-level action.
    """
    print(f"Annotating screenshot with the generated low-level action...")
    print(f"Actions: {actions}")

    for action in actions.split("\n"):
        print(f"\tCreating annotation for action: {action}")
        if action.startswith("pyautogui.click"):
            x, y = extract_x_y(action)
            screenshot = annotate_click(x, y, screenshot)
        elif action.startswith("pyautogui.moveTo"):
            x, y = extract_x_y(action)
            screenshot = annotate_move(x, y, screenshot)
        elif action.startswith("pyautogui.scroll"):
            x, y = convert_scroll(action)
            screenshot = annotate_scroll(x, y, screenshot)

    return screenshot

def extract_x_y(action: str) -> Tuple[int, int]:
    """
    Extract the x and y coordinates from the action.
    """

    # Extract the x and y coordinates
    pattern = r"pyautogui\.click\s*\(\s*x\s*=\s*([-+]?\d*\.?\d+)\s*,\s*y\s*=\s*([-+]?\d*\.?\d+)\s*\)"
    match = re.search(pattern, action)
    if match is None:
        return None, None

    x, y = match.groups()

    return x, y

def annotate_click(x, y, image):
    """
    Draw a point on the image to annotate click action.
    :param image:
    :param point:
    :return:
    """

    print(f"Annotating click at ({x}, {y})")
    x, y = float(x), float(y)

    radius = min(image.width, image.height) // 15

    # scale x, y to 0-1000 range depending on image size
    x = int(x * image.width)
    y = int(y * image.height)

    # Draw a red circle at the specified coordinates
    print(f"Drawing red circle at ({x}, {y})")

    Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), outline='red', width=2)
    Draw(image).ellipse((x - 2, y - 2, x + 2, y + 2), fill='red')

    return image

def annotate_move(x: int, y: int, screenshot: Image) -> Image:
    """
    Annotate the screenshot with the move action by drawing a blue circle at the specified coordinates.
    """

    # Draw a blue circle at the specified coordinates
    draw = ImageDraw(screenshot)
    draw.circle((x, y), 2, fill="blue")
    draw.ellipse((x-10, y-10, x+10, y+10), outline="blue")
    draw.text((x+10, y+10), "MoveTo", fill="blue")

    return screenshot

def annotate_scroll(x, y, screenshot: Image) -> Image:
    """
    Annotate the screenshot with the scroll action by drawing an arrow.
    """
    center_x, center_y = 1200/2, 720/2
    end_x, end_y = center_x + x, center_y + y

    draw = ImageDraw(screenshot)
    # Draw the main arrow line
    draw.line((center_x, center_y, end_x, end_y), fill="black", width=2)

    # Calculate arrow head position and direction
    arrow_length = 15
    arrow_width = 8

    # Calculate angle of the line
    angle = math.atan2(y, x)

    # Calculate arrow head points
    left_x = end_x - arrow_length * math.cos(angle + math.pi/6)
    left_y = end_y - arrow_length * math.sin(angle + math.pi/6)
    right_x = end_x - arrow_length * math.cos(angle - math.pi/6)
    right_y = end_y - arrow_length * math.sin(angle - math.pi/6)

    # Draw arrow head
    draw.line((end_x, end_y, left_x, left_y), fill="black", width=2)
    draw.line((end_x, end_y, right_x, right_y), fill="black", width=2)

    # Add label
    draw.text((end_x+10, end_y+10), f"Scroll page ({y/720})", fill="black")

    return screenshot

