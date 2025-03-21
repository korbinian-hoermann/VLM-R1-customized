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


def convert_scroll(action, screenshot):
    """
    Convert a scroll action string into (x, y) pixel offsets.

    Supports scroll actions specified as key/value pairs, e.g.:
      pyautogui.scroll(page=-0.65)
      pyautogui.scroll(page=-0.65, horizontal=0.15)

    or as positional arguments, e.g.:
      pyautogui.scroll(-0.65, 0.15)

    Args:
        action (str): The scroll action string.
        screenshot: An image-like object with 'width' and 'height' attributes.

    Returns:
        tuple: (x, y) pixel offsets where x is horizontal scroll and y is vertical scroll.
    """
    # Try to extract key/value pairs (e.g., "page=-0.65", "horizontal=0.15")
    pattern = r"(\w+)\s*=\s*(-?\d+\.\d+)"
    kv_pairs = dict(re.findall(pattern, action))

    if kv_pairs:
        # Use the "page" parameter for vertical scrolling (defaulting to 0 if not provided)
        vertical = float(kv_pairs.get('page', 0))
        # Typically vertical scroll is scaled by the image height.
        y = -vertical * screenshot.height  # Invert to match scroll direction

        # Support horizontal scrolling specified as "horizontal" or shorthand "h"
        horizontal_str = kv_pairs.get('horizontal', kv_pairs.get('h', None))
        if horizontal_str is not None:
            x = float(horizontal_str) * screenshot.width
        else:
            x = 0
        return x, y
    else:
        # Fall back to positional numbers, assuming the first is vertical and the second (if exists) is horizontal.
        numbers = re.findall(r"-?\d+\.\d+", action)
        if not numbers:
            return 0, 0

        vertical = float(numbers[0])
        y = -vertical * screenshot.height

        if len(numbers) > 1:
            horizontal = float(numbers[1])
            x = horizontal * screenshot.width
        else:
            x = 0
        return x, y


def annotate_action(actions: str, screenshot: Image) -> Tuple[Image, bool]:
    """
    Annotate the screenshot with the generated low-level action.
    """
    print(f"Annotating screenshot with the generated low-level action...")
    print(f"Actions: {actions}")

    annotated_screenshot = screenshot

    for action in actions.split("\n"):
        print(f"\tCreating annotation for action: {action}")
        if action.startswith("pyautogui.click"):
            x, y = extract_x_y(action)
            annotated_screenshot, success = annotate_click(x, y, screenshot)
        elif action.startswith("pyautogui.moveTo"):
            x, y = extract_x_y(action)
            annotated_screenshot, success = annotate_move(x, y, screenshot)
        elif action.startswith("pyautogui.scroll"):
            x, y = convert_scroll(action, screenshot)
            annotated_screenshot, success = annotate_scroll(x, y, screenshot)

    return annotated_screenshot, success


def extract_x_y(action: str) -> Tuple[int, int]:
    """
    Extract the x and y coordinates from the action.
    """

    # Extract the x and y coordinates
    pattern = r"pyautogui\.(?:click|moveTo)\s*\(\s*x\s*=\s*([-+]?\d*\.?\d+)\s*,\s*y\s*=\s*([-+]?\d*\.?\d+)\s*\)"
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
    success = False
    try:
        color = "red"
        action = "Click"
        print(f"Annotating click at ({x}, {y})")
        x, y = float(x), float(y)

        radius = min(image.width, image.height) // 15
        outer_halo_radius = radius + 2  # Increase the radius for the halo

        # scale x, y to 0-1000 range depending on image size
        x = int(x * image.width)
        y = int(y * image.height)

        # Draw a red circle at the specified coordinates
        print(f"Drawing red circle at ({x}, {y})")

        # Draw outer circle
        Draw(image).ellipse((x - outer_halo_radius, y - outer_halo_radius, x + outer_halo_radius, y + outer_halo_radius),
                            outline="black", width=4)
        Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)

        # Draw inner circle
        Draw(image).ellipse((x - 4, y - 4, x + 4, y + 4), outline="black", width=4)
        Draw(image).ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        # Add label to the circle, indicating the action. make sure the label is visible (inside the image)
        if x + 10 < image.width and y + 10 < image.height:
            Draw(image).text((x + 10, y + 10), action, fill=color, stroke_fill="black", stroke_width=2)

        elif x - 10 > 0 and y - 10 > 0:
            Draw(image).text((x - 10, y - 10), action, fill=color, stroke_fill="black", stroke_width=2)

        elif x + 10 < image.width and y - 10 > 0:
            Draw(image).text((x + 10, y - 10), action, fill=color, stroke_fill="black", stroke_width=2)

        elif x - 10 > 0 and y + 10 < image.height:
            Draw(image).text((x - 10, y + 10), action, fill=color, stroke_fill="black", stroke_width=2)
        success = True
    except Exception as e:
        print(f"Error: {e}")


    return image, success


def annotate_move(x: int, y: int, image: Image) -> Image:
    """
    Annotate the screenshot with the move action by drawing a blue circle at the specified coordinates.
    """

    success = False
    try:
        color = "red"
        action = "MoveTo"
        print(f"Annotating moveTo at ({x}, {y})")
        x, y = float(x), float(y)

        radius = min(image.width, image.height) // 15
        outer_halo_radius = radius + 2  # Increase the radius for the halo

        # scale x, y to 0-1000 range depending on image size
        x = int(x * image.width)
        y = int(y * image.height)

        # Draw a red circle at the specified coordinates
        print(f"Drawing blue circle at ({x}, {y})")

        # Draw outer circle
        Draw(image).ellipse((x - outer_halo_radius, y - outer_halo_radius, x + outer_halo_radius, y + outer_halo_radius),
                            outline="black", width=4)
        Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)

        # Draw inner circle
        Draw(image).ellipse((x - 4, y - 4, x + 4, y + 4), outline="black", width=4)
        Draw(image).ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        # Add label to the circle, indicating the action. make sure the label is visible (inside the image)
        if x + 10 < image.width and y + 10 < image.height:
            Draw(image).text((x + 10, y + 10), action, fill=color, stroke_fill="black", stroke_width=2)

        elif x - 10 > 0 and y - 10 > 0:
            Draw(image).text((x - 10, y - 10), action, fill=color, stroke_fill="black", stroke_width=2)

        elif x + 10 < image.width and y - 10 > 0:
            Draw(image).text((x + 10, y - 10), action, fill=color, stroke_fill="black", stroke_width=2)

        elif x - 10 > 0 and y + 10 < image.height:
            Draw(image).text((x - 10, y + 10), action, fill=color, stroke_fill="black", stroke_width=2)
        success = True
    except Exception as e:
        print(f"Error: {e}")

    return image, success


def compute_scaling_factor(cx, cy, dx, dy, width, height, margin=15):
    factors = []
    # For horizontal direction
    if dx > 0:
        factors.append((width - margin - cx) / dx)
    elif dx < 0:
        factors.append((cx - margin) / abs(dx))
    else:
        factors.append(float('inf'))
    # For vertical direction
    if dy > 0:
        factors.append((height - margin - cy) / dy)
    elif dy < 0:
        factors.append((cy - margin) / abs(dy))
    else:
        factors.append(float('inf'))
    # Do not scale up if the arrow is already within bounds
    return min(1, *factors)


def annotate_scroll(x, y, screenshot: Image) -> Image:
    """
    Annotate the screenshot with the scroll action by drawing an arrow
    with layered lines (an outline) to ensure visibility.
    """

    success = False
    try:
        margin = 15
        # Use the proper center of the image: (width/2, height/2)
        center_x, center_y = screenshot.width // 2, screenshot.height // 2

        # Calculate the scaling factor to ensure the arrow stays inside the image boundaries
        factor = compute_scaling_factor(center_x, center_y, x, y, screenshot.width, screenshot.height, margin)
        end_x = center_x + x * factor
        end_y = center_y + y * factor

        # Create a drawing context
        draw = ImageDraw(screenshot)

        # Calculate arrow head position and direction
        arrow_length = 15
        angle = math.atan2(y * factor, x * factor)
        left_x = end_x - arrow_length * math.cos(angle + math.pi / 6)
        left_y = end_y - arrow_length * math.sin(angle + math.pi / 6)
        right_x = end_x - arrow_length * math.cos(angle - math.pi / 6)
        right_y = end_y - arrow_length * math.sin(angle - math.pi / 6)

        # Set colors and line widths for layered drawing
        outline_color = "black"  # Outline color to contrast with background
        main_color = "red"     # Main arrow color
        outline_width = 4        # Thicker width for the outline
        main_width = 2           # Thinner width for the main arrow

        # --- Draw the outline (layered background) ---
        draw.line((center_x, center_y, end_x, end_y), fill=outline_color, width=outline_width)
        draw.line((end_x, end_y, left_x, left_y), fill=outline_color, width=outline_width)
        draw.line((end_x, end_y, right_x, right_y), fill=outline_color, width=outline_width)

        # --- Draw the main arrow on top ---
        draw.line((center_x, center_y, end_x, end_y), fill=main_color, width=main_width)
        draw.line((end_x, end_y, left_x, left_y), fill=main_color, width=main_width)
        draw.line((end_x, end_y, right_x, right_y), fill=main_color, width=main_width)

        # --- Add a label with text outline for extra readability ---
        # Using stroke parameters (available in newer Pillow versions)
        draw.text(
            (center_x + 10, center_y + 10),
            f"Scroll page ({y / screenshot.width:.2f})",
            fill=main_color,
            stroke_width=2,
            stroke_fill=outline_color
        )
        success = True
    except Exception as e:
        print(f"Error: {e}")

    return screenshot, success