import math
import re
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image
from PIL.Image import Image as ImageObject


# process_image is adapted from EasyR1 (https://github.com/hiyouga/EasyR1).
# Copyright 2024 Bytedance Ltd. and/or its affiliates. Licensed under the Apache License, Version 2.0.
def process_image(
    image: Union[Dict[str, Any], ImageObject, str, bytes],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> ImageObject:
    """Load an image and resize it so that its pixel count lies in [min_pixels, max_pixels]."""
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def extract_last_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...} in `text`, matching nested braces."""
    if not isinstance(text, str) or not text:
        return None

    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not boxed_starts:
        return None

    start_idx = boxed_starts[-1] + len("\\boxed{")
    balance = 1
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            balance += 1
        elif text[i] == "}":
            balance -= 1
        if balance == 0:
            return text[start_idx:i]

    return None


def clean_latex_commands(text: str) -> str:
    """Strip text-style LaTeX wrappers, e.g. \\text{B} or \\mathbf{B} -> B."""
    if not text:
        return ""
    while True:
        prev_text = text
        text = re.sub(
            r"\\(text|textbf|textit|mathrm|mathbf|bf|it|rm|sf)\{([^{}]*)\}", r"\2", text, flags=re.IGNORECASE
        )
        if prev_text == text:
            break
    return text.strip()


def check_correctness(model_output: str, ground_truth: str) -> Tuple[bool, str]:
    """Compare the last \\boxed{} answer in `model_output` with `ground_truth`.

    Returns:
        (is_correct, parsed_answer)
    """
    raw_prediction = extract_last_boxed_answer(model_output)
    if raw_prediction is None:
        return False, "None"

    cleaned_content = clean_latex_commands(raw_prediction)

    target_answer = str(ground_truth).strip().lower()
    target_answer = re.sub(r"[\(\)\.]", "", target_answer)  # e.g. "(B)" -> "b"

    pred_lower = cleaned_content.lower()

    # Multiple-choice answers such as "A", "(B)", "C."
    option_match = re.search(r"^[\(\s]*([a-z])([\)\.\s:]|$)", pred_lower)
    if option_match:
        final_prediction = option_match.group(1)
    else:
        final_prediction = re.sub(r"[\(\)\.]", "", pred_lower).strip()

    if len(target_answer) == 1:
        return final_prediction == target_answer, final_prediction.upper()

    clean_raw_pred = re.sub(r"[\(\)\.]", "", pred_lower).strip()
    return clean_raw_pred == target_answer, cleaned_content
