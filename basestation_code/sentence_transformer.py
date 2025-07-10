import re
from sentence_transformers import SentenceTransformer, util
import torch

# Load sentence-level embedding model 
model = SentenceTransformer("fine_tuned_model")  # Replace with your fine-tuned model path

# Command dictionary
COMMANDS = {
    "forward": ["move forward", "go forward", "advance", "fly ahead", "proceed"],
    "backward": ["move back", "go back", "reverse", "come back", "retreat"],
    "turn_left": ["turn left", "rotate left", "pivot left", "bank left"],
    "turn_right": ["turn right", "rotate right", "pivot right", "bank right"],
    "ascend": ["go up", "ascend", "rise", "climb", "elevate"],
    "descend": ["go down", "descend", "drop", "lower altitude"],
    "stop": ["stop", "hover", "pause", "halt"]
}

DISTANCE_UNITS = ["meter", "meters", "m"]
ANGLE_UNITS = ["degree", "degrees", "°"]

# Pre-encode all command phrases
all_phrases = []
command_labels = []

for label, phrases in COMMANDS.items():
    for phrase in phrases:
        all_phrases.append(phrase)
        command_labels.append(label)

command_embeddings = model.encode(all_phrases, convert_to_tensor=True)

# === Helper Functions ===

def split_input(text):
    return re.split(r'\b(?:and then|then|and|after that|,)\b', text, flags=re.IGNORECASE)

def extract_value(text, unit_keywords):
    pattern = r'(\d+\.?\d*)\s*(%s)' % "|".join(unit_keywords)
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def classify_phrase(text, threshold=0.4):
    phrase_embedding = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(phrase_embedding, command_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    if best_score >= threshold:
        label = command_labels[best_idx]
        value = None

        # Extract value based on command type
        if label in ["forward", "backward", "ascend", "descend"]:
            value = extract_value(text, DISTANCE_UNITS)
        elif label in ["turn_left", "turn_right"]:
            value = extract_value(text, ANGLE_UNITS)

        return {"command": label, "value": value, "confidence": best_score}
    else:
        return None

def parse_input(text):
    phrases = split_input(text)
    results = []
    for phrase in phrases:
        phrase = phrase.strip()
        if phrase:
            result = classify_phrase(phrase)
            if result:
                results.append(result)
    return results

# === Main Program ===

if __name__ == "__main__":
    print("Type natural-language commands or 'quit':\n")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        actions = parse_input(user_input)
        if not actions:
            print("→ No valid commands found.\n")
        else:
            for action in actions:
                print(f"→ Command: {action['command']}, Value: {action['value']}, Confidence: {action['confidence']:.2f}")
        print()
