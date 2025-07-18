from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# === Load model ===
model_path = "gpt2_v2"  # change if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# === Prompt ===
system_prompt = """
You are an AI assistant controlling a Holy Stone HS110G drone. Your goal is to translate user commands into a precise flight plan.

Available Commands:
- FORWARD, [distance_in_meters]
- BACKWARD, [distance_in_meters]
- LEFT, [distance_in_meters] (strafe left)
- RIGHT, [distance_in_meters] (strafe right)
- PAN_LEFT, [angle_in_degrees] (rotate left)
- PAN_RIGHT, [angle_in_degrees] (rotate right)
- UP, [distance_in_meters]
- DOWN, [distance_in_meters]

Your Task:
1. Analyze the user's request.
2. Generate a flight plan as a list of dictionaries, with key-value pairs of command (select one of the commands listed above) and distance (in meters) or angle (in degrees) in a section titled [OUTPUT].
3. Be concise. Do not add conversational text outside of the required sections.
"""

# === Preprocess instruction ===
def split_instruction(text):
    # Lowercase and split based on connectors
    parts = re.split(r'\s*(?:and|then|,|;|\.)\s*', text, flags=re.IGNORECASE)
    # Remove empty strings and strip whitespace
    return [p.strip().capitalize() for p in parts if p.strip()]

# === Inference function ===
def generate_flight_plan(user_instruction, max_new_tokens=350):
    split_instructions = split_instruction(user_instruction)
    all_commands = []

    for inst in split_instructions:
        prompt = f"{system_prompt}\n[USER INPUT] {inst}\n[OUTPUT]\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        command_part = result[len(prompt):].strip()
        all_commands.append(command_part)

    return "\n".join(all_commands)

# === Interactive Loop ===
if __name__ == "__main__":
    while True:
        instruction = input("\nType a drone instruction (or type 'quit' to exit): ").strip()
        if instruction.lower() == 'quit':
            print("Exiting.")
            break
        flight_plan = generate_flight_plan(instruction)
        print(f"\n[USER INPUT] {instruction}")
        print(f"\n[OUTPUT]\n{flight_plan}")