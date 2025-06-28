import os
import gdown
import tarfile
import shutil
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Custom CSS for modern design
st.markdown("""
<style>
body {
    background-color: #F1FAEE;
    color: #2D3436;
    font-family: 'Arial', sans-serif;
}
.stApp {
    background-color: #F1FAEE;
}
h1 {
    color: #4ECDC4;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 0.5em;
}
.stTextInput > div > div > input {
    border: 2px solid #4ECDC4;
    border-radius: 10px;
    padding: 10px;
    background-color: #F7FFF7;
}
.stButton > button {
    background-color: navy;
    color: #F7FFF7;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: blue;
    color: #abffab;
}
.recipe-output {
    background-color: #F7FFF7;
    border: 2px solid #4ECDC4;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
    font-size: 1.1em;
    line-height: 1.5;
}
.recipe-section {
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Function to download and extract model files
def ensure_model_files():
    model_dir = "recipe_model"
    required_files = ["vocab.json", "merges.txt", "config.json", "model.safetensors"]
    if not os.path.exists(model_dir) or not all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
        st.info("Downloading model files...")
        os.makedirs(model_dir, exist_ok=True)
        tar_path = "recipe_model.tar.gz"
        # Update with new Google Drive file ID
        gdown.download("https://drive.google.com/uc?id=1PsxH3PXeqiJsVl0vWS9tXS-dGDG1ZLDL", tar_path, quiet=False)
        with tarfile.open(tar_path, "r:gz") as tar:
            temp_dir = "temp_recipe_model"
            os.makedirs(temp_dir, exist_ok=True)
            tar.extractall(path=temp_dir)
            source_dir = os.path.join(temp_dir, "recipe_model") if os.path.exists(os.path.join(temp_dir, "recipe_model")) else temp_dir
            for file in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, file), os.path.join(model_dir, file))
            shutil.rmtree(temp_dir)
        os.remove(tar_path)
        if not all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
            raise FileNotFoundError("Failed to extract model files: " + ", ".join(required_files))

# Load model and tokenizer
ensure_model_files()
try:
    tokenizer = GPT2Tokenizer.from_pretrained("recipe_model", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("recipe_model", local_files_only=True, torch_dtype=torch.float32)
    # Move model to CPU, handling meta tensors
    if any(param.is_meta for param in model.parameters()):
        st.warning("Model loaded in meta state; attempting to move to CPU.")
        model.to_empty(device="cpu")
    else:
        model.to("cpu")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    raise

def format_recipe_output(raw_text, ingredients):
    if "Recipe:" not in raw_text:
        return f"❌ Invalid recipe format.\n\nRaw Output:\n\n{raw_text}"

    try:
        steps_text = raw_text.split("Recipe:")[1].strip()
        steps_text = steps_text.replace("[", "").replace("]", "").replace("'", "").replace('"', '')
        raw_steps = [s.strip() for s in steps_text.split(",") if s.strip()]
        steps = []

        # Handle sentences with punctuation or no punctuation
        for s in raw_steps:
            if "." in s:
                steps.extend([sub.strip() for sub in s.split(".") if sub.strip()])
            else:
                steps.append(s)

        # Final cleanup
        steps = [step for step in steps if len(step.split()) > 2]  # Filter out junk phrases
    except Exception as e:
        return f"❌ Failed to parse recipe steps. Error: {str(e)}\n\nRaw Output:\n\n{raw_text}"

    # Format ingredients
    ing_list = [ing.strip() for ing in ingredients.split(",") if ing.strip()]
    output = "<h4>🧂 <strong>Ingredients</strong></h4>\n<ul>"
    for ing in ing_list:
        output += f"<li>{ing.capitalize()}</li>"
    output += "</ul>"

    # Format steps
    output += "<h4>📖 <strong>Steps</strong></h4>\n<ol>"
    for step in steps:
        output += f"<li>{step}</li>"
    output += "</ol>"

    output += "<p>✅ <strong>Done!</strong><br>Enjoy your tasty creation 😋</p>"
    return output


def generate_recipe(ingredients):
    input_text = f"Ingredients: {ingredients}\nRecipe:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=50,
        return_attention_mask=True
    ).to("cpu")
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Prevent repetitive phrases
            temperature=0.7  # Control randomness
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating recipe: {str(e)}"

def main():
    st.title("🍳 AI-Powered Recipe Generator")
    st.markdown("Enter ingredients to get a custom recipe!")
    ingredients_input = st.text_input("🧂 Enter ingredients (e.g., chicken, rice, broccoli):")
    if st.button("🎉 Generate Recipe"):
        if ingredients_input:
            with st.spinner("Cooking up a recipe..."):
                try:
                    recipe_text = generate_recipe(ingredients_input)
                    formatted = format_recipe_output(recipe_text, ingredients_input)
                    st.markdown(f"<div class='recipe-output recipe-section'>{formatted}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating recipe: {str(e)}")
        else:
            st.error("Please enter ingredients.")

if __name__ == "__main__":
    main()