
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
# Custom CSS for modern design


import gdown
import tarfile
import os

if not os.path.exists("recipe_model"):
    os.makedirs("recipe_model")
    url = "https://drive.google.com/uc?id=1PsxH3PXeqiJsVl0vWS9tXS-dGDG1ZLDL"
    output = "recipe_model.tar.gz"
    gdown.download(url, output, quiet=False)

    with tarfile.open(output, "r:gz") as tar:
        tar.extractall()

    os.remove("recipe_model.tar.gz")


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
          color:  #abffab;
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

tokenizer = GPT2Tokenizer.from_pretrained("recipe_model")
model = GPT2LMHeadModel.from_pretrained("recipe_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_recipe(ingredients):
    input_text = f"Ingredients: {ingredients}\nRecipe:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=50,
        return_attention_mask=True
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# def generate_recipe(ingredients):
#     tokenizer = GPT2Tokenizer.from_pretrained('recipe_model')
#     model = GPT2LMHeadModel.from_pretrained('recipe_model')
#     tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token
#     tokenizer.pad_token_id = tokenizer.eos_token_id  # 50256 for GPT-2
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     prompt = f"Ingredients: {ingredients} | Recipe:"
#     inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=64,return_attention_mask=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     outputs = model.generate(
        
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=256,
#         num_return_sequences=1,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7
#     )
#     recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return recipe
def format_recipe(recipe_text, ingredients):
    import re

    # Split ingredients and recipe
    parts = recipe_text.split(" | Recipe:")
    if len(parts) < 2:
        return f"Error: Invalid recipe format. Generated text: {recipe_text}"

    # Clean ingredient list
    ingredient_list = [ing.strip() for ing in ingredients.split(',') if ing.strip()]

    # Step 1: Extract raw steps string
    raw_steps = parts[1].strip()

    # Step 2: Clean it as a pseudo-list
    # Remove brackets if present
    raw_steps = raw_steps.strip("[]")
    
    # Split on comma if it looks like a list
    if "'," in raw_steps or '",' in raw_steps:
        steps = re.split(r"'\s*,\s*'|\"\s*,\s*\"", raw_steps)
    else:
        # Fallback: split by period
        steps = raw_steps.split(".")

    # Step 3: Clean individual steps
    cleaned_steps = []
    seen = set()
    for step in steps:
        clean = step.strip(" '\"\n").strip()
        if clean and clean not in seen:
            cleaned_steps.append(clean)
            seen.add(clean)

    # Format output
    output = "🍽️ Simple Recipe Guide\n\n"
    output += "🧂 **Ingredients**\n"
    for ing in ingredient_list:
        output += f"- {ing}\n"

    output += "\n📖 **Recipe Steps**\n"
    for i, step in enumerate(cleaned_steps, 1):
        output += f"{i}. {step}\n"

    output += "\n✅ **Ready to Serve!**\nEnjoy a warm, cozy dish filled with goodness!"
    
    return output




def main():
    st.title("🍽️ AI-Powered Recipe Generator")
    st.write("Enter ingredients, and let AI craft a delicious recipe for you!")

    # Input form
    ingredients = st.text_input("Enter ingredients (comma-separated, e.g., chicken, rice, broccoli):", 
                              placeholder="Type ingredients here...")
    
    if st.button("Generate Recipe"):
        if ingredients:
            with st.spinner("Cooking up a recipe..."):
                recipe_text = generate_recipe(ingredients)
                formatted_recipe = format_recipe(recipe_text, ingredients)
                st.markdown(f"<div class='recipe-output recipe-section'>{formatted_recipe}</div>", 
                           unsafe_allow_html=True)
        else:
            st.warning("Please enter some ingredients!")

if __name__ == "__main__":
    main()