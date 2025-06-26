import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

def generate_recipe(ingredients):
    tokenizer = GPT2Tokenizer.from_pretrained('recipe_model')
    model = GPT2LMHeadModel.from_pretrained('recipe_model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prompt = f"Ingredients: {ingredients} | Recipe:"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        inputs['input_ids'],
        max_length=300,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe

def format_recipe(recipe_text, ingredients):
    # Split ingredients and recipe
    parts = recipe_text.split(" | Recipe:")
    if len(parts) < 2:
        return f"Error: Invalid recipe format. Generated text: {recipe_text}"

    # Clean ingredient list
    ingredient_list = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
    
    # Extract and clean recipe steps (handle list-like strings)
    raw_steps = parts[1].strip()

    # Try to safely evaluate stringified list (if it exists)
    try:
        import ast
        parsed_steps = ast.literal_eval(raw_steps)
        if isinstance(parsed_steps, list):
            steps = [s.strip() for s in parsed_steps if s.strip()]
        else:
            steps = [s.strip() for s in raw_steps.split('.') if s.strip()]
    except:
        # Fall back to basic splitting if not a valid list
        steps = [s.strip() for s in raw_steps.split('.') if s.strip()]
    
    # Remove duplicates
    seen_steps = set()
    cleaned_steps = []
    for step in steps:
        if step not in seen_steps and "Remove from heat and drain" not in step:
            cleaned_steps.append(step)
            seen_steps.add(step)

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