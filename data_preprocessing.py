import pandas as pd
from datasets import load_dataset

def load_data():
    try:
        print("Loading RecipeNLG dataset from local path...")
        dataset = load_dataset('recipe_nlg', data_dir='./data/dataset', split='train[:1000]', trust_remote_code=True)
        df = pd.DataFrame(dataset)
        print(f"Dataset loaded successfully. Number of recipes: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data():
    df = load_data()
    if df is None or df.empty:
        print("No data to process. Exiting.")
        return None
    print("Preprocessing data...")
    # Format: "Ingredients: ingredient1, ingredient2 | Recipe: instructions"
    df['formatted'] = df.apply(lambda x: f"Ingredients: {', '.join(x['ingredients'])} | Recipe: {x['directions']}", axis=1)
    print("Saving to processed_recipes.csv...")
    df[['formatted']].to_csv('processed_recipes.csv', index=False)
    print("Data saved successfully.")
    return df

if __name__ == "__main__":
    preprocess_data()