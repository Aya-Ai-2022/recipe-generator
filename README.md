
# 🍳 AI-Powered Recipe Generator

Welcome to the **AI-Powered Recipe Generator**, a delightful web app that transforms your ingredients into delicious recipes using the power of GPT-2! Built with Streamlit and deployed on Streamlit Community Cloud, this project showcases my expertise in natural language processing, machine learning, and web development.

🌐 **Live Demo**: Try it now!
![https://recipe-generator-gpt2.streamlit.app/]

## 🎯 Project Overview

Ever wonder what to cook with the ingredients in your pantry? This app takes your input (e.g., "chicken, rice, broccoli") and generates a custom recipe with clear, step-by-step instructions. Powered by a fine-tuned GPT-2 model, it delivers creative and practical recipes in a sleek, user-friendly interface.

### Features

- **AI-Driven Recipes**: Generates recipes using a fine-tuned GPT-2 model.
- **Modern UI**: Clean, responsive design with custom CSS for a delightful user experience.
- **Robust Deployment**: Hosted on Streamlit Community Cloud for seamless access.
- **Error Handling**: Fallback steps ensure complete recipes even if the model output is limited.

## 🚀 Getting Started

### Prerequisites

#### 🐍 Python Version

This project uses **Python 3.9** (via `python:3.9-slim` in Docker).

> ⚠️ If you are using a local environment, please ensure your Python version is **>=3.9** for compatibility.

- Dependencies listed in `requirements.txt`
- Access to the GPT-2 model files (hosted on Google Drive)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aya-Ai-2022/recipe-generator.git
   cd recipe-generator
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac or use windows but use git bash terminal
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the app locally:

   ```bash
   streamlit run app.py
   ```

The app will download the GPT-2 model (\~498 MB) from Google Drive on first run and launch at `http://localhost:8501`.

## 🛠️ How It Works

1. **Input**: Enter ingredients (e.g., "chicken, rice, broccoli") in the text box.
2. **Model**: A fine-tuned GPT-2 model generates a recipe based on the input.
3. **Output**: The app formats the recipe with ingredients, numbered steps, and serving suggestions, styled with a modern UI.
4. **Deployment**: Hosted on Streamlit Community Cloud, with model files fetched from Google Drive to optimize performance.

## 📸 Demo

![https://recipe-generator-gpt2.streamlit.app/](demo.gif)

Try it out at https://recipe-generator-gpt2.streamlit.app/!

## 🧠 Technical Details

- **Model**: GPT-2, fine-tuned for recipe generation (model files hosted on Google Drive).
- **Framework**: Streamlit for the web interface.
- **Libraries**: `transformers`, `torch`, `gdown`, `streamlit`.
- **Deployment**: Streamlit Community Cloud.
- **Challenges Overcome**:
  - Fixed meta tensor errors in model loading.
  - Enhanced output with detailed steps and robust formatting.
  - Optimized model storage using Google Drive.

## 🌟 Why This Project Shines

This project demonstrates my ability to:

- Build and deploy AI-powered web applications.
- Fine-tune and integrate large language models like GPT-2.
- Create user-friendly interfaces with custom styling.
- Solve complex deployment issues (e.g., meta tensor errors, model loading).

## 📬 Contact

- **GitHub**: Aya-Ai-2022

- **Portfolio**: \[aya2020.xyz\]

Feel free to star ⭐ this repository or reach out with feedback!

## 🔮 Future Enhancements

- Add support for dietary preferences (e.g., vegan, gluten-free).
- Integrate a recipe rating system.
- Fine-tune the model further for more creative outputs.

Bon appétit! 🍽️