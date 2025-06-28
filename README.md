
# 🍳 AI-Powered Recipe Generator

  Welcome to the **AI-Powered Recipe Generator**, a delightful web app that transforms your ingredients into delicious recipes using the power of GPT-2! Built with Streamlit, containerized with Docker, and deployed on Streamlit Community Cloud, this project showcases my expertise in natural language processing, machine learning, and modern deployment practices.

  🌐 **Live Demo** : [Try it now!](https://recipe-generator-gpt2.streamlit.app/)

  ## 🎯 Project Overview

  Ever wonder what to cook with the ingredients in your pantry? This app takes your input (e.g., "chicken, rice, broccoli") and generates a custom recipe with clear, step-by-step instructions. Powered by a fine-tuned GPT-2 model, it delivers creative and practical recipes in a sleek, user-friendly interface.

  ### Features

  - **AI-Driven Recipes**: Generates recipes using a fine-tuned GPT-2 model.
  - **Modern UI**: Clean, responsive design with custom CSS.
  - **Robust Deployment**: Hosted on Streamlit Community Cloud with Docker support for local runs.
  - **Smart Output**: Filters incoherent steps and provides detailed defaults for consistent recipes.

  ## 🚀 Getting Started

  ### Prerequisites

  - Python 3.13 (for local runs without Docker)
  - [Docker](https://www.docker.com/get-started) (optional, for containerized setup)
  - Dependencies listed in `requirements.txt`
  - Access to GPT-2 model files (hosted on Google Drive)


  ### Option 1: Run with Docker (Recommended)

  1. Pull the Docker image:
     ```bash
     docker pull caffein/recipe-generator:latest
     ```
  2. Run the container:
     ```bash
     docker run -p 8501:8501 caffein/recipe-generator:latest
     ```
  3. Access the app at `http://localhost:8501`.

  The app downloads the GPT-2 model (~498 MB) from Google Drive on first run.

### Option 2: Run Locally (Without Docker)

  1. Clone the repository:
     ```bash
     git clone https://github.com/Aya-Ai-2022/recipe-generator.git
     cd recipe-generator
     ```
  2. Create and activate a virtual environment:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate  # Windows
     source .venv/bin/activate  # Linux/Mac
     ```
  3. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  4. Run the app:
     ```bash
     streamlit run app.py
     ```

  ## 🛠️ How It Works

  1. **Input**: Enter ingredients (e.g., "chicken, rice, broccoli") in the text box.
  2. **Model**: A fine-tuned GPT-2 model generates a recipe, with filtering to remove incoherent outputs.
  3. **Output**: Formatted recipe with ingredients, numbered steps, and serving suggestions, styled with a modern UI.
  4. **Deployment**: Hosted on Streamlit Community Cloud, with Docker support for local reproducibility.

  ## 📸 Demo



  Try it out at [recipe-generator-aya.streamlit.app](https://recipe-generator-gpt2.streamlit.app/)!

  ## 🧠 Technical Details

  - **Model**: GPT-2, fine-tuned for recipe generation (model files hosted on Google Drive).
  - **Framework**: Streamlit for the web interface, Docker for containerization.
  - **Libraries**: `transformers`, `torch`, `gdown`, `streamlit`.
  - **Deployment**: Streamlit Community Cloud (Python 3.13) and Docker.
  - **Challenges Overcome**:
    - Fixed meta tensor errors in model loading.
    - Mitigated hallucinations with enhanced generation parameters and output filtering.
    - Optimized model storage using Google Drive.



  ## This project demonstrates:
  - Build and deploy AI-powered web applications.
  - Fine-tune large language models (**LLM**) like GPT-2.
  - Create user-friendly interfaces with custom styling.
  - Use Docker for portable, reproducible environments.

  ## 📬 Contact

  - **GitHub**: [Aya-Ai-2022](https://github.com/Aya-Ai-2022)

  - **Portfolio**: [Aya portfolio](https://aya2020.xyz/)

  Feel free to star ⭐ this repository or reach out with feedback!

  ## 🔮 Future Enhancements

  - Support for dietary preferences (e.g., gluten-free,diabetes).
  - Recipe rating system.
  - Further model fine-tuning for richer outputs.

