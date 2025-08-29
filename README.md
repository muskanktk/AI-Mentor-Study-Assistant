# AI Mentor Study Assistant

Turn any lecture PDF into **3 learning modes**:

1.  Curated **YouTube** videos for key topics.
2.  An **OpenAI-powered summary** in simple language.
3.  **Hands-on practice** with simulations and quizzes.

You can also generate a **voiceover MP3** of the summary.

> **Live Demo:**
![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/673254b8-1627-4297-811c-8b21b442bd24)


 
> **Browser Link:** [https://ai-mentor-study-assistant-tvv5bhksyvyderelpq8pmh.streamlit.app/](https://ai-mentor-study-assistant-tvv5bhksyvyderelpq8pmh.streamlit.app/)

-----

## 🚀 Key Features

  * **YouTube Topic Guide**: Finds videos aligned with the main concepts in your PDF.
  * **Plain-English Summary**: Uses OpenAI to condense content, adding examples, code, and analogies.
  * **MP3 Voiceover**: Generates an audio narration of the summary with multiple voice options.
  * **Practice Mode**: Creates customizable simulations and quizzes.
  * **One-Click Export**: Saves everything as a Markdown (`.md`) study file.

-----

## 🛠️ Tech Stack

  * **Language:** Python
  * **Frameworks:** `streamlit`
  * **Libraries:** `pdfplumber`, `httpx`, `gTTS`, `python-dotenv`
  * **Tools:** Git, VS Code

-----

## 📦 Installation

To get started, follow these steps:

1.  **Clone the repository and navigate into the directory:**
    ```bash
    git clone https://github.com/<your-username>/AI-Mentor-Study-Assistant.git
    cd AI-Mentor-Study-Assistant
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Mac/Linux
    source venv/bin/activate
    # Windows (PowerShell)
    venv\Scripts\Activate.ps1
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Sample `requirements.txt`

```
streamlit
pdfplumber
gTTS
httpx
python-dotenv
```

-----

## ▶ Run Locally

1.  **Set up your API key:**
    Create a `.env` file in the project root and add your OpenAI key:
    ```
    OPENAI_API_KEY=sk-...
    ```
2.  **Start the application:**
    ```bash
    streamlit run app.py
    ```
    This will open the app in your browser.

-----

## 💡 How to Use

1.  **Upload a PDF** (lecture slides, notes, or an article).
2.  **Select your learning modes**: Choose from videos, a summary, a voiceover, or practice questions.
3.  **Export your study materials**: Click **Export** to save a `.md` file with the summary, links, and notes.

-----

## 📂 Project Structure

```
AI-Mentor-Study-Assistant/
├─ app.py                # Main Streamlit application
├─ requirements.txt      # Python dependencies
├─ README.md             # README
├─ .env                  # API keys (not shown)
├─ src/
│  ├─ audio.py           # Text-to-speech helpers
│  ├─ quiz.py            # Quiz generation logic
│  ├─ summarizer.py      # OpenAI summary helpers
│  ├─ utils.py           # Shared helper functions
│  └─ videos.py          # YouTube video fetchers
└─ data/
   ├─ uploads/           # Uploaded PDFs
   └─ exports/           # Exported study materials
```

-----

## 🛣️ Roadmap

  * Additional voice models.
  * Richer simulations and spaced-repetition scheduling.
  * Per-topic flashcards.

-----

## 🙋 FAQ

**Q: Do I need an OpenAI key?**
A: Yes, it's required for the summary and other AI-powered features.

**Q: Does it work with scanned PDFs?**
A: It works best with text-based PDFs. Scanned PDFs may require additional OCR processing.

-----

## 🤝 Contributing

Please open an issue to discuss any significant changes before submitting a pull request.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

## 🙏 Acknowledgments

  * **OpenAI** for the summarization APIs.
  * **Streamlit** for the rapid UI development.
  * **pdfplumber** for PDF text extraction.
  * **gTTS** for text-to-speech functionality.
