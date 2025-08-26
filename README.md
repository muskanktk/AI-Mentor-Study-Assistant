# AI Mentor Study Assistant

Turn any lecture PDF into **3 learning modes**:

1. Curated **YouTube** videos for key topics,
2. An **OpenAI-powered summary** in simple language (with code snippets and analogies), and
3. **Hands-on practice** with simulations and quizzes.
   Optionally, generate a **voiceover MP3** of the summary in a voice you choose.

**Live demo:** 


**Example PDF:** [cs262\_week12\_slides (Tagged).pdf](https://github.com/user-attachments/files/21974421/cs262_week12_slides.15.1.1.-.Tagged.pdf)


## 🚀 Features

*  **YouTube Topic Guide**: Finds videos aligned to the main concepts in your PDF.
*  **Plain-English Summary**: OpenAI condenses content, adds examples, code, and analogies.
*  **MP3 Voiceover**: Generate an audio narration of the summary with multiple voice options.
*  **Practice Mode**: Simulations and configurable quizzes (choose number of questions).
*  **One-click Export**: Save everything as a `.md` study file for safekeeping.


## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks/Libraries:** `streamlit`, `pdfplumber`, `httpx`, `gTTS`, `python-dotenv`, plus standard libs (`os`, `io`, `json`, `datetime`, `urllib.parse`)
* **Tools:** Git, VS Code / Visual Studio Code


## 📦 Installation

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/AI-Mentor-Study-Assistant.git
cd AI-Mentor-Study-Assistant

# 2) Create & activate a virtual environment
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```
streamlit
pdfplumber
gTTS
httpx
python-dotenv
certifi
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
# Optional if you use other services:
# YOUTUBE_API_KEY=...
```

---

## ▶Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## 💡 How to Use

1. **Upload a PDF** (lecture slides, notes, article).
2. Pick your **learning modes**:

   * *Videos*: fetch topic-aligned YouTube links.
   * *Summary*: generate a plain-English explanation with examples, code, and analogies.
   * *Voiceover*: produce an MP3 narration of the summary and play it in the app.
   * *Practice*: create simulations/quizzes; choose the number of questions.
3. **Export**: click **Export** to save a `.md` file with the summary, links, and notes.

---

## 📂 Project Structure

```
AI-Mentor-Study-Assistant/
├─ app.py               # Streamlit entry point
├─ requirements.txt     # Python dependencies
├─ README.md            # This file
├─ .env                 # Your API keys (not committed)
├─ data/                # Uploaded PDFs / sample assets
└─ src/
   ├─ summarizer.py     # OpenAI summary helpers
   ├─ audio.py          # TTS (gTTS) utilities
   ├─ quiz.py           # Quiz generation / logic
   ├─ videos.py         # YouTube topic fetchers
   └─ utils.py          # shared helpers (parsing, I/O, etc.)
```

> Your exact files may differ—use this as a template.

---

## 🌐 Hosted App

* **Browser link:** [https://ai-mentor-study-assistant-tvv5bhksyvyderelpq8pmh.streamlit.app/](https://ai-mentor-study-assistant-tvv5bhksyvyderelpq8pmh.streamlit.app/)
* Try it with the **example PDF** above or your own notes.

## 🛣️ Roadmap

* [ ] Additional voice models / SSML support
* [ ] Richer simulations and spaced-repetition scheduling
* [ ] Per-topic flashcards and cloze deletions
* [ ] Multi-PDF study packs & cross-topic linking

## FAQ

**Q: Do I need an OpenAI key?**
A: Yes, for summaries and any OpenAI-powered features. Put it in `.env`.

**Q: Where do exports go?**
A: The app saves `.md` outputs to a project subfolder (e.g., `data/exports/`) and may let you download directly.

**Q: Does it work with scanned PDFs?**
A: If OCR is enabled in your build, yes. Otherwise, use a tagged/text PDF.

## Troubleshooting

* **`ModuleNotFoundError`**: Re-run `pip install -r requirements.txt` in the active venv.
* **Streamlit won’t start**: Ensure the venv is active; try `python --version` and `which streamlit`.
* **API errors**: Check `.env` values and your usage limits.
* **gTTS network issues**: Ensure your network allows outbound requests; retry or switch networks.

## Contributing

Please open an issue to discuss substantial changes first.

## License
MIT License – see [LICENSE](LICENSE) for details.


## Acknowledgments

* OpenAI for summarization APIs
* Streamlit for rapid UI
* pdfplumber for PDF text extraction
* gTTS for lightweight text-to-speech

---

## Final Product

- 🎧 Voiceover: [voiceover.mp3](https://github.com/user-attachments/files/21974774/voiceover.3.mp3)  
- 📘 Study Pack: [study_pack.pdf](https://github.com/user-attachments/files/21974831/study_pack.pdf)  
- 🃏 Flashcards: [flashcards.pdf](https://github.com/user-attachments/files/21974817/flashcards.pdf)  

