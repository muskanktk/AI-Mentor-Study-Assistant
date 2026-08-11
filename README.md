# AI Mentor Study Assistant

AI Mentor Study Assistant is an AI-powered learning platform designed to help students turn lecture materials into interactive study resources.

The application uses **OpenAI** to analyze uploaded course materials, extract important topics, generate summaries, and answer questions. It also integrates the **YouTube Data API** to find educational videos related to extracted topics and **MongoDB Atlas** to store uploaded file information and study resources.

> **Live Demo:**
> [AI Mentor Study Assistant](file:///Users/muska/Downloads/streamlit-app-2026-08-11-16-56-00.webm/)

---

## 🚀 Key Features

### 📄 Lecture Material Upload

* Upload course materials directly through the Streamlit interface.
* Supports:

  * PDF
  * DOCX
* Extracts text from uploaded documents for AI processing.
* Stores uploaded file information in MongoDB Atlas.
* Organizes uploaded materials into **Current** and **Archived** sections.

### 🤖 AI-Powered Summaries

* Uses the **OpenAI API** to analyze uploaded lecture material.
* Generates summaries to make complex lecture content easier to understand.
* Provides explanations based on uploaded study materials.
* Allows students to interact with their lecture content through an AI-powered study assistant.

### 🔎 Automatic Topic Extraction

* Uses OpenAI to identify the main topics from uploaded lecture documents.
* Stores extracted topics in MongoDB alongside the uploaded file information.
* Extracted topics can be used to find additional learning resources.

### ▶️ YouTube Learning Resources

* Uses extracted lecture topics to search for relevant educational videos.
* Integrates the **YouTube Data API** to retrieve real YouTube video results.
* Provides students with additional learning resources related to their lecture material.

### 🧠 Flashcards

* Includes a dedicated flashcards page for reviewing course concepts.
* Uses question-and-answer style cards to support active recall.
* Designed to make studying more interactive.

### 💬 AI Study Assistant

* Students can ask questions about their uploaded lecture materials.
* Uses information from the uploaded content to provide context-aware responses.
* Helps explain difficult concepts in a more understandable way.

### 🗄️ MongoDB Storage

* Uses **MongoDB Atlas** as the application's database.
* Stores information about uploaded files and extracted topics.
* Stores file status and related learning-resource information.
* Allows study materials to persist between application sessions.

---

## 🛠️ Tech Stack

### Language

* **Python**

### Framework

* **Streamlit**

### AI & APIs

* **OpenAI API**
* **YouTube Data API**

### Database

* **MongoDB Atlas**
* **PyMongo**

### Document Processing

* **PyPDF2 / pypdf**
* **python-docx**

### Development Tools

* **Git**
* **GitHub**
* **VS Code**
* **GitHub Codespaces**

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/muskanktk/AI-Mentor-Study-Assistant.git
cd AI-Mentor-Study-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Mac/Linux:**

```bash
source venv/bin/activate
```

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 API & Database Configuration

The application requires credentials for OpenAI, YouTube, and MongoDB Atlas.

For local development, configure your Streamlit secrets.

Create the following file:

```text
.streamlit/secrets.toml
```

Add your credentials:

```toml
OPENAI_API_KEY = "your-openai-api-key"
YOUTUBE_API_KEY = "your-youtube-api-key"
MONGODB_URI = "your-mongodb-connection-string"
```

**Never commit API keys, passwords, or database connection strings to GitHub.**

---

## ▶️ Run Locally

After installing the dependencies and configuring your secrets, run:

```bash
streamlit run app.py
```

The application will start locally and Streamlit will provide a URL to open it in your browser.

---

## 💡 How to Use

1. Open the AI Mentor Study Assistant.
2. Upload a lecture PDF, DOCX, or CSV file.
3. The application extracts the document content.
4. OpenAI analyzes the content and identifies the main topics.
5. File information and extracted topics are stored in MongoDB Atlas.
6. Generate an AI-powered summary of the lecture material.
7. Ask questions about the uploaded content using the AI Study Assistant.
8. Use extracted topics to find relevant educational YouTube videos.
9. Review important concepts using the flashcards feature.

---

## 📂 Project Structure

```text
AI-Mentor-Study-Assistant/
│
├── app.py                  # Main Streamlit application
├── database.py             # MongoDB connection and database operations
├── extractTopics.py        # Extracts topics from lecture content using OpenAI
├── store.py                # Handles storing uploaded file information
├── test_mongo.py           # MongoDB connection/testing script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── pages/
│   └── ...                 # Additional Streamlit application pages
│
├── env/                    # Environment-related files
├── venv/                   # Python virtual environment
└── __pycache__/            # Python-generated cache files
```

> **Note:** `venv/`, `env/`, and `__pycache__/` should generally not be committed to GitHub. Add them to `.gitignore` for a cleaner repository.

---

## 🗃️ Database

The application uses **MongoDB Atlas** to persist study-material information.

The database stores information such as:

* Uploaded file names
* Extracted topics
* File status
* Related learning resources
* YouTube links

A typical uploaded file record contains information similar to:

```text
Content: 4-Classification-1.pdf
Topics: Classification, supervised learning, decision trees, ...
Status: Current
Links: YouTube learning resources
```

MongoDB allows the application to retrieve previously uploaded materials without requiring the user to reprocess the files every time.

---

## 🤖 AI Processing

OpenAI is used throughout the application to support learning features.

The application can use OpenAI to:

* Extract important topics from lecture materials.
* Generate summaries.
* Explain concepts.
* Answer questions about uploaded content.
* Help transform lecture material into study resources.

The goal is to make lecture material easier to understand while keeping the generated responses relevant to the student's uploaded content.

---

## ▶️ YouTube Integration

The application integrates with the **YouTube Data API** to provide additional learning resources.

The workflow is:

```text
Lecture PDF
     │
     ▼
Extract Topics
     │
     ▼
OpenAI Topic Processing
     │
     ▼
YouTube Search
     │
     ▼
Relevant Educational Videos
```

The application uses the YouTube API to search for actual videos rather than generating or guessing YouTube URLs.

---

## 🧠 Flashcards

The application includes a dedicated flashcards section designed to help students review concepts from their study material.

Current and planned flashcard functionality includes:

* Question-and-answer flashcards
* Interactive review
* Timed flashcard activities
* More game-like study experiences

---

## 🛣️ Roadmap

Planned improvements include:

* More interactive flashcard games.
* Timed flashcard challenges.
* Expanded quiz functionality.
* Improved document processing.
* Better support for scanned PDFs and OCR.
* More advanced AI-powered question answering.
* Improved YouTube recommendations.
* Additional personalized learning features.
* More interactive study activities.

---

## 🙋 FAQ

### Do I need an OpenAI API key?

Yes. OpenAI is used for topic extraction, summaries, explanations, and AI-powered question answering.

### Do I need a YouTube API key?

Yes, if you want to use the YouTube learning-resource functionality. The YouTube Data API is used to search for educational videos based on extracted lecture topics.

### Do I need MongoDB?

Yes. The current application uses MongoDB Atlas to store uploaded file information, extracted topics, statuses, and related learning resources.

### Does the application support scanned PDFs?

The application works best with documents that contain extractable text. Scanned PDFs may require OCR before their content can be processed effectively.

### Where should I store my API keys?

For local development, API keys and database credentials should be stored using Streamlit secrets.

Do **not** place API keys directly in your Python source code or commit them to GitHub.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

For significant changes, please open an issue first to discuss the proposed feature or modification.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* **OpenAI** — AI-powered topic extraction, summaries, explanations, and question answering.
* **Streamlit** — Application framework and interactive user interface.
* **MongoDB Atlas** — Database storage.
* **YouTube Data API** — Educational video search and recommendations.
* **PyPDF2 / pypdf** — PDF document processing.
* **python-docx** — DOCX document processing.
* **GitHub** — Source control and project hosting.
