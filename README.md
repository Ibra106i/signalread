# SignalRead

SignalRead is a PDF and EPUB reader that extracts only the content that matters. It uses layout classification to identify and skip footnotes, headers, page numbers, and other non-body elements, then passes the clean text through AI for further refinement. The result is high-quality TTS output from a structured, noise-free text stream.

The application is designed as both a web app and an offline Windows desktop app.

**Scope:** SignalRead works with user-owned, DRM-free PDF and EPUB files. DRM-protected files are not supported and will not be supported.

## Setup

> Instructions will be added as each development phase lands.

### Phase 0 — Layout Classification (complete)

Extracts text blocks from PDF and EPUB files and classifies them by type (body text, footnote, header, footer, etc.).

```bash
pip install -r requirements.txt
python engine/phase0_extraction/layout_classifier.py <folder_of_files>
```

### Phase 1 — AI Classification

Uses an LLM (Groq, OpenRouter, or local Ollama) to refine block classification and resolve ambiguous cases.

### Phase 2 — TTS Pipeline

Converts the filtered, body-only text into speech output.

### Phase 3 — API Layer

REST/GraphQL API for programmatic access to the pipeline.

### Frontend

Next.js web interface (planned).

## Project Structure

```
signalread/
├── engine/
│   ├── phase0_extraction/     # PDF/EPUB block extraction
│   ├── phase1_classification/ # AI-assisted classification
│   ├── phase2_tts/            # Text-to-speech output
│   └── phase3_api/            # API layer (future)
├── frontend/                  # Next.js web shell (future)
├── .env.example               # API key placeholders
└── requirements.txt
```

## License

MIT
