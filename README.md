# SAARTHI — The Compassionate AI Vision Companion

### Because independence should never depend on someone else's eyes.

**A little more independence. A little less uncertainty.**

Imagine trying to read a medicine label, locate your keys, understand an unfamiliar room, or complete an everyday task without being able to see the world around you.

For millions of people who are blind or have low vision, these ordinary moments can introduce uncertainty, extra effort, and dependence on others.

**SAARTHI is being built to change that experience.**

SAARTHI is an AI-powered, voice-first vision companion designed to help blind and low-vision people understand visual information, interact with their surroundings, and approach everyday activities with greater confidence.

By combining computer vision, multimodal AI, persistent spatial memory, and spoken interaction, SAARTHI aims to bring several forms of visual assistance together in one intelligent system.

This is more than describing what a camera sees. It is about helping people understand what matters, decide what to do next, and retain control over their everyday experiences.

---

## The problem: A world designed for sight

Much of the world assumes that information is immediately accessible through vision.

A sign tells you where to go. A label tells you what is inside a container. A misplaced object becomes easy to find because you can simply look around. An unfamiliar environment can be understood at a glance.

For someone who cannot rely on sight, these interactions may require additional tools, assistance from another person, or repeated attempts to understand the situation.

Existing accessibility tools can help with individual problems, but switching between separate tools for reading, recognizing objects, understanding surroundings, and following instructions can make everyday interactions more complicated.

**Accessibility should not require people to piece together a collection of disconnected tools just to understand their environment.**

We believe intelligent assistance should be more contextual, more conversational, and designed around the needs of the person using it.

## Our approach: One companion, multiple capabilities

SAARTHI combines six complementary capabilities behind a unified assistant experience.

### 1. Scene understanding

**Understand the surroundings, not just individual objects.**

SAARTHI analyzes camera images and produces natural-language descriptions of visible environments. Users can ask specific questions about objects, furniture, and spatial relationships instead of receiving only a generic description.

The goal is to turn a visual scene into useful, understandable information.

### 2. Text reading

**Make written information more accessible.**

From signs and labels to lecture notes and printed documents, SAARTHI can transcribe visible text through an image-based API.

The text-reading pipeline uses a dedicated model configuration and supports processing modes intended to balance speed and detail.

Handwriting, dense mathematical notation, and unclear images remain challenging and require further accuracy testing.

### 3. Object assistance

**Help people look for the things they need.**

Users can ask about everyday objects such as keys, bottles, bags, or other belongings. SAARTHI analyzes the supplied image and returns a structured result describing whether the requested object appears to be visible and where it appears in relation to other visible features.

When an object cannot be identified confidently, the system should communicate uncertainty rather than inventing its location.

### 4. Step-by-step task guidance

**Turn overwhelming tasks into manageable next steps.**

Some activities are easier when instructions are broken into clear, actionable steps.

SAARTHI's task-guidance capability is designed to generate structured instructions, account for available context, and identify when another image could help verify progress.

The intention is to support users throughout a task rather than provide a long, one-time explanation.

### 5. Spatial memory

**Remember the places and landmarks that matter to the user.**

SAARTHI's spatial-memory service provides persistent storage for user-defined memories, landmarks, descriptions, and textual location references.

Users can create, retrieve, list, update, and delete saved memories.

This provides a foundation for personalized assistance across sessions. It is not automatic indoor positioning, GPS tracking, or a guarantee of a person's current physical location.

### 6. Voice-first assistant

**Interact naturally, without navigating a complicated interface.**

SAARTHI's assistant backend accepts transcribed user requests, identifies the intended capability, and returns concise responses suitable for spoken delivery.

The objective is to let users express what they need naturally, while the assistant determines which functionality is appropriate and requests additional information when required.

Speech recognition and audio playback can be connected through suitable speech services or device capabilities. Optional AI speech synthesis is being integrated and must be verified before it is considered fully operational.

---

## What makes SAARTHI different?

### A connected accessibility workflow

SAARTHI is designed around the relationship between understanding, remembering, and acting.

- **Contextual vision:** describe the scene or answer a question about a supplied image.
- **Specialized transcription:** route text-reading requests to a model configured specifically for that task.
- **Action-oriented assistance:** provide structured steps instead of limiting the interaction to descriptions.
- **Persistent personalization:** retain user-defined landmark information across sessions.
- **Conversational access:** bring capabilities together through one assistant interface.
- **Uncertainty-aware responses:** avoid presenting guesses about object locations or environmental safety as facts.

The central idea is to move from isolated image recognition toward a more useful cycle:

**Understand → Explain → Guide → Remember.**

SAARTHI does not claim to replace established mobility aids, professional support, or human judgment. It aims to make useful information more accessible and everyday interactions less demanding.

## Technology stack

SAARTHI's backend is built with a modular Python architecture.

| Technology | Role |
|---|---|
| Python | Backend implementation |
| FastAPI | REST API and request handling |
| Uvicorn | ASGI server |
| Google Gen AI SDK | Integration with vision and language models |
| Gemma 4 | General scene understanding and object assistance |
| Gemini Flash-Lite | Dedicated text-reading model configuration |
| Pillow | In-memory image preprocessing for text reading |
| Pydantic Settings | Environment-based application configuration |
| SQLite | Persistent spatial-memory storage |
| pytest and HTTPX | Automated testing and API verification |

The architecture separates HTTP endpoints, AI service logic, configuration, and persistent storage to make individual capabilities easier to test and extend.

## Backend architecture

```text
SAARTHI Backend
│
├── Health API
│   └── Backend availability
│
├── Vision APIs
│   ├── Scene understanding
│   └── Text reading
│
├── Object assistance API
│   └── Requested-object analysis
│
├── Task guidance API
│   └── Structured instructions and progress context
│
├── Spatial memory API
│   └── Persistent landmark and place records
│
├── Assistant API
│   └── Intent routing and speech-ready responses
│
├── Services
│   ├── Vision model client
│   └── Image preprocessing
│
└── Configuration and storage
    ├── Environment settings
    └── SQLite database
```

## Getting started

### Prerequisites

- Python 3.12 or a compatible newer version.
- Git.
- A Google Gemini API key for live AI-powered features.
- Internet access for model requests.

### 1. Clone the repository

```bash
git clone https://github.com/vasu-z/Hacktoberfest.git
cd Hacktoberfest
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
python -m pip install -r backend/requirements.txt
```

### 4. Configure environment variables

Create your local environment file from the example:

```powershell
Copy-Item backend/.env.example backend/.env
```

Open `backend/.env` and configure the required credentials and settings:

```dotenv
SAARTHI_SERVICE_NAME=saarthi-backend
SAARTHI_VERSION=0.1.0
SAARTHI_ENV=development

GEMINI_API_KEY=your_actual_api_key

GEMINI_MODEL=gemma-4-26b-a4b-it
GEMINI_OCR_MODEL=gemini-3.5-flash-lite
```

Use the model identifiers supported by your current API account and deployment configuration.

Never commit the actual `.env` file or expose API keys in source code, screenshots, or logs. The example file contains placeholders and is safe to share.

### 5. Start the backend

From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Once the server starts, visit:

- **Interactive API documentation:** http://127.0.0.1:8000/docs
- **Health endpoint:** http://127.0.0.1:8000/api/v1/health

The health endpoint checks backend availability without requiring an AI request.

## API reference

The backend exposes the following capability endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Check backend availability |
| POST | `/api/v1/vision/describe` | Describe an image and answer visual questions |
| POST | `/api/v1/vision/read-text` | Transcribe text visible in an image |
| POST | `/api/v1/objects/assist` | Analyze an image for a requested object |
| POST | `/api/v1/tasks/guide` | Generate structured task guidance |
| POST | `/api/v1/memory` | Save a user-defined memory |
| GET | `/api/v1/memory` | List saved memories for an identifier |
| GET | `/api/v1/memory/{memory_id}` | Retrieve a saved memory |
| PATCH | `/api/v1/memory/{memory_id}` | Update a saved memory |
| DELETE | `/api/v1/memory/{memory_id}` | Delete a saved memory |
| POST | `/api/v1/assistant/ask` | Route a transcribed request to the appropriate assistant capability |

An optional speech-synthesis endpoint may be available once the selected TTS integration has completed implementation and verification. Use the running application's Swagger documentation as the authoritative list of registered routes.

### Example: Scene understanding

Send a multipart request to `POST /api/v1/vision/describe` containing:

- `image`: the image to analyze.
- `question`: an optional question about the image.

Example response:

```json
{
  "description": "A wooden table is positioned in front of a sofa."
}
```

This is an illustrative response format; the actual description depends on the supplied image and model output.

### Example: Text reading

Send a multipart request to `POST /api/v1/vision/read-text` containing:

- `image`: the image to transcribe.
- `mode`: optional processing mode, if enabled in the running version.

Example response:

```json
{
  "text": "EMERGENCY EXIT"
}
```

The output contains the model-generated transcription. Accuracy depends on image quality, handwriting, language, and the complexity of the text.

### Example: Object assistance

Send a multipart request to `POST /api/v1/objects/assist` containing:

- `image`: the image to analyze.
- `target_object`: the requested object.

Example response:

```json
{
  "target_object": "blue mug",
  "found": true,
  "description": "A blue mug is visible near the center of the image.",
  "certainty": "clear"
}
```

The example illustrates the response schema, not a guarantee of detection accuracy.

### Example: Task guidance

Send a request to `POST /api/v1/tasks/guide` using the request format documented in the running Swagger interface. Supply the user's goal and any supported contextual or image information.

The endpoint is designed to return structured guidance, next-step information, and relevant cautions.

### Example: Spatial memory

Use the memory endpoints to save and retrieve user-defined landmarks and place descriptions.

Example creation request:

```json
{
  "user_id": "demo-user",
  "name": "Study desk",
  "description": "A wooden desk used for studying.",
  "location_label": "Beside the bedroom window"
}
```

The `user_id` field is a prototype-level identifier, not an authentication mechanism. A production deployment requires proper authentication and authorization.

## Running the tests

From the repository root, execute:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q
```

The automated test suite covers health checks, configuration, vision-model behavior, image processing, scene understanding, text reading, and object assistance, along with tests for additional capabilities as they are implemented.

Always run the current test suite before submitting changes. Report actual test results rather than assuming every feature is verified because the API starts.

## Responsible design and limitations

SAARTHI is an assistive prototype. Its output can be incomplete, incorrect, or uncertain.

- **Visual understanding is not guaranteed:** models can misidentify objects, positions, and visual details.
- **Handwriting and mathematical notation are challenging:** generated transcriptions can contain errors and should not be treated as authoritative.
- **Object assistance is image-based:** the system does not automatically know an object's physical coordinates or distance.
- **Spatial memory is not live positioning:** saved textual landmarks do not establish a user's current location.
- **Task guidance requires judgment:** image-based assistance cannot guarantee that an environment, action, or path is safe.
- **Prototype memory isolation is limited:** caller-supplied identifiers do not replace secure authentication.
- **Cloud AI has privacy implications:** do not upload sensitive information without understanding the relevant provider's data-handling terms.

SAARTHI should complement established mobility aids and other support, not replace them.

## Our vision

We believe independence grows when information becomes easier to access, choices become easier to understand, and technology adapts to the person instead of forcing the person to adapt to technology.

SAARTHI is an effort to bring those ideas together: to help someone read what was previously inaccessible, understand what surrounds them, remember what matters, and approach everyday tasks with greater confidence.

The measure of this technology is not simply how intelligent a model appears.

It is whether the assistance makes a meaningful difference in someone's life.

**Because the future of accessibility should not merely describe the world. It should help make the world more accessible.**

---

**SAARTHI — A little more independence. A little less uncertainty.**