# MicroCombust-AI Dashboard

The AETHON Next.js dashboard provides interactive telemetry controls, material
risk estimates, a NASA-paper RAG chat, and an exportable executive briefing.

## Run locally

Start the FastAPI service from the repository root:

```bash
python main.py
```

The API selects a free local port, preferring `8000`. Check its status at
`http://127.0.0.1:8000/health` (replace the port if the startup message reports
a fallback port).

In another terminal:

```bash
cd microcombust-ui
npm install
npm run dev
```

Open `http://localhost:3000`. To override API discovery, set
`NEXT_PUBLIC_API_URL` to one or more comma-separated API base URLs before
starting Next.js.

## Dashboard behavior

- Oxygen, pressure, airflow, fuel, and gravity controls update the dashboard
  alerts and local flammability estimates.
- **Auto Simulation (Demo Mode)** cycles through preset telemetry scenarios
  every four seconds. It updates the controls and local estimate; it is not a
  physics solver and does not call FastAPI.
- **RUN SIMULATION & RANK RISK** requests a ranking from `POST /rank-materials`.
  If the API is unavailable, the dashboard shows the local estimate and an
  error message.
- The AI console sends questions and recent conversation turns to
  `POST /query-agent`. Short queries and experiment acronyms are expanded for
  retrieval; answers render Markdown headings, bullets, and clickable NASA
  source links under the references heading.
- **Export Executive Briefing** downloads an HTML report that can be printed
  or saved as PDF.

The flame behavior index currently contains illustrative demo seed data, not
validated NASA measurements. The RAG service reports readiness at `/health`;
provider keys and optional model overrides (`GROQ_MODEL`, `OPENROUTER_MODEL`)
belong in the root `.env` file.
