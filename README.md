
# Neuro-Fit: Hybrid Edge-Cloud Biomechanical AI Trainer

## 🚀 Overview
Neuro-Fit is a real-time, Generative AI-powered fitness application designed to transition fitness tracking from the "Quantified Self" (how much you move) to the "Qualified Self" (how well you move).[1] 

Traditional fitness apps rely on rigid discriminative machine learning to simply count repetitions. Neuro-Fit introduces a hybrid edge-cloud architecture that monitors 3D biomechanics, calculates velocity-based neuromuscular fatigue, and generates context-aware, scientifically-grounded coaching dialogue in real-time.[1] To combat motivation decay, the platform integrates a "FitCoin" gamification economy where users exchange proper form for real-world rewards.[1]

## 🧠 System Architecture
1. **Edge Perception (Fast Loop):** Utilizes MediaPipe to extract 33-point 3D skeletal landmarks in world coordinates at 30fps. A custom Velocity Engine tracks temporal displacement to flag fatigue before form breakdown.[1]
2. **Volumetric Perception (Pro Loop):** Periodically captures physical snapshots and utilizes the SMPL-X parametric model to generate a 3D volumetric mesh (calculating parameters like pose $\theta$ and shape $\beta$) to assess deep spinal/joint alignment.
3. **Cloud Reasoning (Generative Coaching):** Feeds abstracted JSON pose/velocity telemetry into Google's Gemini 1.5 Flash.[1]
4. **Agentic Knowledge Base (SR-RAG):** Employs Self-Reflective RAG (SR-RAG) via a multi-agent LangGraph workflow (Proposer, Refuter, Judge) grounded in NASM clinical guidelines to ensure the AI never hallucinates unsafe physical advice.[1]
5. **Gamification Engine:** Rewards users via the FitCoin Wallet using the formula $Total~Coins=(Reps\times Form\_Score)\times Velocity\_Multiplier$.[1] Integrates with the Open Food Facts API to power the reward marketplace.

---

## 🛠️ Tech Stack
*   **Frontend/UI:** Python, Streamlit
*   **Vision & Biomechanics:** OpenCV, MediaPipe (BlazePose), SMPL-X, NumPy
*   **Generative AI:** Google Vertex AI / Gemini 1.5 Flash API
*   **RAG & Orchestration:** LangGraph, LangChain, FAISS/ChromaDB
*   **Audio:** Pyttsx3 (Local TTS)
*   **External APIs:** Open Food Facts Python SDK (for supplement rewards)

---

## 📋 Master Implementation Plan (AI Agent TODO List)

*Agent Instructions: Execute the following steps sequentially. Ensure modularity by separating vision processing, LLM orchestration, and UI state management.*

### Phase 1: Environment Setup & Foundation
- [ ] **Step 1.1:** Initialize the project directory. Create `requirements.txt` with: `opencv-python`, `mediapipe`, `numpy`, `streamlit`, `google-generativeai`, `langchain`, `langgraph`, `pyttsx3`, `openfoodfacts`.
- [ ] **Step 1.2:** Establish the file structure:
    *   `app.py` (Streamlit UI)
    *   `vision_engine.py` (MediaPipe & Velocity logic)
    *   `smplx_spotcheck.py` (SMPL-X mesh integration)
    *   `rag_orchestrator.py` (LangGraph SR-RAG pipeline)
    *   `fitcoin_economy.py` (Gamification & Open Food Facts API)

### Phase 2: The Edge Perception Layer (Fast Loop)
- [ ] **Step 2.1:** In `vision_engine.py`, initialize `cv2.VideoCapture(0)` and `mediapipe.solutions.pose`.
- [ ] **Step 2.2:** Configure MediaPipe to return 3D world coordinates (`results.pose_world_landmarks`) utilizing the midpoint of the hips as the tracking origin.
- [ ] **Step 2.3:** Write mathematical helper functions to calculate joint angles using 3D coordinate dot products.
- [ ] **Step 2.4 (Velocity Engine):** Implement temporal tracking. Calculate the Y-axis displacement of the wrist/hips over time. Track the time-delta between the Concentric (Up) and Eccentric (Down) phases of the movement.[1]
- [ ] **Step 2.5 (Fatigue Flag):** Implement the Velocity-Based Training (VBT) algorithm: `IF Current_Rep_Time > 1.3 * Average_Rep_Time THEN Flag="Neuromuscular Fatigue"`.[1] Serialize all metrics into a lightweight JSON payload.

### Phase 3: The Pro Loop (SMPL-X Volumetric Spot-Checking)
- [ ] **Step 3.1:** In `smplx_spotcheck.py`, write a function that triggers every 5 seconds or upon a "Fatigue Flag".
- [ ] **Step 3.2:** Pass the captured frame to the SMPL-X model. Regress the pose and shape parameters to construct a 3D mesh.
- [ ] **Step 3.3:** Extract critical safety metrics from the 3D mesh (e.g., spinal curvature/torque) that are obscured in the 2D MediaPipe skeleton. Append this data to the JSON payload.

### Phase 4: Self-Reflective RAG (SR-RAG) Pipeline
- [ ] **Step 4.1:** In `rag_orchestrator.py`, initialize a local vector store (FAISS or Chroma) and populate it with simulated NASM clinical guidelines and physical therapy constraints.
- [ ] **Step 4.2:** Construct a LangGraph multi-agent loop.
    *   **Proposer Agent:** Proposes a coaching cue based on the JSON telemetry.
    *   **Refuter Agent:** Actively searches the vector database for contradictions to the proposed cue (Explicit adversarial refutation).[1]
    *   **Judge Agent:** Arbitrates the dispute. If the cue is deemed unsafe, it forces a refusal/regeneration.[1]

### Phase 5: Cloud Reasoning (Gemini 1.5 Flash)
- [ ] **Step 5.1:** Integrate the `google-generativeai` SDK.
- [ ] **Step 5.2:** Design the Chain-of-Thought System Prompt: "You are an elite AI coach. Step 1: Check spine neutrality. Step 2: Check knee alignment. Step 3: Check velocity. Step 4: Generate a 1-sentence vocal cue." [1]
- [ ] **Step 5.3:** Combine the JSON telemetry from Phase 2/3 with the verified clinical data from Phase 4. Request a structured JSON output containing: `Form_Score`, `Correction_Text`, and `Motivational_Text`.

### Phase 6: FitCoin Gamification & Shop API
- [ ] **Step 6.1:** In `fitcoin_economy.py`, implement the reward calculation logic: $Total~Coins=(Reps\times Form\_Score)\times Velocity\_Multiplier$.[1]
- [ ] **Step 6.2:** Utilize the `openfoodfacts` Python SDK (`api.product.get()` and text search functionality) to query dietary supplements (e.g., whey protein, creatine).
- [ ] **Step 6.3:** Parse the Open Food Facts JSON response to populate a mock "Rewards Store" where users can spend their FitCoins for simulated discounts on these exact health products.

### Phase 7: UI Assembly & Real-Time Audio
- [ ] **Step 7.1:** In `app.py`, build the Streamlit interface using `st.session_state` to persistently track the user's FitCoin wallet balance and inventory.[1]
- [ ] **Step 7.2:** Create a two-column layout: Column A handles the live OpenCV feed with the skeleton overlay; Column B displays real-time velocity metrics and form scores.[1]
- [ ] **Step 7.3:** Integrate the TTS Engine. Pass the `Correction_Text` generated by Gemini to a background audio thread so the AI verbally instructs the user without blocking the live video feed.