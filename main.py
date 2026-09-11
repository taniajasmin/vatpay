from fastapi import FastAPI, UploadFile, Request, BackgroundTasks, Depends, HTTPException, status, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather
from fastapi.responses import FileResponse
import csv
import json
import os
import re
import time
import fcntl
from datetime import datetime
from threading import Lock, Timer
import requests
import queue
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt


from config import HUMAN_AGENT_NUMBER, COMMON_MESSAGE_TEXT

from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    BASE_URL,
    ELEVENLABS_API_KEY,
    VOICE_ID,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

app = FastAPI(title="VetPay Outbound Dialer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)



# ─── Paths ──────────────
CONTACTS_CSV     = "contacts.csv"
RESULTS_JSON     = "call_results.json"
SMS_REQUESTS_JSON = "sms_requests.json"
OUTPUT_CSV_DIR   = "output_results"
AUDIO_DIR        = "audio"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ─── Cross-process shared state ──────────────────────────────────────────
# When the app runs under multiple uvicorn/gunicorn workers (or multiple
# containers), module-level globals are per-process and `/stop-calls` cannot
# see the active call SID owned by the worker running the dialer. We persist
# the small set of fields that must be visible to every worker to a JSON file
# under an OS-level flock.
STATE_FILE = ".dialer_state.json"
STATE_LOCK = STATE_FILE + ".lock"

_DEFAULT_STATE = {
    "stop_requested": False,
    "active_call_sid": None,
    "running": False,
    "total": 0,
    "completed": 0,
    "current": {"phone": "", "name": "", "client": "", "status": "idle"},
}


def _state_read() -> dict:
    """Read shared state from disk; return defaults if missing or corrupt."""
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return {**_DEFAULT_STATE, **data}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_STATE)


def _state_write(updates: dict) -> dict:
    """Atomic read-modify-write of shared state under fcntl.flock."""
    with open(STATE_LOCK, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        state = _state_read()
        state.update(updates)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)
        return state

# ─── ElevenLabs TTS ────
SENTENCE_PAUSE = "0.5s"  # silence inserted after each sentence

def add_sentence_pauses(text: str, pause: str = SENTENCE_PAUSE) -> str:
    """Insert an ElevenLabs <break> tag after every sentence so the
    message doesn't race through the script."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return f'<break time="{pause}" /> '.join(s for s in sentences if s)

def generate_audio(text: str, output_path: str):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": add_sentence_pauses(text),
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75
        }
    }
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"Audio generated: {output_path}")
    else:
        print(f"ElevenLabs failed: {resp.status_code} - {resp.text}")
        raise Exception("TTS generation failed")

# ─── Pre-generate static / common audio files ─────────────────
COMMON_MESSAGE_PATH = os.path.join(AUDIO_DIR, "common_message_v4.mp3")


COMMON_TEXT = COMMON_MESSAGE_TEXT

# One-time generation of the long common part
if not os.path.exists(COMMON_MESSAGE_PATH):
    print("Generating common message audio (one-time task)...")
    generate_audio(COMMON_TEXT, COMMON_MESSAGE_PATH)
    print("Common message audio created.")
else:
    print("Common message audio already exists → skipping generation.")

# Other static phrases
static_texts = {
    "thank_you_goodbye_v4": "Thank you for your time. Goodbye.",
    "please_hold_v4": "Please hold while I transfer you to a VetPay representative.",
    "sms_confirm_v5": "Thank you, you will receive a SMS shortly. Goodbye."
}

for key, txt in static_texts.items():
    path = os.path.join(AUDIO_DIR, f"{key}.mp3")
    if not os.path.exists(path):
        generate_audio(txt, path)

# Sequential calling — 1 call at a time, 5-second gap between calls
CALL_GAP_SECONDS = 5

call_queue = queue.Queue()
next_call_lock = Lock()
is_calling = False


def start_next_call():
    global is_calling

    # Read the stop flag from shared state — another worker may have set it.
    if _state_read()["stop_requested"]:
        print("Stop requested. No more calls will be made.")
        return

    with next_call_lock:
        if is_calling or call_queue.empty():
            return
        is_calling = True

    try:
        phone, name, client_id = call_queue.get_nowait()
        print(f"[OUT] Calling: {phone} ({name}) - Client: {client_id}")

        current_call_info["phone"] = phone
        current_call_info["name"] = name
        current_call_info["client"] = client_id
        current_call_info["status"] = "dialing"
        _state_write({"current": dict(current_call_info)})

        call_log.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "phone": phone,
            "name": name,
            "client": client_id,
            "status": "dialing"
        })
        if len(call_log) > 50:
            call_log.pop()

        call = twilio.calls.create(
            to=phone,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{BASE_URL}/twilio/voice?phone={phone}",
            status_callback=f"{BASE_URL}/twilio/status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            machine_detection="DetectMessageEnd",
            machine_detection_timeout=30,
            machine_detection_speech_threshold=2400,
            machine_detection_speech_end_threshold=1200,
            machine_detection_silence_timeout=5000,
        )
        # Publish the SID to shared state so /stop-calls on any worker can
        # find and hang up this call.
        _state_write({"active_call_sid": call.sid})
    except queue.Empty:
        # Queue was drained (e.g. Stop pressed) between the check and the get
        with next_call_lock:
            is_calling = False
    except Exception as e:
        print(f"[ERROR] Failed to call {phone}: {e}")
        current_call_info["status"] = "failed"
        _state_write({"current": dict(current_call_info)})
        with next_call_lock:
            is_calling = False


stop_requested = False
active_call_sid = None  # Twilio SID of the call currently in flight

# Global state
call_tracker = {
    "total": 0,
    "completed": 0,
    "running": False,
    "lock": Lock()
}

current_call_info = {
    "phone": "",
    "name": "",
    "client": "",
    "status": "idle"  # idle | dialing | ringing | connected | completed
}

call_log = []  # recent activity entries

contact_map: dict[str, dict] = {}

def normalize_phone(p: str) -> str:
    if not p:
        return ""
    cleaned = ''.join(c for c in str(p).strip() if c.isdigit() or c == '+')
    if cleaned.count('+') > 1:
        cleaned = '+' + cleaned.replace('+', '')
    if not cleaned.startswith('+'):
        cleaned = '+' + cleaned
    if cleaned.startswith('+88') and len(cleaned) == 13 and cleaned[3] != '0':
        cleaned = '+880' + cleaned[3:]
    print(f"Normalized: '{p}' → '{cleaned}'")
    return cleaned

def load_contacts_to_memory():
    global contact_map
    contact_map.clear()
    if not os.path.exists(CONTACTS_CSV):
        return 0
    count = 0
    with open(CONTACTS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phone = normalize_phone(row.get("Phone", ""))
            if phone:
                contact_map[phone] = {
                    "name": row.get("Name", "").strip() or "there",
                    "client": row.get("Client", "").strip()
                }
                print(f"Stored contact: {phone} → {contact_map[phone]}")
                count += 1
    return count

# Utils
def save_result(phone: str, name: str, result: str, caller_input: str = ""):
    results = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
        except:
            pass

    # DO NOT overwrite a final in-call outcome (transfer or SMS requested)
    FINAL_RESULTS = {"successfully_transferred", "sms_requested"}
    if results.get(phone, {}).get("result") in FINAL_RESULTS:
        return

    results[phone] = {
        "name": name,
        "phone": phone,
        "result": result,
        "input": caller_input,  # how they chose: "Pressed 1" / 'Said: "text me"'
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def generate_final_output_csv():
    if not os.path.exists(CONTACTS_CSV):
        print("No contacts.csv found")
        return

    results = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
        except:
            pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_CSV_DIR, f"call_results_{ts}.csv")

    # Plain-language label for the option the caller chose
    SELECTION_LABELS = {
        "sms_requested": "SMS payment link (1)",
        "successfully_transferred": "Speak with team (2)",
    }

    with open(CONTACTS_CSV, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        base_fields = reader.fieldnames or ["Client", "Name", "Phone"]
        fieldnames = list(base_fields)
        for col in ("Response", "Selection", "Input"):
            if col not in fieldnames:
                fieldnames = fieldnames + [col]

        rows = []
        for row in reader:
            ph = normalize_phone(row.get("Phone", ""))
            entry = results.get(ph, {})
            resp = entry.get("result", "")
            # Rebuild from known columns only — drops the None restkey that
            # csv.DictReader adds for ragged rows (extra unquoted commas),
            # which would otherwise crash DictWriter.
            new_row = {k: (row.get(k) or "") for k in base_fields}
            new_row["Response"] = resp
            new_row["Selection"] = SELECTION_LABELS.get(resp, "")
            new_row["Input"] = entry.get("input", "")
            rows.append(new_row)

    with open(out_path, "w", newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output CSV created: {out_path}")

# Endpoints
# @app.get("/")
# def serve_home():
#     return FileResponse("index.html")

@app.post("/upload-contacts")
async def upload_contacts(file: UploadFile):
    global is_calling

    if _state_read()["running"]:
        return {"error": "Cannot upload while calls are running. Stop calls first."}

    content = (await file.read()).decode('utf-8').splitlines()
    reader = csv.DictReader(content)

    required = {"Client", "Name", "Phone"}
    if not required.issubset(reader.fieldnames or []):
        return {"error": f"Missing columns: {required - set(reader.fieldnames or [])}"}

    with open(CONTACTS_CSV, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow({k: (v or "").strip() for k, v in row.items()})

    if os.path.exists(RESULTS_JSON):
        os.remove(RESULTS_JSON)
    if os.path.exists(SMS_REQUESTS_JSON):
        os.remove(SMS_REQUESTS_JSON)

    count = load_contacts_to_memory()

    # Reset all shared + local state for fresh daily run
    _state_write({
        "stop_requested": False,
        "active_call_sid": None,
        "running": False,
        "total": 0,
        "completed": 0,
        "current": {"phone": "", "name": "", "client": "", "status": "idle"},
    })
    with call_tracker["lock"]:
        call_tracker.update({"total": 0, "completed": 0, "running": False})

    is_calling = False
    current_call_info.update({"phone": "", "name": "", "client": "", "status": "idle"})
    call_log.clear()
    while not call_queue.empty():
        call_queue.get()

    return {"message": "Contacts uploaded", "count": count}

@app.post("/start-calls")
def start_calls(background_tasks: BackgroundTasks):
    global is_calling

    if _state_read()["running"]:
        return {"error": "Already running"}
    if not os.path.exists(CONTACTS_CSV):
        return {"error": "Upload contacts first"}

    count = load_contacts_to_memory()
    if count == 0:
        return {"error": "No contacts"}

    # Publish the new run to shared state BEFORE launching the background task
    # so any worker polling /live-status sees a consistent picture.
    _state_write({
        "stop_requested": False,
        "active_call_sid": None,
        "running": True,
        "total": count,
        "completed": 0,
        "current": {"phone": "", "name": "", "client": "", "status": "idle"},
    })
    with call_tracker["lock"]:
        call_tracker["total"] = count
        call_tracker["completed"] = 0
        call_tracker["running"] = True

    is_calling = False
    current_call_info.update({"phone": "", "name": "", "client": "", "status": "idle"})
    call_log.clear()

    background_tasks.add_task(run_outbound_calls)
    return {"status": "started", "total": count}

def run_outbound_calls():
    try:
        while not call_queue.empty():
            call_queue.get()

        # Fresh batch: reset stop flag in shared state so a previous Stop
        # doesn't carry over into a new /start-calls run.
        _state_write({"stop_requested": False, "active_call_sid": None})

        with open(CONTACTS_CSV, newline='', encoding='utf-8') as f:
            if _state_read()["stop_requested"]:
                return

            reader = csv.DictReader(f)
            for row in reader:
                if _state_read()["stop_requested"]:
                    print("Stop requested — no more calls will be queued.")
                    return

                phone = normalize_phone(row.get("Phone", ""))
                name = row.get("Name", "").strip() or "there"
                client = row.get("Client", "").strip()

                if phone and client:
                    # Generate hello audio per PHONE (not client)
                    hello_path = os.path.join(AUDIO_DIR, f"hello_{phone}_v4.mp3")

                    if not os.path.exists(hello_path):
                        hello_text = f"Hi {name},"
                        generate_audio(hello_text, hello_path)

                    call_queue.put((phone, name, client))

        # Fire the first batch of concurrent calls
        start_next_call()

    except Exception as e:
        print(f"[ERROR] run_outbound_calls failed: {e}")
        _state_write({"running": False})


@app.post("/stop-calls")
def stop_calls():
    global is_calling

    # 1) Flip the stop flag in shared state FIRST. Every worker reads this
    # before dialling the next contact, so any worker running the dialer will
    # see the flag on its next pass.
    _state_write({"stop_requested": True})

    # 2) Poll the shared state briefly for the active Twilio call SID. The
    # worker that placed the call publishes the SID to shared state right
    # after `twilio.calls.create(...)` returns, but a tiny race window exists
    # where Stop may arrive before that write completes. Polling up to ~2s
    # closes that window without making Stop feel slow.
    active_sid = None
    for _ in range(20):
        active_sid = _state_read().get("active_call_sid")
        if active_sid:
            break
        time.sleep(0.1)

    # 3) Hang up the call that is currently in flight
    if active_sid:
        try:
            twilio.calls(active_sid).update(status="completed")
            print(f"[STOP] Terminated active call: {active_sid}")
        except Exception as e:
            # Call may have already ended on its own — safe to ignore
            print(f"[STOP] Could not terminate call {active_sid}: {e}")

    # 4) Clear SID + reset current call so the dashboard returns to idle
    _state_write({
        "active_call_sid": None,
        "running": False,
        "current": {"phone": "", "name": "", "client": "", "status": "idle"},
    })

    # 5) Drain the in-memory queue held by THIS worker. (Other workers may
    # have their own empty queue copies — that's fine; the stop flag will
    # keep them from dialling anything new.)
    with next_call_lock:
        while not call_queue.empty():
            try:
                call_queue.get_nowait()
            except queue.Empty:
                break
        is_calling = False

    # 6) Sync in-memory caches for the local worker
    current_call_info.update({"phone": "", "name": "", "client": "", "status": "idle"})
    with call_tracker["lock"]:
        call_tracker["running"] = False

    return {"status": "stopped"}



@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    phone = normalize_phone(request.query_params.get("phone"))
    form = await request.form()
    answered_by = form.get("AnsweredBy", request.query_params.get("AnsweredBy", "unknown"))

    vr = VoiceResponse()

    if not phone:
        vr.say("System error. Goodbye.")
        return Response(str(vr), media_type="application/xml")

    contact = contact_map.get(phone)
    name = contact["name"] if contact else "customer"

    # ── Voicemail / machine detected: save result and hang up ──
    if answered_by in ("machine_start", "machine_end_beep", "machine_end_silence", "fax"):
        print(f"[AMD] Voicemail detected for {phone} ({answered_by})")
        save_result(phone, name, "voicemail")
        vr.hangup()
        return Response(str(vr), media_type="application/xml")

    # ── Human or unknown: play the full call flow ──
    # IMPORTANT: audio must be INSIDE the <Gather>. Twilio does NOT buffer
    # DTMF tones pressed during top-level <Play> verbs — if the caller presses
    # 1/2 while the prompt is playing (the natural moment, right after hearing
    # "...press 1..."), the digit is discarded because the Gather hasn't
    # started yet. Nesting the <Play>s inside the Gather makes the digit
    # capture active while the prompt is playing.
    gather = Gather(
        input="speech dtmf",
        speech_timeout="auto",
        timeout=20,
        num_digits=1,
        action=f"{BASE_URL}/twilio/transfer?phone={phone}",
        method="POST"
    )
    # 1) Personalized greeting
    gather.play(f"{BASE_URL}/audio/hello_{phone}_v4.mp3")
    # 2) Full script (press-1 / press-2 prompt)
    gather.play(f"{BASE_URL}/audio/common_message_v4.mp3")

    vr.append(gather)

    # 3) If nothing was pressed at all
    vr.play(f"{BASE_URL}/audio/thank_you_goodbye_v4.mp3")

    return Response(str(vr), media_type="application/xml")



def save_sms_request(phone: str, name: str, client: str):
    """Record that a caller asked for the payment link by SMS.

    Nothing is sent automatically — the dashboard shows these requests in a
    dedicated panel and the team sends the link manually later.
    Keyed by phone so a retried call can't create duplicate rows.
    """
    requests_map = {}
    if os.path.exists(SMS_REQUESTS_JSON):
        try:
            with open(SMS_REQUESTS_JSON, "r", encoding="utf-8") as f:
                requests_map = json.load(f)
        except Exception:
            pass

    requests_map[phone] = {
        "name": name,
        "phone": phone,
        "client": client,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(SMS_REQUESTS_JSON, "w", encoding="utf-8") as f:
        json.dump(requests_map, f, indent=2)
    print(f"[SMS REQUEST] {name} ({phone}) asked for a payment link by SMS")


@app.post("/twilio/transfer")
async def transfer_call(request: Request):
    form = await request.form()

    phone_raw = request.query_params.get("phone")
    phone = normalize_phone(phone_raw)

    digits = form.get("Digits")
    speech = (form.get("SpeechResult") or "").lower()

    print(f"[TRANSFER] phone={phone} digits={digits!r} speech={speech!r}")

    contact = contact_map.get(phone)
    name = contact["name"] if contact else "customer"

    # Word-level matching (not substring) so words like "money"/"context"
    # don't accidentally trigger the "one"/"text" intents.
    spoken_words = set(
        w.strip(".,!?'\"") for w in speech.split()
    )

    # Press 1 / say "text me" → send payment link by SMS
    wants_sms = (
        digits == "1" or
        bool(spoken_words & {
            "text", "texts", "sms", "message", "messages",
            "link", "send", "one"
        })
    )

    # Press 2 / say "transfer" → speak with a member of the team
    wants_transfer = (
        digits == "2" or
        bool(spoken_words & {
            "transfer", "agent", "human", "person", "speak", "talk",
            "operator", "representative", "connect", "two"
        })
    )

    # Human-readable record of HOW the caller responded, for the report.
    # Punctuation (commas/quotes) is stripped so the CSV and the dashboard's
    # simple parser stay safe.
    if digits:
        caller_input = f"Pressed {digits}"
    elif speech:
        cleaned = " ".join(re.sub(r"[^a-z0-9 ]", " ", speech).split())[:60]
        caller_input = f"Said: {cleaned}" if cleaned else "No response"
    else:
        caller_input = "No response"

    vr = VoiceResponse()

    if wants_sms:
        client = contact["client"] if contact else ""
        save_sms_request(phone, name, client)
        save_result(phone, name, "sms_requested", caller_input)
        vr.play(f"{BASE_URL}/audio/sms_confirm_v5.mp3")
    elif wants_transfer:
        save_result(phone, name, "successfully_transferred", caller_input)
        vr.play(f"{BASE_URL}/audio/please_hold_v4.mp3")
        vr.dial(HUMAN_AGENT_NUMBER)
    else:
        # Neither option chosen (e.g. pressed 3 or said something unrelated) —
        # record what they did so the report shows it.
        save_result(phone, name, "completed_no_transfer", caller_input)
        vr.play(f"{BASE_URL}/audio/thank_you_goodbye_v4.mp3")

    return Response(str(vr), media_type="application/xml")



@app.post("/twilio/status")
async def call_status(request: Request):
    global is_calling
    form = await request.form()

    phone_raw = form.get("To") or form.get("Called") or form.get("From")
    phone = normalize_phone(phone_raw)

    status = form.get("CallStatus")
    duration = int(form.get("CallDuration") or 0)

    print(f"STATUS → {phone_raw} → {phone} | {status} | {duration}s")

    contact = contact_map.get(phone)
    if not contact:
        print(f"[WARN] No contact found for {phone_raw}")
        return "ok"

    name = contact["name"]

    # ── Update live call tracking ──
    if status in ("ringing", "answered", "completed", "no-answer", "busy", "failed", "canceled"):
        # Ignore stale callbacks (e.g. the hang-up confirmation arriving after Stop
        # already reset the dashboard to idle)
        if current_call_info["phone"] == phone:
            current_call_info["status"] = status
            _state_write({"current": dict(current_call_info)})
        for entry in call_log:
            if entry["phone"] == phone and entry["status"] in ("dialing", "ringing", "answered"):
                entry["status"] = status
                break

    # ── check if already has a result (transferred or voicemail) ──
    existing_result = ""
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
                existing_result = results.get(phone, {}).get("result", "")
        except:
            pass

    # ── save result only if no result was saved yet ──
    if not existing_result:
        if status == "no-answer":
            save_result(phone, name, "no_answer")
        elif status == "busy":
            save_result(phone, name, "busy")
        elif status in ["failed", "canceled"]:
            save_result(phone, name, status)
        elif status == "completed":
            save_result(phone, name, "completed_no_transfer")


    # ── Only count & continue on FINAL statuses ──
    final_statuses = {"completed", "no-answer", "busy", "failed", "canceled"}
    if status in final_statuses:
        # Call is over — nothing to hang up on Stop. Clear the SID in shared
        # state so /stop-calls on any worker doesn't try to terminate it.
        _state_write({"active_call_sid": None})

        with call_tracker["lock"]:
            call_tracker["completed"] += 1
            done = call_tracker["completed"]
            total = call_tracker["total"]
            print(f"Progress: {done}/{total}")

            if done >= total and total > 0:
                print("All calls finished → generating output CSV")
                generate_final_output_csv()
                call_tracker["running"] = False
                current_call_info["status"] = "idle"
                current_call_info["phone"] = ""
                current_call_info["name"] = ""
                current_call_info["client"] = ""
                _state_write({
                    "running": False,
                    "completed": done,
                    "current": dict(current_call_info),
                })
            else:
                _state_write({"completed": done})

        with next_call_lock:
            is_calling = False

        # Wait 5 seconds, then call the next contact. The shared stop flag
        # is consulted inside start_next_call, so a Stop request that arrives
        # during the gap will abort the next dial.
        Timer(CALL_GAP_SECONDS, start_next_call).start()

    return "ok"

@app.get("/result-csv")
def result_csv():
    if not os.path.exists(OUTPUT_CSV_DIR):
        return {"status": "processing"}

    files = [
        f for f in os.listdir(OUTPUT_CSV_DIR)
        if f.endswith(".csv")
    ]

    if not files:
        return {"status": "processing"}

    # latest generated CSV
    latest_file = max(
        files,
        key=lambda f: os.path.getctime(os.path.join(OUTPUT_CSV_DIR, f))
    )

    file_path = os.path.join(OUTPUT_CSV_DIR, latest_file)

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=latest_file
    )


@app.get("/call-progress")
def call_progress():
    s = _state_read()
    return {
        "total": s["total"],
        "completed": s["completed"]
    }


@app.get("/live-status")
def live_status():
    s = _state_read()
    return {
        "total": s["total"],
        "completed": s["completed"],
        "running": s["running"],
        "current": s["current"],
        "log": call_log[:30]
    }


@app.get("/sms-requests")
def sms_requests():
    """Callers who pressed 1 / asked for the payment link by SMS.

    The team sends the link manually — this is the work list.
    """
    data = {}
    if os.path.exists(SMS_REQUESTS_JSON):
        try:
            with open(SMS_REQUESTS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    requests_list = sorted(
        data.values(),
        key=lambda r: r.get("timestamp", ""),
        reverse=True
    )
    return {"requests": requests_list}




# --- SQLite Setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User Model
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# Create the database file
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# Note: no default admin is seeded — accounts are managed directly in users.db

def hash_password(password: str) -> str:
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Check if the provided password matches the stored hash
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# Custom dependency to get user from Cookie
# This handles API security
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    try:
        # 1. Verify the signature and expiration using your SECRET_KEY
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
            
    except JWTError:
        # This triggers if the token is fabricated, expired, or tampered with
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    # 2. Double check the database to ensure the user still exists
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
        
    return username


# Route to serve the Login Page
@app.get("/login", response_class=HTMLResponse)
async def get_login():
    return FileResponse("login.html")

# Route to serve the Dashboard (index.html)
@app.get("/index.html", response_class=HTMLResponse)
async def get_dashboard(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    
    if not token:
        return RedirectResponse(url="/login")

    try:
        # Verify the token is real
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = db.query(UserDB).filter(UserDB.username == username).first()
        
        if not user:
            return RedirectResponse(url="/login")
            
        # If we reach here, the token is 100% valid and verified
        return FileResponse("index.html")
        
    except JWTError:
        # Token was fabricated or expired! Clear it and send back to login
        response = RedirectResponse(url="/login")
        response.delete_cookie("access_token")
        return response

# Route to redirect the root (/) to the index page
@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/index.html")

# --- AUTH ENDPOINTS ---

@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    
    # Verify user exists and password hash matches
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = jwt.encode(
        {"sub": username, "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)},
        SECRET_KEY, algorithm=ALGORITHM
    )
    
    response = JSONResponse(content={"message": "Logged in"})
    response.set_cookie(
        key="access_token", 
        value=access_token, 
        httponly=True, 
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    return response


@app.post("/update-account")
async def update_account(
    current_password: str = Form(...), 
    new_username: str = Form(None),
    new_password: str = Form(None),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(UserDB).filter(UserDB.username == current_user).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 1. ALWAYS verify the current password before making any changes
    if not verify_password(current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password incorrect")

    # 2. Handle Username Update
    if new_username and new_username != user.username:
        # Check if the new username is already taken by someone else
        existing_user = db.query(UserDB).filter(UserDB.username == new_username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = new_username

    # 3. Handle Password Update
    if new_password:
        user.hashed_password = hash_password(new_password)

    db.commit()
    return {"message": "Account updated successfully"}

@app.post("/logout")
async def logout():
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie("access_token")
    return response
