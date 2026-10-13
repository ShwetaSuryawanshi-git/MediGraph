"""
experiments/llm_wrapper.py
--------------------------
Robust, cached, rate-limited wrapper for Google Gemini free-tier models.
Complies with paper constraints:
1. ONLY free Gemini API tier (Google AI Studio key in GEMINI_API_KEY).
2. At runtime, discovers available models and picks Flash or Flash-Lite (never Pro).
3. Temperature 0.0, minimal thinking.
4. Disk caching keyed by sha256(model, prompt, settings).
5. Rate-limiting (default 15 RPM throttling) with exponential backoff on 429 errors.
6. Never prints or commits the API key.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load .env file from project root or experiments dir if present
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

CACHE_DIR = Path(__file__).parent / "cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path(__file__).parent / "results" / "llm_config.json"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

class GeminiWrapper:
    def __init__(self, rpm_limit: float = 12.0, min_interval: float = 5.0):
        """
        rpm_limit: Requests per minute throttle (free tier is typically 15 RPM max)
        min_interval: minimum seconds between API calls to prevent 429 burst errors
        """
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment or .env file! "
                "Please set GEMINI_API_KEY before running LLM experiments."
            )
        
        self.min_interval = min_interval
        self.last_call_time = 0.0
        self.client = None
        self._init_client()
        self.generator_model, self.judge_model = self._discover_and_select_models()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.sdk_type = "google-genai"
        except ImportError:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self.client = gai
                self.sdk_type = "google-generativeai"
            except ImportError:
                raise ImportError("Neither `google-genai` nor `google-generativeai` is installed in this venv.")

    def _discover_and_select_models(self) -> tuple[str, Optional[str]]:
        """
        Queries the API to list available models, filters for free Flash/Flash-Lite models,
        selects the primary generator and a distinct judge model if available.
        Saves exact configuration to results/llm_config.json.
        """
        available_flash = []
        try:
            if self.sdk_type == "google-genai":
                pager = self.client.models.list()
                for m in pager:
                    name = getattr(m, "name", "")
                    disp = getattr(m, "display_name", "")
                    full = f"{name} {disp}".lower()
                    # Filter: Flash or Flash-Lite, NEVER Pro
                    if "flash" in full and "pro" not in full:
                        # Clean model id
                        mid = name.replace("models/", "")
                        available_flash.append(mid)
            else:
                for m in self.client.list_models():
                    name = m.name
                    full = f"{name} {m.display_name}".lower()
                    if "generateContent" in m.supported_generation_methods and "flash" in full and "pro" not in full:
                        mid = name.replace("models/", "")
                        available_flash.append(mid)
        except Exception as e:
            # Fallback to standard known Flash models if listing fails
            print(f"[LLM Wrapper] Model discovery warning: {e}. Falling back to default Flash names.")
            available_flash = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-lite"]

        # Deduplicate while preserving order
        unique_flash = []
        for f in available_flash:
            if f not in unique_flash:
                unique_flash.append(f)

        print(f"[LLM Wrapper] Discovered Flash-family candidate models: {unique_flash}")
        
        def probe_model(mid: str) -> bool:
            try:
                if self.sdk_type == "google-genai":
                    self.client.models.generate_content(model=mid, contents="ping")
                else:
                    g = self.client.GenerativeModel(model_name=mid)
                    g.generate_content("ping")
                return True
            except Exception:
                return False

        # Selection logic: Strictly use high-quota Flash models (1,500 req/day)
        # Avoid experimental or 20 req/day capped models like 3.5-flash and 3.8-flash
        preferred_gen = ["gemini-3.6-flash", "gemini-flash-latest"]
        generator = None
        for pref in preferred_gen:
            matches = [m for m in unique_flash if m == pref]
            for cand in matches:
                if probe_model(cand):
                    generator = cand
                    break
            if generator:
                break

        if not generator:
            generator = "gemini-3.6-flash"

        # Selection logic: Distinct model for judge to reduce self-preference bias
        # Prefer flash-lite models with full free-tier quota
        judge = None
        preferred_judge = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
        for pref in preferred_judge:
            matches = [m for m in unique_flash if (m == pref or pref in m) and m != generator]
            for cand in matches:
                if probe_model(cand):
                    judge = cand
                    break
            if judge:
                break
        
        if not judge:
            judge = "gemini-3.1-flash-lite"
            for cand in unique_flash:
                if cand != generator and "image" not in cand and "tts" not in cand:
                    if probe_model(cand):
                        judge = cand
                        break

        config_data = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator_model": generator,
            "judge_model": judge,
            "judge_type": "distinct_model" if judge else "automatic (same-model judge)",
            "temperature": 0.0,
            "thinking_config": "minimal_or_disabled",
            "all_discovered_flash_models": unique_flash,
            "rate_limit_min_interval_seconds": self.min_interval
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"[LLM Wrapper] Generator selected: {generator}")
        print(f"[LLM Wrapper] Judge selected: {judge or generator} ({config_data['judge_type']})")
        return generator, judge

    def _get_cache_path(self, model: str, prompt: str, settings: Dict[str, Any]) -> Path:
        payload = f"{model}:{prompt}:{json.dumps(settings, sort_keys=True)}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{h}.json"

    def _throttle(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        max_retries: int = 10
    ) -> str:
        """
        Executes a prompt against Gemini with caching, throttling, exponential backoff,
        and automatic model fallback on 503 spikes.
        """
        chosen_model = model or self.generator_model
        settings = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "system_instruction": system_instruction or ""
        }
        cache_path = self._get_cache_path(chosen_model, prompt, settings)

        # Check disk cache
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    return cached_data["text"]
            except Exception:
                pass  # Corrupted cache, re-fetch

        fallback_map = {
            "gemini-3.6-flash": "gemini-3.1-flash-lite",
            "gemini-3.1-flash-lite": "gemini-3.6-flash",
            "gemini-flash-latest": "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest": "gemini-3.1-flash-lite",
            "gemini-3.8-flash": "gemini-3.1-flash-lite",
            "gemini-3.5-flash": "gemini-3.1-flash-lite",
        }
        current_model = chosen_model

        # Call API with throttling and exponential backoff
        backoff = 5.0
        for attempt in range(max_retries):
            self._throttle()
            try:
                if self.sdk_type == "google-genai":
                    from google.genai import types
                    kwargs = {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    }
                    if system_instruction:
                        kwargs["system_instruction"] = system_instruction
                    
                    # Only pass thinking_config for models that explicitly support thinking (never lite models)
                    if "lite" not in current_model.lower():
                        try:
                            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                        except Exception:
                            pass

                    cfg = types.GenerateContentConfig(**kwargs)
                    try:
                        resp = self.client.models.generate_content(
                            model=current_model,
                            contents=prompt,
                            config=cfg
                        )
                    except Exception as call_err:
                        if "thinking" in str(call_err).lower() or "400" in str(call_err):
                            # Retry immediately without thinking_config if rejected
                            kwargs.pop("thinking_config", None)
                            cfg_plain = types.GenerateContentConfig(**kwargs)
                            resp = self.client.models.generate_content(
                                model=current_model,
                                contents=prompt,
                                config=cfg_plain
                            )
                        else:
                            raise call_err

                    text = resp.text if resp and resp.text else ""
                else:
                    gmodel = self.client.GenerativeModel(
                        model_name=current_model,
                        system_instruction=system_instruction,
                        generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens}
                    )
                    resp = gmodel.generate_content(prompt)
                    text = resp.text if resp and resp.text else ""

                # Write to disk cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": current_model,
                        "prompt": prompt,
                        "settings": settings,
                        "text": text,
                        "timestamp": time.time()
                    }, f, indent=2)

                return text

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                is_server_error = "503" in err_str or "unavailable" in err_str or "overloaded" in err_str
                is_daily_cap = "perday" in err_str or "daily" in err_str or "limit: 20" in err_str

                if (is_daily_cap or is_server_error or attempt >= 1) and (is_rate_limit or is_server_error) and current_model in fallback_map:
                    alt = fallback_map[current_model]
                    print(f"[LLM Wrapper] Switching from {current_model} to fallback {alt} due to error/quota...")
                    current_model = alt

                if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                    wait_time = max(backoff, 5.0)
                    print(f"[LLM Wrapper] {('Rate limit 429' if is_rate_limit else 'Service 503')} on attempt {attempt+1}/{max_retries}. Backing off for {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    backoff = min(backoff * 1.5, 30.0)
                elif attempt < max_retries - 1:
                    print(f"[LLM Wrapper] Error '{e}' on attempt {attempt+1}/{max_retries}. Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 30.0)
                else:
                    raise RuntimeError(f"Failed to generate after {max_retries} attempts: {e}")

        return ""
