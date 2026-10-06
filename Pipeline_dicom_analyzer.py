import os
import re
import base64
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-2.5-pro", "env": "GEMINI_API_KEY"},
    "openai": {"model": "gpt-5", "env": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-sonnet-5-5", "env": "ANTHROPIC_API_KEY"},
}


# ---------------------------------------------------------------------------
# DICOM -> PNG
# ---------------------------------------------------------------------------

class DicomConverter:
    """Handles the conversion of DICOM files to PNG format."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path, verbose: bool = False):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.verbose = verbose

    def process_batch(self) -> bool:
        """Walks through the directory and converts all .dcm files."""
        if not self.input_path.exists():
            print(f"❌ Error: Input directory '{self.input_path}' not found.")
            return False

        dicom_files = list(self.input_path.rglob("*.dcm"))

        if not dicom_files:
            print("⚠️ No .dcm files found. Skipping conversion.")
            return False

        print(f"🔍 Found {len(dicom_files)} DICOM files. Starting conversion...")

        for dcm_file in dicom_files:
            relative_path = dcm_file.parent.relative_to(self.input_path)
            target_folder = self.output_path / relative_path
            self._convert_dicom_to_png(dcm_file, target_folder)

        return True

    def _convert_dicom_to_png(self, dicom_path: Path, output_folder: Path) -> None:
        if self.verbose:
            print(f"🔄 Processing: {dicom_path.name}")

        try:
            ds = pydicom.dcmread(dicom_path)
        except Exception as e:
            print(f"❌ Failed to read {dicom_path}: {e}")
            return

        if 'PixelData' not in ds:
            if self.verbose:
                print(f"⏭️ Skipping {dicom_path.name}: No image pixels found.")
            return

        try:
            pixel_array = ds.pixel_array
        except Exception as e:
            print(f"❌ Failed to extract pixel data from {dicom_path.name}: {e}")
            return

        try:
            data = apply_voi_lut(pixel_array, ds)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ VOI LUT failed for {dicom_path.name}, using raw pixels: {e}")
            data = pixel_array

        if ds.get('PhotometricInterpretation') == "MONOCHROME1":
            data = np.amax(data) - data

        data = data.astype(float)
        if data.max() - data.min() != 0:
            data = (data - data.min()) / (data.max() - data.min()) * 255.0
        else:
            data = np.zeros(data.shape)

        data = data.astype(np.uint8)
        output_folder.mkdir(parents=True, exist_ok=True)
        base_name = dicom_path.stem

        if data.ndim == 2:
            self._save_image(data, output_folder / f"{base_name}.png")
        elif data.ndim == 3:
            if self.verbose:
                print(f"📂 Detected 3D volume ({data.shape[0]} frames) in {base_name}")
            for i, frame in enumerate(data):
                self._save_image(frame, output_folder / f"{base_name}_frame_{i:03d}.png")

    def _save_image(self, array: np.ndarray, output_path: Path) -> None:
        """Helper to save numpy array as image."""
        image = Image.fromarray(array)
        image.save(output_path)
        if self.verbose:
            print(f"✅ Saved Image: {output_path}")


# ---------------------------------------------------------------------------
# AI providers (one small adapter per vendor, same interface)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Provider-independent result: the model's reasoning (if any) and its answer."""
    thoughts: str
    text: str


class VisionProvider(ABC):
    """Common interface: send a prompt + PNG images, get back an AnalysisResult."""

    name: str = "base"

    def __init__(self, api_key: str, model_id: str, verbose: bool = False):
        self.api_key = api_key
        self.model_id = model_id
        self.verbose = verbose

    @abstractmethod
    def analyze(self, prompt: str, images: list[bytes]) -> AnalysisResult:
        """Send the prompt and PNG image bytes to the model."""


class GeminiProvider(VisionProvider):
    name = "gemini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from google import genai  # lazy import: only needed for this provider
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=self.api_key)

    def analyze(self, prompt: str, images: list[bytes]) -> AnalysisResult:
        types = self._types
        contents = [prompt] + [
            types.Part.from_bytes(data=img, mime_type="image/png") for img in images
        ]
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            ),
        )

        thoughts, final_text = "", ""
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thoughts += part.text
            else:
                final_text += part.text
        return AnalysisResult(thoughts=thoughts, text=final_text)


class OpenAIProvider(VisionProvider):
    name = "openai"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from openai import OpenAI  # lazy import
        self.client = OpenAI(api_key=self.api_key)

    def analyze(self, prompt: str, images: list[bytes]) -> AnalysisResult:
        content = [{"type": "input_text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            })

        request = dict(model=self.model_id, input=[{"role": "user", "content": content}])

        try:
            # Ask for a reasoning summary (only supported by reasoning models).
            response = self.client.responses.create(
                **request, reasoning={"effort": "medium", "summary": "auto"}
            )
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Reasoning options rejected by '{self.model_id}', retrying without: {e}")
            response = self.client.responses.create(**request)

        # OpenAI does not expose raw chain-of-thought, only (optional) summaries.
        thoughts = ""
        for item in response.output:
            if getattr(item, "type", None) == "reasoning":
                for summary in getattr(item, "summary", None) or []:
                    thoughts += getattr(summary, "text", "") + "\n\n"

        return AnalysisResult(thoughts=thoughts.strip(), text=response.output_text or "")


class AnthropicProvider(VisionProvider):
    name = "anthropic"

    THINKING_BUDGET = 8000
    MAX_TOKENS = 16000  # must be larger than the thinking budget

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import anthropic  # lazy import
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze(self, prompt: str, images: list[bytes]) -> AnalysisResult:
        # Anthropic recommends placing images before the text instruction.
        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode("utf-8"),
                },
            })
        content.append({"type": "text", "text": prompt})

        request = dict(
            model=self.model_id,
            max_tokens=self.MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
        )

        try:
            response = self.client.messages.create(
                **request,
                thinking={"type": "enabled", "budget_tokens": self.THINKING_BUDGET},
            )
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Extended thinking rejected by '{self.model_id}', retrying without: {e}")
            response = self.client.messages.create(**request)

        thoughts, final_text = "", ""
        for block in response.content:
            if block.type == "thinking":
                thoughts += block.thinking
            elif block.type == "text":
                final_text += block.text
        return AnalysisResult(thoughts=thoughts, text=final_text)


def build_provider(provider: str, api_key: str, model_id: str, verbose: bool = False) -> VisionProvider:
    """Factory returning the right provider adapter."""
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(providers)}")
    return providers[provider](api_key=api_key, model_id=model_id, verbose=verbose)


# ---------------------------------------------------------------------------
# Analysis agent (provider-agnostic)
# ---------------------------------------------------------------------------

class MedicalAnalysisAgent:
    """Groups images into cases, asks the selected AI provider to analyze them, saves reports."""

    def __init__(self, provider: VisionProvider, input_dir: str | Path, output_dir: str | Path, verbose: bool = False):
        self.provider = provider
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.verbose = verbose

    def run_analysis(self) -> None:
        """Main execution flow for the AI agent."""
        self._setup_folders()
        image_groups = self._group_images_by_id()

        if not image_groups:
            print(f"⚠️ No PNG files found in '{self.input_path}'.")
            return

        print(
            f"🧠 Found {len(image_groups)} cases. Starting AI analysis using "
            f"{self.provider.name} / '{self.provider.model_id}'..."
        )

        for group_id, files in image_groups.items():
            result = self._analyze_group(group_id, files)
            if result:
                self._save_report(group_id, result, files)

    def _setup_folders(self) -> None:
        """Ensures the output folder exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"📁 Reports will be saved to: {self.output_path.resolve()}")

    def _group_images_by_id(self) -> dict[str, list[Path]]:
        """Groups PNG files by their original DICOM filename."""
        groups: dict[str, list[Path]] = {}
        for img_file in sorted(self.input_path.rglob("*.png")):
            group_id = re.sub(r'_frame_\d+$', '', img_file.stem)
            groups.setdefault(group_id, []).append(img_file)
        return groups

    @staticmethod
    def _build_prompt(group_id: str) -> str:
        return (
            f"You are an expert radiologist AI analyzing Case {group_id}. "
            "Compare all provided views for abnormalities.\n\n"
            "You MUST output your final response STRICTLY following the exact Markdown structure below. "
            "Do not add any introductory or concluding text outside of this template.\n\n"
            "## 📝 Analysis\n"
            "**Analysis of Case:** [Provide a brief high-level overview]\n"
            "**Date of Exam:**[Extract if visible, otherwise state 'Not provided']\n"
            "**Modality:**[e.g., X-Ray, CT, MRI, Ultrasound]\n"
            "**Views Provided:** [List the specific views provided]\n\n"
            "**Step-by-step Visual Findings:**\n"
            "[Describe each provided image one by one in detail]\n\n"
            "**Comparison of Views and Abnormality Analysis:**\n"
            "[Compare the views and detail any abnormalities found]\n\n"
            "## Final report\n"
            "**Clinical information:** [Detail if apparent, otherwise state 'Not provided']\n"
            "**Findings:** [Provide formal detailed radiological findings]\n"
            "**Impression:**[Provide the clinical impression here. CRITICAL: This specific section must be written in plain language readable by a patient without any prior medical background.]\n"
            "**Recommendations:** [Suggested next actionable steps or 'None']\n"
            "**Exam assessment:** [Overall assessment of exam quality and findings]"
        )

    def _analyze_group(self, group_id: str, file_paths: list[Path]) -> AnalysisResult | None:
        """Sends a group of images to the selected provider for comparative analysis."""
        print(f"\n⚙️ Processing Case: {group_id} ({len(file_paths)} views)...")
        if self.verbose:
            print(f"   Files: {[f.name for f in file_paths]}")

        images = [path.read_bytes() for path in file_paths]

        try:
            if self.verbose:
                print(f"   Sending request to {self.provider.name} API...")
            return self.provider.analyze(self._build_prompt(group_id), images)
        except Exception as e:
            print(f"❌ API Error for Case {group_id}: {e}")
            return None

    def _save_report(self, group_id: str, result: AnalysisResult, file_list: list[Path]) -> None:
        """Saves the thinking and response to a Markdown file."""
        safe_id = re.sub(r'[^\w\-_\.]', '_', group_id)
        filename = f"report_{safe_id}.md"
        filepath = self.output_path / filename

        thoughts = result.thoughts.strip() or "No internal reasoning returned."
        quoted_thoughts = "\n".join(f"> {line}" for line in thoughts.splitlines())

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Analysis Report: Case {group_id}\n\n")
            f.write("> ⚠️ **DISCLAIMER: This report is generated by an Artificial Intelligence model for informational and research purposes only. It does NOT constitute professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.**\n\n")
            f.write(f"**Model:** {self.provider.name} / {self.provider.model_id}\n\n")
            f.write(f"**Source Images:** {', '.join(p.name for p in file_list)}\n")
            f.write("\n---\n")
            f.write("## 💭 Internal Thinking Process\n")
            f.write(f"{quoted_thoughts}\n")
            f.write("\n---\n")
            f.write(result.text.strip())

        print(f"✅ Saved Report: {filename}")


# ---------------------------------------------------------------------------
# Pipeline + CLI
# ---------------------------------------------------------------------------

class ImagingPipeline:
    """Orchestrates the conversion and analysis workflows."""

    def __init__(self, dicom_dir: str, png_dir: str, report_dir: str, provider: VisionProvider, verbose: bool = False):
        self.converter = DicomConverter(input_dir=dicom_dir, output_dir=png_dir, verbose=verbose)
        self.agent = MedicalAnalysisAgent(
            provider=provider,
            input_dir=png_dir,
            output_dir=report_dir,
            verbose=verbose,
        )

    def run(self):
        print("🚀 Starting Medical Imaging Pipeline...")
        print("-" * 40)

        # Step 1: Convert DICOMs to PNGs
        self.converter.process_batch()

        print("-" * 40)

        # Step 2: Analyze PNGs with the selected AI provider
        self.agent.run_analysis()

        print("-" * 40)
        print("🎉 Pipeline Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DICOM to PNG Converter and AI Analyzer Pipeline (Gemini, OpenAI or Anthropic)."
    )
    parser.add_argument("-i", "--input", default="Pipeline_dcm_analyzer/dicom_input", help="Directory containing input DICOM files")
    parser.add_argument("-p", "--png-output", default="Pipeline_dcm_analyzer/png_output", help="Directory to save converted PNG files")
    parser.add_argument("-r", "--report-output", default="Pipeline_dcm_analyzer/reports", help="Directory to save generated Markdown reports")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_DEFAULTS),
        default="gemini",
        help="AI provider to use (default: gemini)",
    )
    parser.add_argument(
        "-k", "--api-key",
        default=None,
        help="API key (defaults to GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY depending on --provider)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model ID (defaults: gemini-2.5-pro / gpt-5 / claude-sonnet-5-5 depending on --provider)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    defaults = PROVIDER_DEFAULTS[args.provider]
    api_key = args.api_key or os.environ.get(defaults["env"], "")
    model_id = args.model or defaults["model"]

    if not api_key:
        print(f"❌ Error: API key required for '{args.provider}'. Pass it via --api-key or set the {defaults['env']} environment variable.")
        exit(1)

    try:
        provider = build_provider(args.provider, api_key, model_id, verbose=args.verbose)
    except ImportError as e:
        print(f"❌ Error: missing Python package for '{args.provider}': {e}")
        print("   Install it with: pip install google-genai | openai | anthropic")
        exit(1)

    pipeline = ImagingPipeline(
        dicom_dir=args.input,
        png_dir=args.png_output,
        report_dir=args.report_output,
        provider=provider,
        verbose=args.verbose,
    )

    pipeline.run()
