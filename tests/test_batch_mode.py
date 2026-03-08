import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pypdf import PdfReader, PdfWriter

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


def make_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as file:
        writer.write(file)


def make_png(path: Path) -> None:
    image = Image.new("RGB", (16, 16), (255, 255, 255))
    image.save(path, format="PNG")


class StubDownloadResponse:
    def __init__(self, content: str) -> None:
        self._content = content.encode("utf-8")

    def read(self) -> bytes:
        return self._content


class StubFiles:
    def __init__(self, batch_output_lines=None) -> None:
        self.batch_output_lines = batch_output_lines or []
        self.uploaded_page_counts = {}
        self.deleted = []
        self.batch_uploads = []

    def upload(self, file, purpose):
        data = file["content"].read()
        if purpose == "ocr":
            file_id = f"ocr_{len(self.uploaded_page_counts)}"
            self.uploaded_page_counts[file_id] = len(PdfReader(BytesIO(data)).pages)
            return SimpleNamespace(id=file_id)
        file_id = f"batch_{len(self.batch_uploads)}"
        self.batch_uploads.append(data.decode("utf-8"))
        return SimpleNamespace(id=file_id)

    def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)

    def download(self, file_id: str):
        if file_id == "batch-output":
            return StubDownloadResponse("\n".join(self.batch_output_lines))
        return StubDownloadResponse("")


class StubBatchJobs:
    def __init__(self, *, create_error: Exception | None = None) -> None:
        self.create_error = create_error
        self.created = []

    def create(self, **kwargs):
        if self.create_error:
            raise self.create_error
        self.created.append(kwargs)
        return SimpleNamespace(id="job-1", status="QUEUED")

    def get(self, job_id: str):
        return SimpleNamespace(
            id=job_id,
            status="SUCCESS",
            completed_requests=2,
            total_requests=2,
            output_file="batch-output",
            error_file=None,
        )


class StubBatch:
    def __init__(self, jobs: StubBatchJobs) -> None:
        self.jobs = jobs


class StubClient:
    def __init__(self, files: StubFiles, jobs: StubBatchJobs) -> None:
        self.files = files
        self.batch = StubBatch(jobs)


def test_process_batch_files_reports_free_trial_error(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    make_png(image_path)

    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", mode="batch")
    processor.client = StubClient(
        StubFiles(),
        StubBatchJobs(
            create_error=RuntimeError(
                'API error occurred: Status 402. Body: {"detail":"You cannot launch batch jobs this big with your free trial."}'
            )
        ),
    )
    processor.errors = []
    processor.processed_files = []

    success_count, total_count = processor._process_batch_files([image_path], tmp_path / "out")

    assert (success_count, total_count) == (0, 1)
    assert len(processor.errors) == 1
    assert "Try rerunning without `--mode batch`" in processor.errors[0]["error"]


def test_process_batch_files_reassembles_pdf_chunks(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, pages=3)

    batch_output_lines = [
        json.dumps(
            {
                "custom_id": "req-0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "pages": [
                            {"index": 0, "markdown": "page 1"},
                            {"index": 1, "markdown": "page 2"},
                        ]
                    },
                },
            }
        ),
        json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {"pages": [{"index": 0, "markdown": "page 3"}]},
                },
            }
        ),
    ]

    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", mode="batch", max_pages=0)
    processor.client = StubClient(StubFiles(batch_output_lines=batch_output_lines), StubBatchJobs())
    processor.errors = []
    processor.processed_files = []

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr("mistral_ocr.processor.PDF_REQUEST_PAGE_LIMIT", 2)

    success_count, total_count = processor._process_batch_files([pdf_path], output_dir)

    assert (success_count, total_count) == (1, 1)
    markdown = (output_dir / "sample.md").read_text()
    assert "## Page 1" in markdown
    assert "## Page 3" in markdown
    assert processor.client.files.deleted == ["batch_0", "ocr_0", "ocr_1"]
