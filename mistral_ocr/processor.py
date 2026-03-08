"""Core OCR processing module using Mistral AI."""

import json
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from mistralai import Mistral
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import Config
from .utils import (
    create_data_uri,
    determine_output_path,
    format_file_size,
    get_image_base64_data,
    get_pdf_page_count,
    get_supported_files,
    load_metadata,
    sanitize_filename,
    save_base64_image,
    save_metadata,
    split_pdf_into_chunks,
)

console = Console()
PDF_REQUEST_PAGE_LIMIT = 1000
BATCH_POLL_INTERVAL_SECONDS = 5


def _is_retryable_ocr_error(error: Exception) -> bool:
    """Return True when an OCR error is likely transient."""
    error_str = str(error).lower()
    retry_markers = ("429", "rate limit", "rate_limit", "timeout", "502", "503", "504")
    return any(marker in error_str for marker in retry_markers)


class OCRProcessor:
    """OCR processor using Mistral AI API."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        try:
            self.client = Mistral(api_key=config.api_key)
        except Exception as e:
            console.print(f"[red]Failed to initialize Mistral client: {e}[/red]")
            raise
        self.errors: List[Dict[str, str]] = []
        self.processed_files: List[Dict[str, Any]] = []

    def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single file with OCR."""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if self.config.verbose:
                console.print(f"[dim]File size: {file_size_mb:.2f} MB[/dim]")

            if not hasattr(self.client, "ocr"):
                raise AttributeError(
                    "OCR endpoint not available in Mistral client. "
                    "Please ensure you have the latest mistralai package "
                    "and OCR access enabled for your API key."
                )

            if self.config.verbose:
                console.print("[dim]Sending to Mistral OCR API...[/dim]")
                console.print(f"[dim]Model: {self.config.model}[/dim]")

            if file_path.suffix.lower() == ".pdf":
                response = self._process_pdf_file(file_path)
            else:
                self.config.validate_file_size(file_path)
                response = self._process_image_file(file_path)

            return {
                "file_path": file_path,
                "response": response,
                "success": True,
            }

        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            if self.config.verbose:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.errors.append({
                "file": str(file_path),
                "error": str(e),
            })
            return None

    def _process_with_retry(self, document: Any) -> object:
        """Call Mistral OCR with bounded retry/backoff for transient errors."""
        max_retries = 3
        base_delay_seconds = 0.5

        for attempt in range(max_retries + 1):
            try:
                return self.client.ocr.process(
                    model=self.config.model,
                    document=document,
                    include_image_base64=self.config.include_images,
                )
            except Exception as error:
                if attempt >= max_retries or not _is_retryable_ocr_error(error):
                    raise

                delay = base_delay_seconds * (2 ** attempt) + random.uniform(0, 0.25)
                if self.config.verbose:
                    console.print(
                        f"[yellow]Transient OCR error, retrying in {delay:.2f}s "
                        f"({attempt + 1}/{max_retries})[/yellow]"
                    )
                time.sleep(delay)

        raise RuntimeError("OCR retry loop exited unexpectedly")

    def _process_image_file(self, file_path: Path) -> object:
        """Process an image file through the direct OCR API."""
        if self.config.verbose:
            console.print(f"[dim]Creating data URI for {file_path.suffix} file...[/dim]")

        data_uri = create_data_uri(file_path)
        return self._process_with_retry({"type": "image_url", "image_url": data_uri})

    def _process_pdf_file(self, file_path: Path) -> object:
        """Process a PDF file via uploaded file chunks."""
        page_limit = self.config.max_pages_limit
        page_count = get_pdf_page_count(file_path)

        with tempfile.TemporaryDirectory(prefix="mistral_ocr_") as temp_dir:
            chunks = split_pdf_into_chunks(
                file_path,
                Path(temp_dir),
                max_pages_per_chunk=PDF_REQUEST_PAGE_LIMIT,
                max_chunk_size_mb=self.config.max_file_size_mb,
                max_pages=page_limit,
            )

            combined_pages: List[SimpleNamespace] = []
            for chunk_path, start_page, _page_count in chunks:
                uploaded_file_id = None
                try:
                    with open(chunk_path, "rb") as handle:
                        uploaded = self.client.files.upload(
                            file={"file_name": chunk_path.name, "content": handle},
                            purpose="ocr",
                        )
                    uploaded_file_id = uploaded.id

                    response = self._process_with_retry(
                        {"type": "file", "file_id": uploaded.id}
                    )
                finally:
                    if uploaded_file_id:
                        self._delete_uploaded_file(uploaded_file_id)

                for local_index, page in enumerate(getattr(response, "pages", [])):
                    combined_pages.append(
                        SimpleNamespace(
                            index=start_page + local_index,
                            markdown=getattr(page, "markdown", ""),
                            images=getattr(page, "images", []),
                        )
                    )

        truncated_message = self._get_truncated_message(file_path, page_count)
        return SimpleNamespace(pages=combined_pages, truncated_message=truncated_message)

    def _process_batch_files(
        self,
        file_paths: List[Path],
        output_dir: Path,
    ) -> Tuple[int, int]:
        """Process files via the Mistral Batch API."""
        if not hasattr(self.client, "batch"):
            raise AttributeError(
                "Batch endpoint not available in Mistral client. "
                "Please ensure you have a recent mistralai package."
            )

        total_count = len(file_paths)
        start_time = time.time()

        with tempfile.TemporaryDirectory(prefix="mistral_ocr_batch_") as temp_dir:
            temp_path = Path(temp_dir)
            request_entries, request_map, file_state, cleanup_file_ids = self._build_batch_requests(
                file_paths,
                temp_path,
            )

            if not request_entries:
                return 0, total_count

            batch_file = temp_path / "ocr_batch.jsonl"
            self._write_batch_file(batch_file, request_entries)

            uploaded_batch_id = None
            try:
                with open(batch_file, "rb") as handle:
                    uploaded_batch = self.client.files.upload(
                        file={"file_name": batch_file.name, "content": handle},
                        purpose="batch",
                    )
                uploaded_batch_id = uploaded_batch.id

                job = self.client.batch.jobs.create(
                    input_files=[uploaded_batch.id],
                    model=self.config.model,
                    endpoint="/v1/ocr",
                    metadata={"source": "mistral-ocr-cli", "requests": str(len(request_entries))},
                )
                job = self._wait_for_batch_job(job.id)

                output_records = self._get_batch_records(getattr(job, "output_file", None))
                error_records = self._get_batch_records(getattr(job, "error_file", None))
            except Exception as error:
                error_message = self._format_batch_error(error)
                console.print(f"[red]{error_message}[/red]")
                for state in file_state.values():
                    self.errors.append({
                        "file": str(state["file_path"]),
                        "error": error_message,
                    })
                return 0, total_count
            finally:
                if uploaded_batch_id:
                    self._delete_uploaded_file(uploaded_batch_id)
                for file_id in cleanup_file_ids:
                    self._delete_uploaded_file(file_id)

        self._apply_batch_output_records(
            output_records,
            error_records,
            request_map,
            file_state,
        )

        success_count = 0
        for state in file_state.values():
            pages: List[SimpleNamespace] = state["pages"]
            errors: List[str] = state["errors"]
            file_path: Path = state["file_path"]

            if not pages:
                errors.append("No OCR output was returned from the batch job.")

            if errors:
                error_message = "; ".join(errors)
                self.errors.append({
                    "file": str(file_path),
                    "error": error_message,
                })
                continue

            pages.sort(key=lambda page: page.index)
            response = SimpleNamespace(
                pages=pages,
                truncated_message=state["truncated_message"],
            )
            self.save_results({"file_path": file_path, "response": response}, output_dir)
            base_name = sanitize_filename(file_path.stem, max_length=None)
            self.processed_files.append({
                "file": str(file_path),
                "size": file_path.stat().st_size,
                "output": str(output_dir / f"{base_name}.md"),
                "batch_mode": True,
            })
            success_count += 1

        if self.config.verbose:
            elapsed = time.time() - start_time
            console.print(f"[dim]Batch processing time: {elapsed:.2f} seconds[/dim]")

        return success_count, total_count

    def _build_batch_requests(
        self,
        file_paths: List[Path],
        temp_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
        """Build batch requests and metadata for the selected files."""
        request_entries: List[Dict[str, Any]] = []
        request_map: Dict[str, Dict[str, Any]] = {}
        file_state: Dict[str, Dict[str, Any]] = {}
        cleanup_file_ids: List[str] = []
        request_index = 0

        for file_path in file_paths:
            file_key = str(file_path)
            file_state[file_key] = {
                "file_path": file_path,
                "pages": [],
                "errors": [],
                "truncated_message": None,
            }

            if file_path.suffix.lower() == ".pdf":
                page_count = get_pdf_page_count(file_path)
                file_state[file_key]["truncated_message"] = self._get_truncated_message(
                    file_path,
                    page_count,
                )
                chunk_dir = temp_dir / sanitize_filename(file_path.stem, max_length=None)
                chunks = split_pdf_into_chunks(
                    file_path,
                    chunk_dir,
                    max_pages_per_chunk=PDF_REQUEST_PAGE_LIMIT,
                    max_chunk_size_mb=self.config.max_file_size_mb,
                    max_pages=self.config.max_pages_limit,
                )
                for chunk_path, start_page, _page_count in chunks:
                    with open(chunk_path, "rb") as handle:
                        uploaded = self.client.files.upload(
                            file={"file_name": chunk_path.name, "content": handle},
                            purpose="ocr",
                        )
                    cleanup_file_ids.append(uploaded.id)
                    custom_id = f"req-{request_index}"
                    request_entries.append({
                        "custom_id": custom_id,
                        "body": {
                            "document": {"type": "file", "file_id": uploaded.id},
                            "include_image_base64": self.config.include_images,
                        },
                    })
                    request_map[custom_id] = {"file_path": file_path, "start_page": start_page}
                    request_index += 1
            else:
                self.config.validate_file_size(file_path)
                data_uri = create_data_uri(file_path)
                custom_id = f"req-{request_index}"
                request_entries.append({
                    "custom_id": custom_id,
                    "body": {
                        "document": {"type": "image_url", "image_url": data_uri},
                        "include_image_base64": self.config.include_images,
                    },
                })
                request_map[custom_id] = {"file_path": file_path, "start_page": 0}
                request_index += 1

        return request_entries, request_map, file_state, cleanup_file_ids

    def _write_batch_file(self, batch_file: Path, request_entries: List[Dict[str, Any]]) -> None:
        """Write batch requests to a JSONL file."""
        with open(batch_file, "w", encoding="utf-8") as handle:
            for entry in request_entries:
                handle.write(json.dumps(entry) + "\n")

    def _wait_for_batch_job(self, job_id: str) -> Any:
        """Poll the batch job until it reaches a terminal state."""
        while True:
            job = self.client.batch.jobs.get(job_id=job_id)
            if self.config.verbose:
                console.print(
                    "[dim]Batch status: "
                    f"{job.status} ({job.completed_requests}/{job.total_requests})[/dim]"
                )

            if job.status not in {"QUEUED", "RUNNING"}:
                return job

            time.sleep(BATCH_POLL_INTERVAL_SECONDS)

    def _get_batch_records(self, file_id: Optional[str]) -> List[Dict[str, Any]]:
        """Download and parse a batch output/error JSONL file."""
        if not file_id:
            return []

        response = self.client.files.download(file_id=file_id)
        content = response.read().decode("utf-8")
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _apply_batch_output_records(
        self,
        output_records: List[Dict[str, Any]],
        error_records: List[Dict[str, Any]],
        request_map: Dict[str, Dict[str, Any]],
        file_state: Dict[str, Dict[str, Any]],
    ) -> None:
        """Map batch output records back to per-file OCR results."""
        seen_request_ids = set()

        for record in output_records:
            custom_id = record.get("custom_id")
            if not isinstance(custom_id, str):
                continue
            request_info = request_map.get(custom_id)
            if not request_info:
                continue

            seen_request_ids.add(custom_id)
            state = file_state[str(request_info["file_path"])]
            response = record.get("response") or {}
            status_code = response.get("status_code", 200)
            body = response.get("body") or {}

            if status_code >= 400:
                state["errors"].append(f"Batch request failed with status {status_code}.")
                continue

            pages = body.get("pages") or []
            for page_index, page in enumerate(pages):
                local_index = page.get("index", page_index)
                state["pages"].append(
                    self._make_page_namespace(page, request_info["start_page"] + local_index)
                )

        for record in error_records:
            custom_id = record.get("custom_id")
            if not isinstance(custom_id, str):
                continue
            request_info = request_map.get(custom_id)
            if not request_info:
                continue

            seen_request_ids.add(custom_id)
            state = file_state[str(request_info["file_path"])]
            error = record.get("error") or {}
            if isinstance(error, dict):
                message = error.get("message") or error.get("detail") or json.dumps(error)
            else:
                message = str(error)
            state["errors"].append(f"Batch request error: {message}")

        for custom_id, request_info in request_map.items():
            if custom_id in seen_request_ids:
                continue
            state = file_state[str(request_info["file_path"])]
            state["errors"].append("No batch result was returned for one or more requests.")

    def _make_page_namespace(self, page: Dict[str, Any], index: int) -> SimpleNamespace:
        """Convert a JSON page payload into the structure expected by save_results."""
        images = []
        for image in page.get("images", []) or []:
            images.append(
                SimpleNamespace(
                    base64=image.get("image_base64") or image.get("base64"),
                )
            )

        return SimpleNamespace(
            index=index,
            markdown=page.get("markdown", ""),
            images=images,
        )

    def _get_truncated_message(self, file_path: Path, page_count: int) -> Optional[str]:
        """Return a truncation note when max-pages clipped the document."""
        page_limit = self.config.max_pages_limit
        if page_limit and page_count > page_limit:
            return f"Document truncated to first {page_limit} of {page_count} pages."
        return None

    def _format_batch_error(self, error: Exception) -> str:
        """Produce a friendlier batch error for unsupported accounts/keys."""
        error_text = str(error)
        error_text_lower = error_text.lower()
        if "status 402" in error_text_lower and "free trial" in error_text_lower:
            return (
                "Batch OCR is unavailable for this API key/account on the current free trial. "
                "Try rerunning without `--mode batch` or enable batch access in the Mistral console."
            )
        return f"Batch OCR failed: {error_text}"

    def _delete_uploaded_file(self, file_id: str) -> None:
        """Best-effort cleanup for uploaded OCR or batch input files."""
        try:
            self.client.files.delete(file_id=file_id)
        except Exception:
            if self.config.verbose:
                console.print(f"[dim]Failed to delete uploaded file: {file_id}[/dim]")

    def save_results(
        self,
        result: Dict[str, Any],
        output_dir: Path,
        is_single_file: bool = False,
    ) -> None:
        """Save OCR results to files."""
        file_path = result["file_path"]
        response = result["response"]

        base_name = sanitize_filename(file_path.stem, max_length=None)
        markdown_path = output_dir / f"{base_name}.md"

        markdown_content = []

        if self.config.include_metadata:
            markdown_content.append("# OCR Results\n\n")
            markdown_content.append(f"**Original File:** {file_path.name}\n")
            markdown_content.append(f"**Full Path:** `{file_path}`\n")
            markdown_content.append(f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            if getattr(response, "truncated_message", None):
                markdown_content.append(f"**Note:** {response.truncated_message}\n\n")
            markdown_content.append("---\n\n")

        if hasattr(response, "pages"):
            for page in response.pages:
                if self.config.include_page_headings:
                    markdown_content.append(f"## Page {page.index + 1}\n\n")

                if hasattr(page, "markdown"):
                    markdown_content.append(page.markdown)
                    markdown_content.append("\n\n")

                if self.config.include_images and hasattr(page, "images") and page.images:
                    images_dir = output_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)

                    for idx, image in enumerate(page.images):
                        image_base64 = get_image_base64_data(image)
                        if image_base64:
                            image_filename = f"page{page.index + 1}_img{idx + 1}.png"
                            image_path = images_dir / image_filename
                            save_base64_image(image_base64, image_path)
                            markdown_content.append(
                                f"![Image {idx + 1}](./images/{image_filename})\n\n"
                            )

        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write("".join(markdown_content))

        if self.config.verbose:
            console.print(f"[green]✓[/green] Saved results to {markdown_path}")

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        add_timestamp: bool = False,
        reprocess: bool = False,
    ) -> Tuple[int, int]:
        """Process all supported files in a directory."""
        files = get_supported_files(input_dir)

        if not files:
            console.print("[yellow]No supported files found in the directory.[/yellow]")
            return 0, 0

        output_path = determine_output_path(input_dir, output_dir, add_timestamp=add_timestamp)
        existing_metadata = load_metadata(output_path)
        existing_files_set = {item["file"] for item in existing_metadata["files_processed"]}

        files_to_process = []
        skipped_files = []
        for file_path in files:
            if str(file_path) in existing_files_set and not reprocess:
                skipped_files.append(file_path)
                if self.config.verbose:
                    console.print(f"[dim]Skipping already processed: {file_path.name}[/dim]")
            else:
                files_to_process.append(file_path)

        if skipped_files:
            console.print(f"[yellow]Skipping {len(skipped_files)} already processed file(s)[/yellow]")
            if not self.config.verbose:
                console.print("[dim]Use --verbose to see which files were skipped[/dim]")

        if not files_to_process:
            console.print("[green]All files already processed. Use --reprocess to force reprocessing.[/green]")
            return 0, 0

        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output directory: {output_path}[/blue]\n")

        start_time = time.time()
        if self.config.mode == "batch":
            success_count, total_count = self._process_batch_files(files_to_process, output_path)
        else:
            success_count = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Processing files...", total=len(files_to_process))

                for file_path in files_to_process:
                    file_size = format_file_size(file_path.stat().st_size)
                    progress.update(
                        task,
                        description=f"Processing {file_path.name} ({file_size})...",
                    )

                    result = self.process_file(file_path)
                    if result:
                        self.save_results(result, output_path, is_single_file=False)
                        success_count += 1
                        base_name = sanitize_filename(file_path.stem, max_length=None)
                        self.processed_files.append({
                            "file": str(file_path),
                            "size": file_path.stat().st_size,
                            "output": str(output_path / f"{base_name}.md"),
                        })

                    progress.update(task, advance=1)
            total_count = len(files_to_process)

        processing_time = time.time() - start_time
        save_metadata(output_path, self.processed_files, processing_time, self.errors)
        return success_count, total_count

    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        add_timestamp: bool = False,
        reprocess: bool = False,
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            output_dir = determine_output_path(input_path, output_path, add_timestamp=add_timestamp)
            existing_metadata = load_metadata(output_dir)
            existing_files_set = {item["file"] for item in existing_metadata["files_processed"]}

            if str(input_path) in existing_files_set and not reprocess:
                base_name = sanitize_filename(input_path.stem, max_length=None)
                output_file = output_dir / f"{base_name}.md"
                console.print(f"[yellow]File already processed: {input_path.name}[/yellow]")
                console.print(f"[dim]Output exists at: {output_file}[/dim]")
                console.print("[dim]Use --reprocess to force reprocessing.[/dim]")
                return

            console.print(f"[blue]Processing file: {input_path}[/blue]")
            console.print(f"[blue]Output directory: {output_dir}[/blue]\n")

            start_time = time.time()
            if self.config.mode == "batch":
                success_count, _total_count = self._process_batch_files([input_path], output_dir)
                processing_time = time.time() - start_time
                save_metadata(output_dir, self.processed_files, processing_time, self.errors)
                if success_count == 1:
                    console.print("\n[green]✓ Successfully processed 1 file[/green]")
                    console.print(f"[dim]Processing time: {processing_time:.2f} seconds[/dim]")
                else:
                    console.print("\n[red]✗ Failed to process file[/red]")
                return

            result = self.process_file(input_path)
            if result:
                self.save_results(result, output_dir, is_single_file=True)
                base_name = sanitize_filename(input_path.stem, max_length=None)
                self.processed_files.append({
                    "file": str(input_path),
                    "size": input_path.stat().st_size,
                    "output": str(output_dir / f"{base_name}.md"),
                })

                processing_time = time.time() - start_time
                save_metadata(output_dir, self.processed_files, processing_time, self.errors)

                console.print("\n[green]✓ Successfully processed 1 file[/green]")
                console.print(f"[dim]Processing time: {processing_time:.2f} seconds[/dim]")
            else:
                console.print("\n[red]✗ Failed to process file[/red]")

        elif input_path.is_dir():
            success_count, total_count = self.process_directory(
                input_path,
                output_path,
                add_timestamp,
                reprocess,
            )

            console.print(f"\n[green]✓ Successfully processed {success_count}/{total_count} files[/green]")
            if self.errors:
                console.print(f"[red]✗ {len(self.errors)} file(s) failed[/red]")

        else:
            raise ValueError(f"Input path does not exist: {input_path}")
