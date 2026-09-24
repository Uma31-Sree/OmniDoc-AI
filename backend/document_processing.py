from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.utils.utils import create_hash
import base64


class DocumentProcessor:
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process(self, pdf_path: str, min_chunk_chars: int = 120):
        """Extracts text chunks and images from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            min_chunk_chars: Minimum characters per chunk. Short lines are merged
                with the next line until this threshold is met.
        """
        conv_res = self.converter.convert(pdf_path)
        doc = conv_res.document

        raw_chunks = []

        # ---- Extract raw text lines from all elements ----
        try:
            for item, _ in doc.iterate_items():
                if hasattr(item, "text") and item.text and item.text.strip():
                    raw_chunks.append(item.text.strip())
        except Exception as e:
            print(f"[WARN] Text extraction issue: {e}")

        # ---- Merge short chunks into meaningful paragraphs ----
        merged_chunks = []
        buffer = ""
        for chunk in raw_chunks:
            if not buffer:
                buffer = chunk
            else:
                buffer = buffer + " " + chunk
            if len(buffer) >= min_chunk_chars:
                merged_chunks.append(buffer.strip())
                buffer = ""

        if buffer.strip():
            merged_chunks.append(buffer.strip())

        # ---- Extract images from pages ----
        images = []
        pages = doc.pages
        if isinstance(pages, dict):
            page_items = pages.items()
        elif isinstance(pages, list):
            page_items = enumerate(pages, start=1)
        else:
            page_items = []

        for page_no, page in page_items:
            try:
                page_img = getattr(page, "image", None)
                if page_img is None:
                    page_img = getattr(page, "_image", None)
                if page_img is None:
                    imgs = getattr(page, "images", None)
                    if isinstance(imgs, list) and imgs:
                        page_img = imgs[0]

                if page_img is not None:
                    img_bytes = page_img.tobytes() if hasattr(page_img, "tobytes") else bytes(page_img)
                    images.append({
                        "page": page_no,
                        "bytes": img_bytes,
                        "width": getattr(page_img, "width", None),
                        "height": getattr(page_img, "height", None),
                        "hash": create_hash(conv_res.input.document_hash + ":" + str(page_no - 1)),
                        "base64": base64.b64encode(img_bytes).decode("utf-8")
                    })
            except Exception as e:
                print(f"[WARN] Could not extract image from page {page_no}: {e}")

        print(f"   → Merged {len(raw_chunks)} raw lines into {len(merged_chunks)} coherent chunks")
        return merged_chunks, images