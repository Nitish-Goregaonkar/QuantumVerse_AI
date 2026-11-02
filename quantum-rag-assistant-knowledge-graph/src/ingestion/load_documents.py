"""Robust local document extraction with page/slide metadata and optional OCR.

OCR requires Tesseract installed on the host and ``pytesseract``/Pillow installed.
Visual understanding is not inferred from OCR; OCR extracts readable text only.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from src.utils.logger import get_logger
log = get_logger(__name__)

@dataclass
class Document:
    content: str
    source: str
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    def __post_init__(self):
        # Keep OCR line breaks for headings, bullets, tables and timeline layouts.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in self.content.splitlines()]
        self.content = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    def load_directory(self, directory: str | Path) -> List[Document]:
        directory=Path(directory)
        if not directory.exists(): raise FileNotFoundError(f"Directory not found: {directory}")
        documents=[]
        for fp in sorted(directory.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try: documents.extend(self.load_file(fp))
                except Exception as exc: log.exception("Failed to load %s: %s", fp.name, exc)
        return documents
    def load_file(self, file_path: str | Path) -> List[Document]:
        fp=Path(file_path); ext=fp.suffix.lower()
        dispatch={".pdf":self._load_pdf,".txt":self._load_text,".md":self._load_text,
                 ".docx":self._load_docx,".pptx":self._load_pptx}
        if ext in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}: return self._load_image(fp)
        if ext not in dispatch: raise ValueError(f"Unsupported file type: {ext}")
        return dispatch[ext](fp)
    @staticmethod
    def _ocr_image(image):
        """Return OCR text, or raise a clear optional-dependency error."""
        try:
            import pytesseract
            return pytesseract.image_to_string(image).strip()
        except ImportError as exc:
            raise RuntimeError("OCR needs pytesseract (pip install pytesseract) and the Tesseract OCR executable installed and on PATH.") from exc
        except Exception as exc:
            # pytesseract raises TesseractNotFoundError when the Python wrapper
            # exists but the native executable is missing. Normalize that case.
            if "tesseract" in str(exc).lower() or exc.__class__.__name__ == "TesseractNotFoundError":
                raise RuntimeError("Tesseract OCR executable is missing. Install Tesseract and add its folder to PATH, or configure pytesseract.pytesseract.tesseract_cmd.") from exc
            raise
    def _load_pdf(self, fp):
        try: import fitz
        except ImportError as exc: raise ImportError("Install PyMuPDF: pip install pymupdf") from exc
        docs=[]
        with fitz.open(str(fp)) as pdf:
            if pdf.is_encrypted and not pdf.authenticate(""): raise ValueError("PDF is password-protected.")
            for i,page in enumerate(pdf,1):
                text=page.get_text("text").strip()
                if len(text)<20:
                    try:
                        pix=page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
                        from PIL import Image
                        import io
                        text=self._ocr_image(Image.open(io.BytesIO(pix.tobytes("png"))))
                    except (ImportError, RuntimeError) as exc:
                        log.warning("OCR unavailable for %s page %d: %s",fp.name,i,exc)
                        # Do not abort ingestion of other readable pages/files.
                if text: docs.append(Document(text,fp.name,i,{"file_path":str(fp),"total_pages":len(pdf),"format":"pdf"}))
        return docs
    def _load_text(self,fp):
        text=fp.read_text(encoding="utf-8",errors="replace")
        return [Document(text,fp.name,None,{"file_path":str(fp),"format":fp.suffix[1:]})] if text.strip() else []
    @staticmethod
    def _picture_to_pil(image_blob: bytes, image_ext: str = ""):
        """Decode a PowerPoint picture into a real RGB Pillow image.

        python-pptx can expose SVG/EMF/WMF assets as well as ordinary raster
        images. Pillow handles common raster formats; SVG is rendered through
        the optional CairoSVG package. Unsupported vector formats are reported
        clearly instead of passing an invalid object to Tesseract.
        """
        import io
        from PIL import Image, ImageOps

        ext = (image_ext or "").lower().lstrip(".")
        blob = bytes(image_blob)
        if ext == "svg" or blob.lstrip().startswith(b"<svg") or b"<svg" in blob[:500]:
            try:
                import cairosvg
            except ImportError as exc:
                raise RuntimeError(
                    "This slide contains an SVG image. Install CairoSVG with "
                    "`pip install cairosvg` to OCR embedded SVG graphics."
                ) from exc
            blob = cairosvg.svg2png(bytestring=blob, output_width=1800)

        try:
            with Image.open(io.BytesIO(blob)) as opened:
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Embedded picture format '{ext or 'unknown'}' cannot be decoded "
                "for OCR. Rasterize SVG/EMF/WMF images or export them as PNG/JPEG."
            ) from exc

    @staticmethod
    def _iter_picture_shapes(shapes):
        """Yield picture shapes, including pictures nested inside group shapes."""
        try:
            from pptx.enum.shapes import MSO_SHAPE_TYPE
            picture_type = MSO_SHAPE_TYPE.PICTURE
            group_type = MSO_SHAPE_TYPE.GROUP
        except ImportError:
            picture_type, group_type = 13, 6

        for shape in shapes:
            if getattr(shape, "shape_type", None) == picture_type:
                yield shape
            elif getattr(shape, "shape_type", None) == group_type:
                yield from DocumentLoader._iter_picture_shapes(shape.shapes)

    def _load_docx(self,fp):
        try: from docx import Document as WordDocument
        except ImportError as exc: raise ImportError("Install python-docx") from exc
        import io
        doc=WordDocument(str(fp)); parts=[]
        for para in doc.paragraphs:
            if para.text.strip(): parts.append(para.text)
        for table in doc.tables:
            parts.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
        # Embedded image OCR is best-effort and explicitly marked.
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image=self._picture_to_pil(rel.target_part.blob, Path(rel.target_part.partname).suffix)
                    text=self._ocr_image(image)
                    if text: parts.append("[Embedded image OCR] "+text)
                except Exception as exc: log.warning("DOCX image OCR skipped: %s",exc)
        content="\\n".join(parts)
        return [Document(content,fp.name,None,{"file_path":str(fp),"format":"docx"})] if content.strip() else []

    def _load_pptx(self,fp):
        try: from pptx import Presentation
        except ImportError as exc: raise ImportError("Install python-pptx: pip install python-pptx") from exc
        prs=Presentation(str(fp)); docs=[]
        for i,slide in enumerate(prs.slides,1):
            parts=[]
            # Preserve the reading order of text, tables and embedded pictures.
            for shape in slide.shapes:
                if getattr(shape,"has_text_frame",False):
                    txt="\\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip())
                    if txt: parts.append(txt)
                if getattr(shape,"has_table",False):
                    parts.extend(" | ".join(c.text.strip() for c in row.cells) for row in shape.table.rows)

            # Pictures may be nested in PowerPoint group shapes. Decode every
            # picture to RGB before OCR; do not rely on a numeric shape-type
            # magic value or assume the embedded asset is PNG/JPEG.
            for picture in self._iter_picture_shapes(slide.shapes):
                try:
                    image = self._picture_to_pil(picture.image.blob, picture.image.ext)
                    txt = self._ocr_image(image)
                    if txt:
                        parts.append("[Embedded image OCR] "+txt)
                except Exception as exc:
                    image_name = getattr(getattr(picture, "image", None), "ext", "unknown")
                    log.warning(
                        "Slide image OCR skipped: %s (slide=%d, image_format=%s)",
                        exc, i, image_name
                    )

            try:
                notes=slide.notes_slide.notes_text_frame.text
                if notes.strip(): parts.append("[Speaker notes] "+notes)
            except Exception: pass
            if parts:
                docs.append(Document("\\n".join(parts),fp.name,i,{
                    "file_path":str(fp),"format":"pptx","total_slides":len(prs.slides),
                    "contains_embedded_image_ocr":"[Embedded image OCR]" in "\\n".join(parts)
                }))
        return docs

    def _load_image(self,fp):
        try:
            from PIL import Image, ImageOps, ImageEnhance
        except ImportError as exc:
            raise ImportError("Install Pillow and pytesseract for image OCR") from exc
        with Image.open(fp) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            # Improve small infographic labels and low-contrast screenshots for OCR.
            if image.width < 1800:
                scale = min(3.0, 1800 / max(1, image.width))
                image = image.resize((int(image.width * scale), int(image.height * scale)))
            gray = ImageOps.autocontrast(ImageOps.grayscale(image))
            gray = ImageEnhance.Contrast(gray).enhance(1.25)
            # Colored infographic text is often recognized more reliably after
            # grayscale/threshold preprocessing than from the original RGB pixels.
            thresholded = gray.point(lambda pixel: 255 if pixel > 145 else 0)
            attempts = []
            for candidate_image, psm in ((gray, 11), (thresholded, 6), (gray, 6)):
                try:
                    import pytesseract
                    candidate = pytesseract.image_to_string(candidate_image, config=f"--oem 3 --psm {psm}").strip()
                    if candidate:
                        attempts.append(candidate)
                except ImportError as exc:
                    raise RuntimeError("OCR needs pytesseract and the Tesseract OCR executable installed.") from exc
            # Prefer OCR passes that capture dates and numeric labels as well as text.
            text = max(
                attempts,
                key=lambda value: len(value) + 60 * len(re.findall(r"\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2})?\b", value)),
                default="",
            )
        metadata={"file_path":str(fp),"format":fp.suffix[1:],"extraction":"ocr","ocr_layout_preserved":True}
        return [Document(text,fp.name,None,metadata)] if text else []
