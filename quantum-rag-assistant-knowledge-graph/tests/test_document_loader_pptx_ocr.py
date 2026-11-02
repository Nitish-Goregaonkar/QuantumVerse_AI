"""Regression tests for PowerPoint embedded-image OCR."""
from io import BytesIO

import pytest


def _png_bytes():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (120, 60), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_pptx_embedded_picture_is_converted_to_rgb_and_ocrd(tmp_path, monkeypatch):
    pytest.importorskip("pptx")
    from pptx import Presentation
    from src.ingestion.load_documents import DocumentLoader

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.shapes.add_picture(BytesIO(_png_bytes()), 0, 0)
    deck_path = tmp_path / "embedded-image.pptx"
    deck.save(deck_path)

    seen = []
    monkeypatch.setattr(
        DocumentLoader, "_ocr_image",
        staticmethod(lambda image: seen.append(image.mode) or "Recognized slide infographic text"),
    )

    documents = DocumentLoader().load_file(deck_path)

    assert len(documents) == 1
    assert "Recognized slide infographic text" in documents[0].content
    assert seen == ["RGB"]
    assert documents[0].metadata["format"] == "pptx"
    assert documents[0].metadata["contains_embedded_image_ocr"] is True


def test_pptx_text_uses_real_line_breaks(tmp_path):
    pytest.importorskip("pptx")
    from pptx import Presentation
    from src.ingestion.load_documents import DocumentLoader

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    box = slide.shapes.add_textbox(0, 0, 5000000, 1000000)
    box.text_frame.text = "First heading"
    box.text_frame.add_paragraph().text = "Second heading"
    deck_path = tmp_path / "text-slide.pptx"
    deck.save(deck_path)

    documents = DocumentLoader().load_file(deck_path)
    assert len(documents) == 1
    assert "First heading\\nSecond heading" in documents[0].content
    assert "\\\\n" not in documents[0].content
