from app import tools


def test_parse_resume_text_pdf(monkeypatch):
    called = {}

    def fake_parse_pdf(file_bytes: bytes) -> str:
        called["pdf"] = True
        return "PDF TEXT"

    monkeypatch.setattr(tools, "parse_pdf", fake_parse_pdf)
    result = tools.parse_resume_text(b"dummy", "cv.PDF")
    assert result == "PDF TEXT"
    assert called.get("pdf") is True


def test_parse_resume_text_docx(monkeypatch):
    called = {}

    def fake_parse_docx(file_bytes: bytes) -> str:
        called["docx"] = True
        return "DOCX TEXT"

    monkeypatch.setattr(tools, "parse_docx", fake_parse_docx)
    result = tools.parse_resume_text(b"dummy", "cv.docx")
    assert result == "DOCX TEXT"
    assert called.get("docx") is True


def test_parse_resume_text_plain():
    txt = b"hello world"
    result = tools.parse_resume_text(txt, "notes.txt")
    assert result == "hello world"
