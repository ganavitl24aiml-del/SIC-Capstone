
import os
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page in document:

        text = page.get_text()

        if text.strip():
            pages.append(text)

    document.close()

    return "\n".join(pages)


def extract_all_pdfs(data_dir):

    documents = []

    for root, _, files in os.walk(data_dir):

        for filename in files:

            if filename.lower().endswith(".pdf"):

                path = os.path.join(
                    root,
                    filename
                )

                text = extract_text_from_pdf(path)

                documents.append({
                    "source": filename,
                    "path": path,
                    "text": text
                })

    return documents
