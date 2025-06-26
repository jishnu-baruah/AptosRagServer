import json
from pypdf import PdfReader
import re

PDF_FILE = 'aptos-whitepaper_en.pdf'
OUTPUT_FILE = 'whitepaper_chunks.jsonl'
MIN_WORDS = 200
MAX_WORDS = 400

# Regex to match section headers like '1 Introduction', '2 The Aptos vision', etc.
SECTION_HEADER_RE = re.compile(r'^(\d+(?:\.\d+)*)([\s\-:]+)([A-Z][^\n]+)', re.MULTILINE)

# Helper to split text into paragraphs
PARA_SPLIT = re.compile(r'\n\s*\n')

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return '\n'.join(text)

def split_by_sections(text):
    # Find all section headers and their positions
    matches = list(SECTION_HEADER_RE.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_title = match.group(0).strip()
        section_text = text[start:end].strip()
        sections.append((section_title, section_text))
    return sections

def chunk_section(section_title, section_text):
    # Split section text into paragraphs
    paragraphs = [p.strip() for p in PARA_SPLIT.split(section_text) if p.strip()]
    chunks = []
    current = []
    current_word_count = 0
    for para in paragraphs:
        words = para.split()
        if current_word_count + len(words) > MAX_WORDS and current:
            # Write current chunk
            chunk_text = '\n\n'.join(current)
            chunks.append({'text': chunk_text, 'section': section_title})
            current = []
            current_word_count = 0
        current.append(para)
        current_word_count += len(words)
        # If the paragraph itself is very large, split it
        while current_word_count > MAX_WORDS:
            para_words = ' '.join(current).split()
            chunk_words = para_words[:MAX_WORDS]
            rest_words = para_words[MAX_WORDS:]
            chunks.append({'text': ' '.join(chunk_words), 'section': section_title})
            current = [' '.join(rest_words)] if rest_words else []
            current_word_count = len(rest_words)
    # Write any remaining chunk (if it's not too small, or if it's the only chunk)
    if current and (current_word_count >= MIN_WORDS or len(chunks) == 0):
        chunk_text = '\n\n'.join(current)
        chunks.append({'text': chunk_text, 'section': section_title})
    elif current and chunks:
        # Merge with previous chunk if too small
        prev = chunks.pop()
        merged_text = prev['text'] + '\n\n' + '\n\n'.join(current)
        chunks.append({'text': merged_text, 'section': section_title})
    return chunks

def main():
    print(f"Extracting text from {PDF_FILE} ...")
    text = extract_pdf_text(PDF_FILE)
    print(f"Splitting by sections ...")
    sections = split_by_sections(text)
    all_chunks = []
    for section_title, section_text in sections:
        section_chunks = chunk_section(section_title, section_text)
        all_chunks.extend(section_chunks)
    print(f"Writing {len(all_chunks)} chunks to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Done.")

if __name__ == '__main__':
    main() 