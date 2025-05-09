import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except FileNotFoundError:
        return "Error: PDF file not found."
    return text

if __name__ == '__main__':
    pdf_file = '../../data/college_rules.pdf'
    extracted_text = extract_text_from_pdf(pdf_file)
    if isinstance(extracted_text, str) and not extracted_text.startswith("Error"):
        print("Successfully extracted text.")
        # You can print a snippet here to verify:
        # print(extracted_text[:500])
    else:
        print(extracted_text)
