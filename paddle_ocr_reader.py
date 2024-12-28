from paddleocr import PaddleOCR
import re  # Import for filtering alphabetic characters

def extract_filtered_alpha_words(image_path):
    """
    Extract and filter only alphabetic words from an image using PaddleOCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: A list of filtered alphabetic words.
    """
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English OCR model

    # Perform OCR
    results = ocr.ocr(image_path, cls=True)

    # Function to filter only alphabetic words (A-Z, a-z)
    def is_alpha_word(text):
        return bool(re.match(r'^[A-Za-z]+$', text))

    # Extract and filter words
    filtered_words = [line[1][0] for line in results[0] if is_alpha_word(line[1][0])]
    filtered_text = re.sub(r'[\[\]\'\"]', '', str(filtered_words))

    return filtered_text

# # Example usage
# if __name__ == "__main__":
#     image_path = r"DataSet\0000014.jpg"  # Path to the image
#     filtered_words = extract_filtered_alpha_words(image_path)

#     print("Filtered Alphabetic Words:")
#     for word in filtered_words:
#         print(word)

    # Optionally, save the filtered words to a file
    # output_file = "filtered_words.txt"
    # with open(output_file, "w") as file:
    #     file.writelines(word + "\n" for word in filtered_words)
    # print(f"Filtered words saved to {output_file}")
