import torch
import matplotlib.pyplot as plt

def infer_text_from_image(original_image, model, processor, task_prompt=None, show_image=True):
    """
    Infers text from an image using the given model and processor.

    Args:
        original_image: PIL.Image or similar image loaded
        model: pretrained model (TrOCR, Donut, VisionEncoderDecoder)
        processor: corresponding processor
        task_prompt (str or None): prompt to guide decoding (if needed, e.g. for Donut)
        show_image (bool): whether to display the image and result

    Returns:
        str: decoded text result
    """
    # Prepare decoder inputs if needed
    if task_prompt is not None:
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    else:
        decoder_input_ids = None

    # Process image
    pixel_values = processor(images=original_image, return_tensors="pt").pixel_values

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pixel_values = pixel_values.to(device)

    # Move decoder inputs if needed
    if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids.to(device)

    # Run inference
    if hasattr(model, "generate"):
        if decoder_input_ids is not None:
            outputs = model.generate(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        else:
            outputs = model.generate(pixel_values=pixel_values)
    else:
        raise ValueError("Model does not have a .generate() method!")

    result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display
    if show_image:
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(result, fontsize=12)
        plt.show()

    return result
