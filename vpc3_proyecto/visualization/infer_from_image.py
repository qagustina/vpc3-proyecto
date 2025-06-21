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
    # Get model's device and data type
    # Move to model's device and dtype
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Move and convert input to match model's device and dtype
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)


    # Move decoder inputs if needed
    if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids.to(device)

    # Run inference
    if hasattr(model, "generate"):
        if decoder_input_ids is not None:
            # Generate with prompt conditioning
            # Generate output
            outputs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=128,
            )
        else:
            outputs = model.generate(pixel_values=pixel_values)
    else:
        raise ValueError("Model does not have a .generate() method!")

    # Fixed version - use batch_decode instead
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Display
    if show_image:
        plt.imshow(original_image)
        plt.axis('off')
        plt.title("texto detectado: " + result, fontsize=12)
        plt.show()

    return result
