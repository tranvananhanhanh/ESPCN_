from PIL import Image
import os

def resize_and_convert(input_folder, output_folder, target_size=(128, 128), target_mode="RGB"):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            input_path = os.path.join(input_folder, filename)

            image = Image.open(input_path)

        
            resized_image = image.resize(target_size)

            
            if target_mode and image.mode != target_mode:
                resized_image = resized_image.convert(target_mode)

            
            output_path = os.path.join(output_folder, filename)

            
            resized_image.save(output_path,format="JPEG")


input_folder = "/Users/jmac/Desktop/ESPCN/data"


output_folder = "/Users/jmac/Desktop/ESPCN/resizee"


target_size = (128, 128)

target_mode = "YCbCr"

resize_and_convert(input_folder, output_folder, target_size, target_mode)
