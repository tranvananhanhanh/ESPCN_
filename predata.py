from PIL import Image
import os

def resize_and_convert(input_folder, output_folder, target_size=(128, 128), target_mode="RGB"):
    # Tạo thư mục đầu ra nếu nó không tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lặp qua tất cả các ảnh trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Đường dẫn đầy đủ đến ảnh đầu vào
            input_path = os.path.join(input_folder, filename)

            # Đọc ảnh
            image = Image.open(input_path)

            # Resize ảnh
            resized_image = image.resize(target_size)

            # Đổi kênh màu (chế độ màu)
            if target_mode and image.mode != target_mode:
                resized_image = resized_image.convert(target_mode)

            # Đường dẫn đầy đủ đến ảnh đầu ra
            output_path = os.path.join(output_folder, filename)

            # Lưu ảnh đã resize và đổi kênh màu
            resized_image.save(output_path,format="JPEG")

# Đường dẫn đến thư mục chứa ảnh cần resize và đổi kênh màu
input_folder = "/Users/jmac/Desktop/ESPCN/data"

# Đường dẫn đến thư mục lưu ảnh sau khi resize và đổi kênh màu
output_folder = "/Users/jmac/Desktop/ESPCN/resizee"

# Kích thước mục tiêu
target_size = (128, 128)

# Chế độ màu (VD: "RGB", "YCbCr", "L", ...)
target_mode = "YCbCr"

# Resize và đổi kênh màu ảnh trong thư mục đầu vào và lưu vào thư mục đầu ra
resize_and_convert(input_folder, output_folder, target_size, target_mode)
