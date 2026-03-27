import os
import base64
import json
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load env in case it hasn't been loaded
load_dotenv()

def draw_ruler_on_image(img: Image.Image) -> Image.Image:
    """Vẽ thước đo 0-100 dưới đáy ảnh để GPT Vision lấy tọa độ."""
    width, height = img.size
    
    # Tạo không gian thêm ở dưới cùng để vẽ thước
    ruler_height = 40
    new_img = Image.new('RGB', (width, height + ruler_height), (255, 255, 255))
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Vẽ thước từ 0 đến 100
    segments = 100
    draw.rectangle([0, height, width, height + ruler_height], fill="black")
    
    for i in range(segments + 1):
        x = int(i * (width / segments))
        # Vạch chính mỗi 10, vạch phụ mỗi 5, vạch nhỏ mỗi 1
        if i % 10 == 0:
            draw.line([x, height, x, height + ruler_height], fill="white", width=2)
        elif i % 5 == 0:
            draw.line([x, height, x, height + ruler_height - 15], fill="white", width=2)
        else:
            draw.line([x, height, x, height + ruler_height - 25], fill="white", width=1)
            
        # Vẽ số dọc cho mỗi vạch chẵn để không đè nhau
        if i % 2 == 0:
            txt_img = Image.new('RGBA', (30, 15), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0, 0), str(i), fill="white", font=font)
            txt_img = txt_img.rotate(90, expand=True)
            new_img.paste(txt_img, (x - 7, height + 5), txt_img)
            
    return new_img

def encode_pil_image(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def ask_gpt_for_split_mark(img_with_ruler: Image.Image) -> int:
    """Gắn ảnh có thước lên GPT và lấy về số hiệu vạch chia cắt."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠️ Không tìm thấy OPENAI_API_KEY, bỏ qua chia ảnh bằng GPT.")
        return -1
        
    base64_image = encode_pil_image(img_with_ruler)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Bắt buộc dùng gpt-4o-mini (rẻ, nhanh và đủ đáp ứng sau khi test)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Ảnh này là một khung chứa quảng cáo sản phẩm. Thường nó chứa 1 hoặc 2 sản phẩm riêng biệt nằm cạnh nhau. Chú ý: 1 sản phẩm thường bao gồm ảnh, giá tiền, chữ và có thể có banner đỏ đi kèm. Tôi đã vẽ một cây thước màu đen với các vạch từ 0 đến 100 ở dưới cùng bức ảnh.\n\nCâu hỏi:\n1. Ảnh này có chứa mấy sản phẩm khác nhau? (ví dụ: 1 hoặc 2 sản phẩm)\n2. Nếu có 2 sản phẩm, hãy xác định đường ranh giới dọc chính xác nhất chia tách 2 sản phẩm đó. Đường này quét qua vạch số mấy trên cây thước ở dưới cùng? (Hãy nhìn thật kỹ và cho số chính xác đến hàng đơn vị, ví dụ: 58, 62, 53, 47. KHÔNG ĐƯỢC làm tròn thành bội số của 5 hay 10 trừ khi nó thực sự nằm đúng vạch đó). Nếu chỉ có 1 sản phẩm thì trả về -1.\n\nHãy trả về ĐÚNG định dạng JSON sau, không kèm bất kỳ đoạn text markdown nào khác:\n{\"product_count\": 2, \"split_mark\": 58}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            res_text = response.json()["choices"][0]["message"]["content"]
            if "```" in res_text:
                res_text = res_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(res_text)
            if result.get("product_count", 0) > 1:
                return int(result.get("split_mark", -1))
    except Exception as e:
        print(f"  ⚠️ GPT Splitter Error: {e}")
        
    return -1

def split_bbox_if_needed(bbox_image: Image.Image) -> list[Image.Image]:
    """
    Nhận vào 1 ảnh bbox PIL. 
    Nếu ảnh quá rộng (khả nghi dính 2 id), dùng GPT phân tích và cắt làm 2.
    Trả về danh sách 1 ảnh (nếu ko cắt) hoặc 2 ảnh (nếu cắt thành công).
    """
    width, height = bbox_image.size
    aspect_ratio = width / height
    
    # 1. Filter: Chỉ gửi GPT nếu ảnh ngang và rộng bất thường (>= 1.5)
    if aspect_ratio < 1.5:
        return [bbox_image]
        
    print(f"  🔍 Bbox tỷ lệ {aspect_ratio:.2f} (rộng bất thường). Đang gọi GPT để kiểm tra...")
    
    # 2. Draw ruler
    img_with_ruler = draw_ruler_on_image(bbox_image)
    
    # 3. Ask GPT
    split_mark = ask_gpt_for_split_mark(img_with_ruler)
    
    # 4. Crop if valid
    if split_mark > 0 and split_mark < 100:
        split_pixel_x = int(width * (split_mark / 100.0))
        print(f"  ✂️ GPT yêu cầu cắt ảnh tại vạch {split_mark} (Pixel X: {split_pixel_x}).")
        
        part1 = bbox_image.crop((0, 0, split_pixel_x, height))
        part2 = bbox_image.crop((split_pixel_x, 0, width, height))
        return [part1, part2]
    else:
        print("  ✅ GPT báo ảnh này chỉ có 1 sản phẩm. Không cắt.")
        return [bbox_image]
