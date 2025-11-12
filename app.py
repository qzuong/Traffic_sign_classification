import tkinter as tk
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Frame, Label
from tkinter import filedialog
from PIL import Image, ImageTk

# Kích thước ảnh mà mô hình của bạn đã được huấn luyện
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Load mô hình đã huấn luyện 
try:
    model = tf.keras.models.load_model('model.h5')
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"LỖI: Không thể tải model.h5. Lỗi: {e}")
    exit()

# Dictionary map ClassId sang tên biển báo
classes = { 0:'Tốc độ tối đa (20km/h)',
            1:'Tốc độ tối đa (30km/h)', 
            2:'Tốc độ tối đa (50km/h)', 
            3:'Tốc độ tối đa (60km/h)', 
            4:'Tốc độ tối đa (70km/h)', 
            5:'Tốc độ tối đa (80km/h)', 
            6:'Hết hạn chế tốc độ (80km/h)', 
            7:'Tốc độ tối đa (100km/h)', 
            8:'Tốc độ tối đa (120km/h)', 
            9:'Cấm vượt', 
            10:'Cấm xe tải vượt', 
            11:'Giao nhau với đường ưu tiên', 
            12:'Đường ưu tiên', 
            13:'Nhường đường', 
            14:'Dừng lại (Stop)', 
            15:'Đường cấm', 
            16:'Cấm xe tải (trên 3.5 tấn)', 
            17:'Cấm đi ngược chiều', 
            18:'Nguy hiểm khác', 
            19:'Chỗ ngoặt nguy hiểm (trái)', 
            20:'Chỗ ngoặt nguy hiểm (phải)', 
            21:'Đường cong liên tục', 
            22:'Đường gồ ghề', 
            23:'Đường trơn trượt', 
            24:'Đường hẹp (phải)', 
            25:'Công trường', 
            26:'Đèn tín hiệu giao thông', 
            27:'Đường người đi bộ', 
            28:'Trẻ em qua đường', 
            29:'Đường xe đạp', 
            30:'Cẩn thận băng tuyết',
            31:'Thú rừng qua đường', 
            32:'Hết tất cả lệnh cấm', 
            33:'Rẽ phải', 
            34:'Rẽ trái', 
            35:'Đi thẳng', 
            36:'Đi thẳng hoặc rẽ phải', 
            37:'Đi thẳng hoặc rẽ trái', 
            38:'Giữ bên phải', 
            39:'Giữ bên trái', 
            40:'Vòng xuyến', 
            41:'Hết cấm vượt', 
            42:'Hết cấm xe tải vượt' }

g_ori_image = None

# Hàm xóa ảnh cũ trong Frame
def clearFrame(frame):
    for widgets in frame.winfo_children():
        widgets.destroy()

def BrowseImage(frame):
    global g_ori_image

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.tif")]
    )
    if not file_path:
        return

    # Ảnh hiển thị
    img_pil = Image.open(file_path).resize((400, 400))
    photo = ImageTk.PhotoImage(img_pil)

    # Ảnh gốc để xử lý
    g_ori_image = cv2.imread(file_path)

    clearFrame(frame)

    labelImg = Label(frame, image=photo)
    labelImg.image = photo
    labelImg.pack()

# Hàm dự đoán biển báo 
def predict_sign():
    global g_ori_image
    
    if g_ori_image is None:
        lblResult.config(text="Lỗi: Vui lòng chọn ảnh trước!")
        return

    try:
        image = g_ori_image
        image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        expand_input = np.expand_dims(image_rgb, axis=0)
        input_data = expand_input / 255.0

        pred_probabilities = model.predict(input_data)

        result_id = np.argmax(pred_probabilities, axis=1)[0] 
        result_name = classes.get(result_id, "Không nhận dạng được")
        
        lblResult.config(text=f"ID: {result_id}\n{result_name}")

    except Exception as e:
        lblResult.config(text=f"Lỗi khi dự đoán: {e}")

# Tạo cửa sổ chính
window = tk.Tk()
window.geometry("1000x750")
window.configure(bg="#D4D4CE")

# Cấu hình lưới để căn giữa
window.grid_columnconfigure(0, weight=1) 
window.grid_columnconfigure(1, weight=1) 
window.grid_rowconfigure(0, weight=0)    
window.grid_rowconfigure(1, weight=1)   
window.grid_rowconfigure(2, weight=0)   
window.grid_rowconfigure(3, weight=0)    
window.grid_rowconfigure(4, weight=1)   
window.grid_rowconfigure(5, weight=0)    

# Tiêu đề chính
lbTitle = tk.Label(window, text="Traffic Sign Recognition",
                   font=("Segoe UI", 28, "bold"), fg="#ffffff", background="#023246")
lbTitle.grid(row=0, column=0, columnspan=2, sticky="nsew", ipady=20)

tieudeImageOri = tk.Label(window, text="Ảnh Gốc", 
                          font=('Segoe UI', 16, 'bold'), fg="#023246", bg="#D4D4CE")
tieudeImageOri.grid(row=2, column=0, sticky="s", pady=(20, 5)) 

tieudeResult = tk.Label(window, text="Kết Quả", 
                        font=('Segoe UI', 16, 'bold'), fg="#023246", bg="#D4D4CE")
tieudeResult.grid(row=2, column=1, sticky="s", pady=(20, 5))

# Tạo Frame chứa ảnh gốc và kết quả
frameImageOri = Frame(window, width=400, height=400, bg="#F6F6F6", border=2, relief="ridge")
frameImageOri.grid(row=3, column=0, padx=30, pady=(0, 20), sticky="n")
frameImageOri.pack_propagate(False)

frameResult = Frame(window, width=400, height=400, bg="#F6F6F6", border=2, relief="ridge")
frameResult.grid(row=3, column=1, padx=30, pady=(0, 20), sticky="n")
frameResult.pack_propagate(False)

lblResult = Label(frameResult, text="", font=('Segoe UI', 18, 'bold'), 
                  fg="#023246", bg="#F6F6F6", wraplength=380, justify="center")
lblResult.pack(expand=True, fill="both", padx=10, pady=10)

# Tạo Frame chứa nút bấm
frameButtons = tk.Frame(window, bg="#D4D4CE")
frameButtons.grid(row=5, column=0, columnspan=2, pady=(0, 40)) # pady dưới 40px

bntBrowse = tk.Button(frameButtons, text="Browse", fg="#ffffff", bg="#287094",
    font=('Segoe UI', 14, 'bold'), width=15, height=2, relief="raised", cursor="hand2",
    command=lambda: BrowseImage(frameImageOri))
bntBrowse.pack(side="left", padx=30)

bntPredict = tk.Button(frameButtons, text="Nhận dạng", fg="#ffffff", bg="#008000",
    font=('Segoe UI', 14, 'bold'), width=15, height=2, relief="raised", cursor="hand2",
    command=predict_sign)
bntPredict.pack(side="left", padx=30)

window.mainloop()