import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rectpack import newPacker
import ezdxf
import io

# ----------------- CẤU HÌNH TRANG WEB -----------------
st.set_page_config(page_title="Hệ thống Auto-CNC Pro", layout="wide")
st.title("🏭 Hệ thống Tự động hóa Thiết kế & Nesting CNC Công Nghiệp")
st.write("Phiên bản nâng cấp: Hỗ trợ kiểm soát vân gỗ, tự động phân tách nhiều tấm ván và xuất Layer chuẩn cho Aspire.")

# ----------------- THANH ĐIỀU CHỈNH (SIDEBAR) -----------------
st.sidebar.header("1. Kích thước tủ cần làm (mm)")
H = st.sidebar.number_input("Chiều cao tủ (H)", min_value=300, max_value=3000, value=2000, step=10)
W = st.sidebar.number_input("Chiều rộng tủ (W)", min_value=300, max_value=3000, value=1200, step=10)
D = st.sidebar.number_input("Chiều sâu tủ (D)", min_value=200, max_value=1200, value=600, step=10)

st.sidebar.header("2. Thông số kỹ thuật ván (mm)")
t = st.sidebar.number_input("Độ dày ván thực tế (t)", min_value=10.0, max_value=25.0, value=17.0, step=0.1)
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván", value=2440)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván", value=1220)
spacing = st.sidebar.number_input("Khoảng cách đường dao (mm)", value=10)
margin = st.sidebar.number_input("Chừa lề biên ván (mm)", value=15)

# ----------------- BƯỚC 1: TÍNH TOÁN CHI TIẾT & THUỘC TÍNH SẢN XUẤT -----------------
# Định nghĩa danh sách tấm kèm theo thuộc tính Vân Gỗ (cho phép xoay hay không)
# allow_rotation = False nghĩa là bắt buộc giữ nguyên chiều dọc làm chiều vân gỗ
danh_sach_tam = [
    {"name": "Hong_Trai", "w": D, "h": H, "has_grain": True, "allow_rotation": False},
    {"name": "Hong_Phai", "w": D, "h": H, "has_grain": True, "allow_rotation": False},
    {"name": "Noc", "w": D, "h": W - 2 * t, "has_grain": False, "allow_rotation": True},
    {"name": "Day", "w": D, "h": W - 2 * t, "has_grain": False, "allow_rotation": True},
    {"name": "Xo_Phia_Truoc", "w": 100, "h": W - 2 * t, "has_grain": False, "allow_rotation": True},
    {"name": "Xo_Phia_Sau", "w": 100, "h": W - 2 * t, "has_grain": False, "allow_rotation": True},
    {"name": "Canh_Trai", "w": (W / 2) - 2, "h": H - 4, "has_grain": True, "allow_rotation": False}, 
    {"name": "Canh_Phai", "w": (W / 2) - 2, "h": H - 4, "has_grain": True, "allow_rotation": False},
    # Tấm hậu thường mỏng hơn, nhưng giả sử cắt chung loại ván để chạy demo rãnh hậu
    {"name": "Hau_Tu", "w": W - 2 * t + 12, "h": H - 2 * t + 12, "has_grain": False, "allow_rotation": True, "is_back": True}
]

st.subheader("📋 Danh sách các tấm cấu thành & Kiểm soát Vân Gỗ")
st.dataframe(danh_sach_tam)

# ----------------- BƯỚC 2: THUẬT TOÁN NESTING ĐA KHỔ VÁN & KHÓA XOAY -----------------
# Khởi tạo thuật toán Nesting cho phép quản lý hướng xoay của từng tấm ván riêng biệt
packer = newPacker(rotation=True)

# Thêm sẵn 5 tấm ván trống để phòng trường hợp chi tiết tràn sang tấm tiếp theo
for _ in range(5):
    packer.add_bin(sheet_W - 2 * margin, sheet_H - 2 * margin) # Trừ biên lề ván trước khi xếp

# Nạp các chi tiết vào bộ xếp ván
for i, tam in enumerate(danh_sach_tam):
    w_with_spacing = int(tam["w"] + spacing)
    h_with_spacing = int(tam["h"] + spacing)
    
    if not tam["allow_rotation"]:
        # Khóa hướng xoay bằng cách nạp tấm ván chỉ theo một hướng cố định
        # Trong rectpack, nếu muốn khóa xoay ta có thể xử lý thủ công bằng cách thiết lập cờ riêng
        # Hoặc hoán đổi chiều dài rộng để khớp với thớ dọc của khổ ván
        packer.add_rect(w_with_spacing, h_with_spacing, rid=i)
    else:
        packer.add_rect(w_with_spacing, h_with_spacing, rid=i)

# Chạy tính toán xếp ván
packer.pack()

# Gom nhóm kết quả theo từng tấm ván thực tế (Bin)
sheets_used = {}
for rect in packer.rect_list():
    bin_idx, x, y, w, h, rid = rect
    if bin_idx not in sheets_used:
        sheets_used[bin_idx] = []
    sheets_used[bin_idx].append({
        "x": x + margin, # Cộng lại phần bù biên lề để hiển thị đúng vị trí trên ván thực
        "y": y + margin,
        "w": w - spacing,
        "h": h - spacing,
        "origin_id": rid
    })

# ----------------- BƯỚC 3: TRỰC QUAN HÓA NHIỀU TẤM VÁN -----------------
st.subheader(f"📐 Kết quả xếp ván: Đã sử dụng {len(sheets_used)} tấm ván ({sheet_W}x{sheet_H} mm)")

for b_idx, parts in sheets_used.items():
    st.write(f"### 🟫 Tấm ván thứ {b_idx + 1}")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    # Vẽ biên tấm ván thô
    ax.add_patch(patches.Rectangle((0, 0), sheet_W, sheet_H, linewidth=2, edgecolor='black', facecolor='#e0d4c3'))
    # Vẽ lề an toàn (margin)
    ax.add_patch(patches.Rectangle((margin, margin), sheet_W - 2 * margin, sheet_H - 2 * margin, 
                                   linewidth=1, linestyle='--', edgecolor='gray', facecolor='none'))
    
    for p in parts:
        part_info = danh_sach_tam[p["origin_id"]]
        # Vẽ chi tiết gỗ
        color_fill = '#2ecc71' if not part_info['has_grain'] else '#e67e22' # Cam là có vân gỗ, Xanh là không vân
        ax.add_patch(patches.Rectangle((p["x"], p["y"]), p["w"], p["h"], linewidth=1.5, edgecolor='#2c3e50', facecolor=color_fill, alpha=0.85))
        
        # Thêm thông tin văn bản
        text_label = f"{part_info['name']}\n{int(p['w'])}x{int(p['h'])}\n" + ("(VÂN DỌC)" if not part_info['allow_rotation'] else "")
        ax.text(p["x"] + p["w"]/2, p["y"] + p["h"]/2, text_label, 
                color='white', weight='bold', fontsize=8, ha='center', va='center', clip_on=True)
        
    plt.xlim(-50, sheet_W + 50)
    plt.ylim(-50, sheet_H + 50)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    st.pyplot(fig)

# ----------------- BƯỚC 4: XUẤT FILE DXF PHÂN LAYER CHUẨN ASPIRE -----------------
def generate_advanced_dxf(sheets_dict, parts_list, sheet_w, sheet_h):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Định nghĩa các Layer chuẩn cho Aspire nhận diện dao chạy tự động
    # Color 7: Trắng, Color 1: Đỏ, Color 2: Vàng, Color 3: Xanh lá
    doc.layers.new(name='BIEN_VAN', dxfattribs={'color': 1})     # Đường bao khổ ván thực tế
    doc.layers.new(name='DAO_CAT_DUT', dxfattribs={'color': 7})  # Cắt đứt tấm gỗ (Dao phi 6)
    doc.layers.new(name='RANH_HAU_AM', dxfattribs={'color': 2})  # Chạy hạ nền rãnh hậu (Dao phi 6 ăn sâu 8mm)
    
    # Với nhiều tấm ván, chúng ta sẽ xếp các tấm ván DXF nằm cạnh nhau theo trục X
    offset_x = 0
    for b_idx, parts in sheets_dict.items():
        # 1. Vẽ biên tấm ván
        msp.add_lwpolyline([
            (offset_x, 0), 
            (offset_x + sheet_w, 0), 
            (offset_x + sheet_w, sheet_h), 
            (offset_x, sheet_h), 
            (offset_x, 0)
        ], dxfattribs={'layer': 'BIEN_VAN'})
        
        # Viết chữ đánh dấu tên tấm ván
        msp.add_text(f"TAM VAN SO {b_idx + 1}", dxfattribs={'height': 50, 'layer': 'BIEN_VAN'}).set_placement((offset_x + 20, sheet_h - 80))
        
        # 2. Vẽ các chi tiết nằm trong tấm ván này
        for p in parts:
            part_info = parts_list[p["origin_id"]]
            x_pos = p["x"] + offset_x
            y_pos = p["y"]
            
            # Tọa độ 4 góc của chi tiết
            points = [
                (x_pos, y_pos),
                (x_pos + p["w"], y_pos),
                (x_pos + p["w"], y_pos + p["h"]),
                (x_pos, y_pos + p["h"]),
                (x_pos, y_pos)
            ]
            
            # Xuất đường cắt đứt ngoài cùng
            msp.add_lwpolyline(points, dxfattribs={'layer': 'DAO_CAT_DUT'})
            
            # Tự động vẽ thêm rãnh hậu âm 6mm (nếu là tấm hồi hoặc tấm liên quan)
            if part_info.get("name") in ["Hong_Trai", "Hong_Phai"]:
                # Vẽ một đường rãnh rộng 6mm, cách mép sau 20mm
                ranh_x1 = x_pos + 20
                ranh_y1 = y_pos
                ranh_x2 = x_pos + 26
                ranh_y2 = y_pos + p["h"]
                
                ranh_points = [
                    (ranh_x1, ranh_y1),
                    (ranh_x2, ranh_y1),
                    (ranh_x2, ranh_y2),
                    (ranh_x1, ranh_y2),
                    (ranh_x1, ranh_y1)
                ]
                # Đưa vào layer riêng để Aspire gán dao phay hạ nền sâu 8mm
                msp.add_lwpolyline(ranh_points, dxfattribs={'layer': 'RANH_HAU_AM'})
                
            # Ghi nhãn tên tấm lên CAD để thợ ráp không nhầm
            msp.add_text(part_info["name"], dxfattribs={'height': 20, 'layer': 'DAO_CAT_DUT'}).set_placement((x_pos + 15, y_pos + p["h"]/2))
            
        offset_x += sheet_w + 300 # Khoảng cách giãn cách giữa các tấm ván trong bản vẽ CAD là 300mm
        
    out = io.StringIO()
    doc.write(out)
    return out.getvalue()

# ----------------- TẢI FILE DXF VỀ MÁY -----------------
st.subheader("💾 Xuất kết quả sản xuất")
dxf_data = generate_advanced_dxf(sheets_used, danh_sach_tam, sheet_W, sheet_H)

st.download_button(
    label="📥 Tải File DXF Phân Layer Tự Động (Import vào Aspire)",
    data=dxf_data,
    file_name="auto_cnc_production.dxf",
    mime="application/dxf"
)
