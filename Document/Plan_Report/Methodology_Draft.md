# 📝 Methodology Draft — Hà Quốc Khánh

> **Mục tiêu**: Tài liệu hóa quy trình xây dựng mô hình dự báo để đưa vào báo cáo NeurIPS. (Cập nhật cho Ngày 8)

## 1. Dữ liệu & Tiền xử lý (Preprocessing)
- **Detrending Strategy**: Doanh thu và giá vốn hàng bán (COGS) có xu hướng tăng trưởng mạnh qua các năm. Chúng tôi thực hiện chuẩn hóa (Normalization) bằng cách chia giá trị `Revenue` và `COGS` cho `Mean` trung bình của từng năm. Khi dự báo tương lai, chúng tôi ngoại suy (extrapolate) mức cơ sở (base level) bằng hàm hồi quy tuyến tính trên 3 năm gần nhất. Điều này giúp mô hình tập trung học các biến động ngắn hạn và tính chu kỳ (Seasonality) mà không bị nhiễu bởi xu hướng tăng trưởng dài hạn.

## 2. Đặc trưng (Feature Engineering)
- **Calendar Features**: Tháng, Ngày, Ngày trong tuần, Cuối tuần, Ngày trả lương (15, 30, 31).
- **Fourier Terms**: Sử dụng biến đổi Fourier (K=4) theo cả chu kỳ năm (365.25 ngày) và chu kỳ tuần (7 ngày) để mô hình hóa tính thời vụ nhịp nhàng mà không cần dùng biến Lag, giúp dễ dàng dự báo dài hạn (18 tháng).
- **Tet Holiday Effects**: Xây dựng bộ đặc trưng riêng cho dịp Tết Nguyên Đán (`is_pre_tet`, `is_tet_week`, `is_post_tet` và `tet_score` suy giảm theo hàm mũ) để mô hình bắt được các đỉnh mua sắm đặc thù tại Việt Nam.
- **YoY Anchors**: Tạo các biến trễ cùng kỳ năm ngoái (YoY lag) và trung bình trượt 7 ngày cùng kỳ (YoY roll7) để mô hình nắm bắt được quán tính của thị trường.

## 3. Kiến trúc Mô hình (Ensemble Architecture)
Mô hình cuối cùng là sự kết hợp (Blending) của 3 mô hình độc lập, sau đó được tối ưu hóa bằng kỹ thuật Pseudo-Labeling:
1. **Seasonal Profile**: Dựa trên phân phối trung bình theo tháng và ngày trong 3 năm gần nhất.
2. **LightGBM Fourier**: Sử dụng 49 biến (Fourier K=4 và Calendar) để bắt nhịp thời vụ.
3. **LightGBM YoY**: Bổ sung các biến cùng kỳ năm ngoái để nắm bắt quán tính tăng trưởng.

Đặc biệt, chúng tôi áp dụng kỹ thuật **Pseudo-Labeling (Self-Training)**: Sử dụng kết quả tốt nhất từ ensemble để làm nhãn giả cho tập Test, sau đó tái huấn luyện toàn bộ hệ thống trên tập dữ liệu mở rộng. Kỹ thuật này giúp mô hình "mượt hóa" các dự báo và nắm bắt cấu trúc ẩn của tập Test.

## 4. Đánh giá (Validation & Results)
- **Strategy**: Multi-cutoff Backtest kết hợp với Pseudo-Labeling refinement.
- **Trọng số Ensemble**: Được tự động tính toán dựa trên Inverse-MAE.
- **Kết quả**: Kỹ thuật Pseudo-Labeling đã giúp cải thiện MAE thêm khoảng 1-2% trên Leaderboard, đạt mức ổn định cao nhất cho dự báo 18 tháng.
- **Explainability**: Fourier terms và tính thời vụ của Tết đóng vai trò then chốt, trong khi Pseudo-labeling giúp tinh chỉnh sai số ở các giai đoạn biến động mạnh. 

## 5. Kết luận
- Pipeline dự báo dài hạn 18 tháng đã được tối ưu hoàn toàn (Vectorized) không dùng Lags đệ quy, tránh được Data Leakage.
- Mô hình LightGBM Fourier cho hiệu suất tốt nhất, nhưng việc kết hợp Ensemble giúp tăng cường độ ổn định ở những khoảng thời gian bất thường.
- Kết quả cuối cùng (`submission_final.csv`) đã qua 7 bước kiểm tra tự động (Data Integrity) và sẵn sàng để nộp lên Kaggle.
