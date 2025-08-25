import streamlit as st
import pandas as pd
import joblib

# Tải mô hình đã được huấn luyện
# Đảm bảo file .joblib nằm cùng thư mục với file app.py này
try:
    model = joblib.load('G:/My Drive/Machine Learning/streamlit/best_stroke_prediction_model.joblib')
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file mô hình 'best_stroke_prediction_model.joblib'. Vui lòng đảm bảo file này tồn tại trong cùng thư mục.")
    st.stop()


st.set_page_config(page_title="Dự Đoán Đột Quỵ", page_icon="⚕️")

st.title('⚕️ Ứng dụng Dự đoán Nguy cơ Đột quỵ')
st.write('Điền thông tin của bệnh nhân vào các trường dưới đây để hệ thống đưa ra dự đoán về nguy cơ đột quỵ.')

# ---- Tạo Form Nhập Liệu ----
with st.form("prediction_form"):
    st.header("Thông tin Bệnh nhân")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Giới tính', ['Male', 'Female'])
        age = st.number_input('Tuổi', min_value=1, max_value=120, value=55)
        hypertension = st.selectbox('Tiền sử cao huyết áp?', [0, 1], format_func=lambda x: 'Có' if x == 1 else 'Không')
        heart_disease = st.selectbox('Tiền sử bệnh tim?', [0, 1], format_func=lambda x: 'Có' if x == 1 else 'Không')
        ever_married = st.selectbox('Đã từng kết hôn chưa?', ['Yes', 'No'])

    with col2:
        work_type = st.selectbox('Loại hình công việc', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        Residence_type = st.selectbox('Khu vực sinh sống', ['Urban', 'Rural'])
        avg_glucose_level = st.slider('Mức đường huyết trung bình', 50.0, 300.0, 100.0)
        bmi = st.slider('Chỉ số BMI', 10.0, 100.0, 28.5)
        smoking_status = st.selectbox('Tình trạng hút thuốc', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    # Nút submit của form
    submitted = st.form_submit_button('Dự đoán')

# ---- Xử lý sau khi nhấn nút ----
if submitted:
    # Tạo DataFrame từ input của người dùng
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Thực hiện dự đoán
    try:
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.subheader('Kết quả Dự đoán')

        if prediction == 1:
            st.error(f'**Cảnh báo: Bệnh nhân có nguy cơ bị đột quỵ cao.**')
            st.progress(prediction_proba)
            st.metric(label="Tỷ lệ rủi ro ước tính", value=f"{prediction_proba:.2%}")
        else:
            st.success(f'**Thông báo: Bệnh nhân có nguy cơ bị đột quỵ thấp.**')
            st.progress(prediction_proba)
            st.metric(label="Tỷ lệ rủi ro ước tính", value=f"{prediction_proba:.2%}")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")

st.write('---')
st.warning('**Lưu ý:** Kết quả dự đoán này chỉ mang tính chất tham khảo và không thể thay thế cho chẩn đoán y tế chuyên nghiệp. Luôn tham khảo ý kiến bác sĩ để có kết luận chính xác nhất.')