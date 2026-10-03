from MyLinear_Regression import LinearRegression
from MyLinear_Regression import MultipleLinearRegression
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# amount=[34,108,64,88,99,51]
# tip=[5,17,11,8,14,5]
# model=LinearRegression(amount,tip)
# model.fit()
# print(model.intercept,model.slope)
# test=[70,2000]
# p=model.predict(test)
# print(p)

# ## Multimodal Linear Regression Implementation
# # Training Data
# Product1 = [1, 2, 3, 4]
# Product2 = [4, 5, 8, 2]
# Weekly_Sales = [1, 6, 8, 12]
# # Create Object
# model = MultipleLinearRegression(
#     Product1,
#     Product2,
#     Weekly_Sales
# )
# # Check Length
# if model.check_training_length():
#     model.create_x_matrix()
#     model.calculate_coefficients()
#     # Scatter Plots
#     model.product1_vs_weekly_sales()
#     model.product1_with_line()
#     model.product2_vs_weekly_sales()
#     model.product2_with_line()
#     # Prediction
#     model.input_prediction_values()
#     if model.check_prediction_length():
#         model.predict()
#         model.actual_vs_predicted()




# STREAMLIT UI
st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="wide"
)
st.title("📊 Multiple Linear Regression")
st.write("Product1 and Product2 ke basis par Weekly Sales predict karein.")

# Convert string into numpy arrays
try:

    Product1 = [1, 2, 3, 4]
    Product2 = [4, 5, 8, 2]
    Weekly_Sales = [1, 6, 8, 12]
except:
    st.error("Please enter valid comma-separated numbers.")
    st.stop()

# CREATE MODEL
model = MultipleLinearRegression(Product1,Product2,Weekly_Sales)

# -------- TRAINING DATAFRAME --------
training_df = pd.DataFrame({
    "Product1": Product1,
    "Product2": Product2,
    "Weekly Sales": Weekly_Sales
})
st.subheader("Training Data")
st.dataframe(
    training_df,
    use_container_width=True
)

# CHECK LENGTH
if model.check_training_length():
    st.success("✅ Length of all arrays are same.")
else:
    st.error("❌ Length of all arrays are not same.")
    st.stop()

# X MATRIX
X_matrix = model.create_x_matrix()
st.subheader("X Matrix")
st.dataframe(
    X_matrix,
    use_container_width=True
)

# COEFFICIENTS
A = model.calculate_coefficients()
st.subheader("Regression Coefficients")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Intercept (a0)",f"{A[0]:.2f}")
with c2:
    st.metric("Product1 Slope (a1)",f"{A[1]:.2f}")
with c3:
    st.metric("Product2 Slope (a2)",f"{A[2]:.2f}")
st.write(f"### Regression Equation")

st.latex(f"Y = {A[0]:.2f} + "f"({A[1]:.2f})X_1 + "f"({A[2]:.2f})X_2")

# GRAPHS
st.header("2️⃣ Scatter Plots")

# Graph 1 and 2
graph_col1, graph_col2=st.columns(2)
with graph_col1:
    st.subheader("Product1 vs Weekly Sales")
    fig1 = model.product1_vs_weekly_sales()
    st.pyplot(fig1,width=500)
with graph_col2:
    st.subheader("Product1 vs Weekly Sales with Regression Line")
    fig2 = model.product1_with_line()
    st.pyplot(fig2,width=500)
# Graph 3 and 4
graph_col3, graph_col4=st.columns(2)
with graph_col3:
    st.subheader("Product2 vs Weekly Sales")
    fig3 = model.product2_vs_weekly_sales()
    st.pyplot(fig3, width=500)
with graph_col4:
    st.subheader("Product2 with Regression Line")
    fig4 = model.product2_with_line()
    st.pyplot(fig4, width=500)

# PREDICTION
st.header("3️⃣ Prediction")
col4, col5 = st.columns(2)
with col4:
    X1_input = st.text_input( "Enter X1 / Product1 values","5,6,7")
with col5:
    X2_input = st.text_input("Enter X2 / Product2 values","3,4,5")

# Convert prediction input
try:
    X1 = np.array(list(map(float, X1_input.split(","))))
    X2 = np.array(list(map(float, X2_input.split(","))))
except:
    st.error("Please enter valid prediction values.")
    st.stop()

# Check prediction length
if len(X1) == len(X2):
    st.success("✅ Length of X1 and X2 are same.")
else:
    st.error("❌ Length of X1 and X2 are not same.")
    st.stop()

# PREDICT BUTTON
if st.button("🔮 Predict Weekly Sales",type="primary"):
    Y_pred = model.predict(X1,X2)
    st.subheader("Predicted Values")
    st.write(Y_pred)
    st.subheader("Prediction Graph")
    fig5 = model.prediction_graph()
    st.pyplot(fig5,width=500)
