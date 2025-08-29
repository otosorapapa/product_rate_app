import streamlit as st

st.set_page_config(page_title="賃率ダッシュボード", layout="wide")

st.title("製品賃率ダッシュボード")
st.caption("📊 Excel（標賃 / R6.12）から賃率KPIを自動計算し、SKU別の達成状況を可視化します。")

st.markdown("""
**次のページへ:**  
1) 👉 [データ入力 & 取り込み](pages/01_データ入力.py)  
2) 📈 [ダッシュボード](pages/02_ダッシュボード.py)  
3) 🧮 [標準賃率 計算/感度分析](pages/03_標準賃率計算.py)
""")

st.info("まずは『データ入力 & 取り込み』で Excel を読み込むか、サンプルを使用してください。")
