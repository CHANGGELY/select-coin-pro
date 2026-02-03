import streamlit as st

st.set_page_config(
    page_title="分析报告",
    page_icon="👋",
)

st.write("# 欢迎使用《马代策略查看器》 👋")

st.sidebar.success("请选择上述功能进行使用")

st.markdown(
    """
    本框架代码基于 [@chandler](https://bbs.quantclass.cn/user/40013) 老板的二次开发
    
    原贴链接：https://bbs.quantclass.cn/thread/58410
    
    ---
    
    > 更新说明
    > - 删掉目前未使用的一些脚本
    > - 新增“多策略资金曲线比较”页面

"""
)
