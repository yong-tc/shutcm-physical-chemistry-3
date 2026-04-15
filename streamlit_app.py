import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline, PchipInterpolator

# ==================== 页面配置 ====================
st.set_page_config(page_title="最大气泡法测定表面张力", layout="wide")
st.title("🧪 最大气泡法测定溶液的表面张力")
st.markdown("根据液柱差数据计算仪器常数、各浓度溶液的表面张力，绘制 σ-c 曲线，并计算指定浓度下的表面吸附量。")

# ==================== 初始化 session_state ====================
if "data" not in st.session_state:
    # 内置示例数据（根据PDF）
    default_data = {
        "溶液": ["蒸馏水", "1.56%乙醇", "3.125%乙醇", "6.25%乙醇", "12.5%乙醇", "25%乙醇", "50%乙醇", "100%乙醇"],
        "体积比 (%)": [0, 1.56, 3.125, 6.25, 12.5, 25, 50, 100],
        "液柱差 ΔP": [0.228, 0.220, 0.203, 0.177, 0.132, 0.119, 0.090, 0.073]
    }
    st.session_state.data = pd.DataFrame(default_data)
    st.session_state.K = None
    st.session_state.calc_df = None
    st.session_state.fit_result = None
    st.session_state.sigma_water = 72.75e-3  # N/m, 20℃
    st.session_state.temp = 20.0

# ==================== 辅助函数 ====================
def vol_to_molar_concentration(vol_percent, ethanol_density=0.789, ethanol_molar_mass=46.07):
    """
    将体积百分比（假设为乙醇体积占比）转换为摩尔浓度 (mol/m³)
    vol_percent: 例如 12.5 表示 12.5% 乙醇
    返回 mol/m³
    """
    if vol_percent == 0:
        return 0.0
    vol_ethanol = vol_percent  # mL
    mass_ethanol = vol_ethanol * ethanol_density  # g
    moles_ethanol = mass_ethanol / ethanol_molar_mass
    volume_solution = 100  # mL = 0.1 L
    conc_mol_per_L = moles_ethanol / (volume_solution / 1000)  # mol/L
    return conc_mol_per_L * 1000  # mol/m³

def calculate_surface_tension(df, sigma_water, deltaP_water):
    """
    计算仪器常数 K，各溶液的表面张力 σ，以及摩尔浓度 c
    返回新 DataFrame 及仪器常数 K
    """
    df = df.copy()
    K = sigma_water / deltaP_water
    df["表面张力 σ (N/m)"] = K * df["液柱差 ΔP"]
    # 计算摩尔浓度 (mol/m³) – 乙醇密度和摩尔质量使用全局设置
    df["摩尔浓度 c (mol/m³)"] = df["体积比 (%)"].apply(
        lambda x: vol_to_molar_concentration(x, st.session_state.ethanol_density, st.session_state.ethanol_molar_mass)
    )
    return df, K

def numerical_derivative(f, x, dx=1e-6):
    """中心差分法求一阶导数"""
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def compute_adsorption(df, target_vol_percent, temp, ethanol_density=0.789, ethanol_molar_mass=46.07):
    """
    在目标浓度点（体积百分比）计算吸附量 Γ
    使用样条插值得到 σ(c) 曲线，然后数值求导
    返回 Γ (mol/m²)
    """
    # 提取非零浓度数据（蒸馏水为0浓度，表面张力已知但导数无定义）
    df_pos = df[df["体积比 (%)"] > 0].copy()
    if len(df_pos) < 3:
        return None
    c = df_pos["摩尔浓度 c (mol/m³)"].values
    sigma = df_pos["表面张力 σ (N/m)"].values
    # 排序
    idx = np.argsort(c)
    c_sorted = c[idx]
    sigma_sorted = sigma[idx]
    # 插值函数
    try:
        f = PchipInterpolator(c_sorted, sigma_sorted)
    except:
        f = CubicSpline(c_sorted, sigma_sorted)
    # 目标浓度
    target_c = vol_to_molar_concentration(target_vol_percent, ethanol_density, ethanol_molar_mass)
    if target_c < c_sorted.min() or target_c > c_sorted.max():
        st.warning(f"目标浓度 {target_c:.0f} mol/m³ 超出数据范围，无法计算导数。")
        return None
    # 数值求导 dσ/dc
    dsigma_dc = numerical_derivative(f, target_c, dx=1e-3)
    # 吸附量 Γ = - (c/(RT)) * (dσ/dc)
    R = 8.314  # J/(mol·K)
    T = temp + 273.15
    Gamma = - (target_c / (R * T)) * dsigma_dc
    return Gamma, target_c, dsigma_dc

# ==================== 侧边栏：参数设置 ====================
st.sidebar.header("⚙️ 实验参数")
temp = st.sidebar.number_input("实验温度 (℃)", value=st.session_state.temp, step=0.1, format="%.1f")
sigma_water_mNm = st.sidebar.number_input("纯水表面张力 (mN/m)", value=72.75, step=0.01, format="%.2f")
sigma_water = sigma_water_mNm * 1e-3
ethanol_density = st.sidebar.number_input("乙醇密度 (g/mL)", value=0.789, step=0.001, format="%.3f")
ethanol_molar_mass = st.sidebar.number_input("乙醇摩尔质量 (g/mol)", value=46.07, step=0.01, format="%.2f")

# 保存到 session_state
st.session_state.sigma_water = sigma_water
st.session_state.temp = temp
st.session_state.ethanol_density = ethanol_density
st.session_state.ethanol_molar_mass = ethanol_molar_mass

# ==================== 数据输入 ====================
st.subheader("📝 实验数据录入")
data_source = st.radio("数据来源", ["使用内置示例数据", "手动编辑表格", "上传 CSV 文件"], horizontal=True)

if data_source == "使用内置示例数据":
    df_input = st.session_state.data.copy()
elif data_source == "手动编辑表格":
    df_input = st.data_editor(
        st.session_state.data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "溶液": st.column_config.TextColumn("溶液"),
            "体积比 (%)": st.column_config.NumberColumn("体积比 (%)", min_value=0, max_value=100, step=0.1),
            "液柱差 ΔP": st.column_config.NumberColumn("液柱差 ΔP", format="%.3f")
        }
    )
else:
    uploaded = st.file_uploader("上传 CSV 文件 (需包含列: 溶液, 体积比 (%), 液柱差 ΔP)", type="csv")
    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.dataframe(df_input, use_container_width=True)
    else:
        st.stop()

st.session_state.data = df_input

# ==================== 计算 ====================
if st.button("🔍 计算表面张力及吸附量"):
    if df_input.empty:
        st.error("无有效数据")
    else:
        # 获取纯水对应的液柱差
        water_row = df_input[df_input["溶液"].str.contains("水", na=False)]
        if water_row.empty:
            st.error("未找到纯水（蒸馏水）数据，请确保表格中包含“蒸馏水”行")
            st.stop()
        deltaP_water = water_row.iloc[0]["液柱差 ΔP"]
        # 计算
        df_calc, K = calculate_surface_tension(df_input, sigma_water, deltaP_water)
        st.session_state.K = K
        st.session_state.calc_df = df_calc
        
        st.success(f"仪器常数 K = {K:.4f} m")
        st.subheader("📊 表面张力及浓度计算结果")
        display_cols = ["溶液", "体积比 (%)", "液柱差 ΔP", "表面张力 σ (N/m)", "摩尔浓度 c (mol/m³)"]
        st.dataframe(df_calc[display_cols], use_container_width=True)
        
        # 绘制 σ-c 曲线
        df_plot = df_calc[df_calc["摩尔浓度 c (mol/m³)"] > 0]  # 剔除纯水点（浓度为0）
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["摩尔浓度 c (mol/m³)"],
            y=df_plot["表面张力 σ (N/m)"],
            mode='lines+markers',
            name='σ-c 曲线',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="乙醇水溶液表面张力 σ 与浓度 c 的关系",
            xaxis_title="摩尔浓度 c (mol/m³)",
            yaxis_title="表面张力 σ (N/m)",
            width=700, height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 吸附量计算
        st.subheader("🧪 表面吸附量 Γ 计算")
        # 让用户选择要计算的溶液（体积百分比）
        valid_solutions = df_calc[df_calc["体积比 (%)"] > 0]["溶液"].tolist()
        if valid_solutions:
            target_solution = st.selectbox("选择要计算吸附量的溶液", valid_solutions)
            target_vol = df_calc[df_calc["溶液"] == target_solution]["体积比 (%)"].values[0]
            result = compute_adsorption(df_calc, target_vol, temp, ethanol_density, ethanol_molar_mass)
            if result is not None:
                Gamma, target_c, dsigma_dc = result
                st.success(f"**溶液：{target_solution}**")
                st.write(f"- 摩尔浓度 c = {target_c:.2f} mol/m³")
                st.write(f"- 切线斜率 dσ/dc = {dsigma_dc:.4e} N·m²/mol")
                st.write(f"- 吸附量 Γ = {Gamma:.4e} mol/m²")
                # 保存到 session_state 用于报告
                st.session_state.Gamma = Gamma
                st.session_state.target_c = target_c
                st.session_state.dsigma_dc = dsigma_dc
                st.session_state.target_solution = target_solution
            else:
                st.error("计算吸附量失败，请检查数据")
        else:
            st.info("无有效溶液数据，无法计算吸附量")

# ==================== 报告生成 ====================
st.markdown("---")
st.subheader("🖨️ 生成实验报告（PDF）")
if st.button("📄 生成并打印报告"):
    if st.session_state.calc_df is None:
        st.warning("请先点击上方按钮进行计算")
    else:
        df_calc = st.session_state.calc_df
        K = st.session_state.K
        # 准备表格 HTML
        display_cols = ["溶液", "体积比 (%)", "液柱差 ΔP", "表面张力 σ (N/m)", "摩尔浓度 c (mol/m³)"]
        table_html = df_calc[display_cols].to_html(index=False)
        # 获取吸附量计算结果（如果已计算）
        gamma_html = ""
        if hasattr(st.session_state, 'Gamma') and st.session_state.Gamma is not None:
            gamma_html = f"""
            <div class="info">
                <strong>吸附量计算结果（{st.session_state.target_solution}）：</strong><br>
                摩尔浓度 c = {st.session_state.target_c:.2f} mol/m³<br>
                切线斜率 dσ/dc = {st.session_state.dsigma_dc:.4e} N·m²/mol<br>
                吸附量 Γ = {st.session_state.Gamma:.4e} mol/m²
            </div>
            """
        # 重新生成 σ-c 曲线图（确保在报告中显示）
        df_plot = df_calc[df_calc["摩尔浓度 c (mol/m³)"] > 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["摩尔浓度 c (mol/m³)"],
            y=df_plot["表面张力 σ (N/m)"],
            mode='lines+markers',
            name='σ-c 曲线',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="乙醇水溶液表面张力 σ 与浓度 c 的关系",
            xaxis_title="摩尔浓度 c (mol/m³)",
            yaxis_title="表面张力 σ (N/m)",
            width=700, height=500
        )
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        full_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>最大气泡法测定表面张力实验报告</title>
            <style>
                body {{ font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin: 30px 0; page-break-inside: avoid; break-inside: avoid; }}
                .info {{ margin: 20px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #2c3e50; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>最大气泡法测定溶液的表面张力实验报告</h1>
            <p>生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>1. 实验数据及计算结果</h2>
            {table_html}
            <div class="info">
                <strong>实验条件：</strong> 温度 {temp} ℃，纯水表面张力 {sigma_water_mNm:.2f} mN/m<br>
                <strong>仪器常数 K：</strong> {K:.4f} m
            </div>
            {gamma_html}
            <h2>2. σ-c 曲线</h2>
            <div class="chart">
                {fig_html}
            </div>
            <h2>3. 实验结论</h2>
            <p>随着乙醇浓度增加，溶液表面张力逐渐降低，呈现典型的表面活性物质吸附行为。根据 Gibbs 吸附公式计算得到吸附量 Γ，表明乙醇在溶液表面产生正吸附。</p>
            <script>
                window.onload = function() {{ window.print(); }};
            </script>
        </body>
        </html>
        """
        st.components.v1.html(full_html, height=0, scrolling=False)
        st.success("报告已生成，请在弹出的打印对话框中选择「另存为 PDF」")

# ==================== 数据导出 ====================
if st.session_state.calc_df is not None:
    st.subheader("💾 导出计算结果")
    csv = st.session_state.calc_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下载计算结果 CSV",
        data=csv,
        file_name="surface_tension_results.csv",
        mime="text/csv"
    )

# ==================== 实验原理说明 ====================
with st.expander("ℹ️ 实验原理及数据处理方法"):
    st.markdown(r"""
    **1. 最大气泡法原理**  
    当气泡从毛细管口形成并即将脱离时，气泡内外压力差最大，满足 Laplace 公式：
    \[
    \Delta P_{\text{max}} = \frac{2\sigma}{r}
    \]
    其中 \(r\) 为毛细管半径。对于同一根毛细管，仪器常数 \(K = \frac{r}{2}\)，则 \(\sigma = K \cdot \Delta P\)。  
    通过测定已知表面张力的纯水（20℃时 \(\sigma_{\text{水}} = 72.75 \text{ mN/m}\)）的 \(\Delta P_{\text{水}}\)，可求得 \(K = \sigma_{\text{水}} / \Delta P_{\text{水}}\)。

    **2. 表面吸附量计算**  
    根据 Gibbs 吸附公式：
    \[
    \Gamma = -\frac{c}{RT} \left( \frac{d\sigma}{dc} \right)_T
    \]
    在 \(\sigma-c\) 曲线上某浓度点作切线，得到斜率 \(d\sigma/dc\)，代入公式即可求得吸附量 \(\Gamma\)（单位：mol/m²）。

    **3. 浓度换算**  
    乙醇体积百分比转换为摩尔浓度（mol/m³）：
    \[
    c = \frac{\rho_{\text{乙醇}} \cdot V_{\text{乙醇}} / M_{\text{乙醇}}}{V_{\text{溶液}} / 1000} \times 1000
    \]
    式中 \(\rho_{\text{乙醇}}\) 为乙醇密度（g/mL），\(M_{\text{乙醇}}\) 为乙醇摩尔质量（g/mol）。
    """)
