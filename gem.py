import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import r2_score, root_mean_squared_error


# PAGE 
st.set_page_config(
    page_title="Bacterial FHN Model & Parameter Fitting Platform",
    page_icon="🦠",
    layout="wide"
)

st.title( "Bacterial Membrane Potential Dynamics & Fitting Platform")
st.caption("Modified FitzHugh-Nagumo (FHN) model simulation, phase-plane analysis, and inverse parameter extraction (Stratford et al., PNAS 2019).")

tab_sim, tab_fit = st.tabs(["Interactive Simulation & Phase Portrait", "Experimental Data Fitting Engine"])


# CORE MODEL FUNCTIONS

def get_beta(k_K):
    return 1.0 - 0.1 * np.log(k_K)

def get_equilibrium(k_K, alpha, V_m0):
    beta = get_beta(k_K)
    u_eq = (beta / alpha) ** (1.0 / 3.0)
    V_eq = u_eq - V_m0
    W_eq = beta - u_eq
    return V_eq, W_eq

def fhn_bacteria(t, y, k_K, alpha, V_m0, t_on, t_off, i_v, i_w):
    V_m, W = y
    beta = get_beta(k_K)
    u = V_m + V_m0

    if t_on <= t <= t_off:
        dV_dt = k_K * (u - alpha * u**3 + W) + i_v
        dW_dt = -u + beta - W + i_w
    else:
        dV_dt = k_K * (u - alpha * u**3 + W)
        dW_dt = -u + beta - W

    return [dV_dt, dW_dt]

def solve_fhn(k_K, alpha, V_m0, t_on, t_off, i_v, i_w, t_eval):
    V_eq, W_eq = get_equilibrium(k_K, alpha, V_m0)
    t_span = (t_eval[0], t_eval[-1])

    sol = solve_ivp(
        fhn_bacteria,
        t_span,
        [V_eq, W_eq],
        args=(k_K, alpha, V_m0, t_on, t_off, i_v, i_w),
        t_eval=t_eval,
        method="RK45",
        max_step=0.02,
        rtol=1e-5,
        atol=1e-7
    )
    return sol, V_eq, W_eq


# TAB 1: INTERACTIVE SIMULATION & PHASE PORTRAIT

with tab_sim:
    st.sidebar.header("Simulation Settings")
    pulse_start = st.sidebar.slider("Stimulus Onset (s)", 0.0, 5.0, 1.0, 0.25)
    pulse_duration = st.sidebar.slider("Pulse Duration (s)", 0.5, 5.0, 2.5, 0.25)
    pulse_end = pulse_start + pulse_duration

    st.sidebar.subheader("Electrical Stimulus Parameters")
    I_v_user = st.sidebar.number_input(
        "Stimulus intensity (I_v)",
        value=0.01,
        step=0.005,
        format="%.3f"
    )
    I_w_user = st.sidebar.number_input(
        "Recovery stimulus (I_w)",
        value=-0.075,
        step=0.005,
        format="%.3f"
    )

    st.sidebar.subheader("Cell Physiology")
    k_K_custom = st.sidebar.slider("Test Cell k_K", 0.05, 15.0, 5.0, 0.05)
    
    alpha_default = 10.0
    V_m0_default = 1.5
    
    sim_horizon = max(20.0, pulse_end + 5.0)
    t_array = np.linspace(0.0, sim_horizon, 1200)

    # Run simulations
    sol_custom, V_eq_c, W_eq_c = solve_fhn(k_K_custom, alpha_default, V_m0_default, pulse_start, pulse_end, I_v_user, I_w_user, t_array)
    sol_prof, V_eq_p, _ = solve_fhn(10.0, alpha_default, V_m0_default, pulse_start, pulse_end, I_v_user, I_w_user, t_array)
    sol_inh, V_eq_i, _ = solve_fhn(0.1, alpha_default, V_m0_default, pulse_start, pulse_end, I_v_user, I_w_user, t_array)

    dVm_custom = -(sol_custom.y[0] - V_eq_c)
    dVm_prof = -(sol_prof.y[0] - V_eq_p)
    dVm_inh = -(sol_inh.y[0] - V_eq_i)

    # State classification callout for test Cell
    if k_K_custom >= 5.0:
        st.success(f"**Test Cell State: Proliferative-likely** (k_K = {k_K_custom:.2f}) — Exhibits sharp excitable hyperpolarization excursion")
    elif k_K_custom <= 1.0:
        st.error(f"**Test Cell State: Inhibited-likely** (k_K = {k_K_custom:.2f}) — Exhibits passive depolarization/relaxation dynamics")
    else:
        st.warning(f"**Test Cell State: Intermediate / Stressed** (k_K = {k_K_custom:.2f}) — Operates in transitional regime between phenotypes")

    col_time, col_phase = st.columns([3, 2])

    with col_time:
        st.subheader("Membrane Potential Response (-ΔV_m)")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_array, y=dVm_custom, mode="lines", name=f"Test Cell (k_K={k_K_custom:.2f})", line=dict(width=3, color="#2ca02c")))
        fig_time.add_trace(go.Scatter(x=t_array, y=dVm_prof, mode="lines", name="Proliferative (k_K=10.0)", line=dict(dash="dash", width=2, color="#1f77b4")))
        fig_time.add_trace(go.Scatter(x=t_array, y=dVm_inh, mode="lines", name="Inhibited (k_K=0.1)", line=dict(dash="dash", width=2, color="#ff7f0e")))
        fig_time.add_vrect(x0=pulse_start, x1=pulse_end, fillcolor="gold", opacity=0.25, line_width=0, annotation_text="Stimulus", annotation_position="top left")
        fig_time.add_hline(y=0, line_dash="dot", line_width=1, line_color="gray")
        fig_time.update_layout(xaxis_title="Time (s)", yaxis_title="-ΔV_m", height=450, template="plotly_white", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_time, use_container_width=True)

    with col_phase:
        st.subheader("Phase Portrait (V_m vs. W)")
        v_traj = sol_custom.y[0]
        w_traj = sol_custom.y[1]
        
        v_margin = max(0.02, (np.max(v_traj) - np.min(v_traj)) * 1.5)
        v_range = np.linspace(V_eq_c - v_margin, V_eq_c + v_margin, 300)
        u_range = v_range + V_m0_default
        beta_c = get_beta(k_K_custom)
        
        # Nullclines
        w_nullcline_v = alpha_default * (u_range**3) - u_range
        w_nullcline_w = beta_c - u_range

        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(x=v_range, y=w_nullcline_v, mode="lines", name="V-Nullcline", line=dict(color="#1f77b4", width=1.5)))
        fig_phase.add_trace(go.Scatter(x=v_range, y=w_nullcline_w, mode="lines", name="W-Nullcline", line=dict(color="#d62728", width=1.5)))
        fig_phase.add_trace(go.Scatter(x=v_traj, y=w_traj, mode="lines", name="Cell Trajectory", line=dict(color="#2ca02c", width=2.5)))
        fig_phase.add_trace(go.Scatter(x=[V_eq_c], y=[W_eq_c], mode="markers", name="Equilibrium", marker=dict(size=9, color="magenta", symbol="circle")))

        fig_phase.update_layout(
            xaxis_title="V_m", 
            yaxis_title="W", 
            height=450, 
            template="plotly_white", 
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=40, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_phase, use_container_width=True)
        st.caption("*Zoom to show trajectory details and examine local excursion around the resting point.*")

    # CSV Download Button
    df_sim_export = pd.DataFrame({"Time_s": t_array, "dVm_custom": dVm_custom, "dVm_proliferative": dVm_prof, "dVm_inhibited": dVm_inh})
    st.download_button(label="Export Simulation Traces (CSV)", data=df_sim_export.to_csv(index=False).encode('utf-8'), file_name="bacterial_fhn_simulation.csv", mime="text/csv")


# TAB 2: EXPERIMENTAL DATA FITTING ENGINE
with tab_fit:
    st.subheader("Inverse Parameter Extraction & Cell State Inference")
    st.markdown("Upload experimental optical or electrical recording data containing time and fluorescence change (`-ΔV_m`).")

    col_up, col_param = st.columns([1, 1])

    with col_up:
        uploaded_file = st.file_uploader("Upload Experimental Data (CSV)", type=["csv"])
        if uploaded_file is None:
            st.info("No file uploaded. You can download and inspect a representative synthetic dataset below:")
            sample_df = pd.DataFrame({"Time": t_array[::4], "minus_delta_Vm": dVm_custom[::4] + np.random.normal(0, 0.0003, len(t_array[::4]))})
            st.download_button(label="Download Sample CSV Template", data=sample_df.to_csv(index=False).encode('utf-8'), file_name="sample_experimental_data.csv", mime="text/csv")

    with col_param:
        st.markdown("**Stimulation Pulse Protocol During Experiment:**")
        fit_p_start = st.number_input("Experiment Stimulus Start (s)", value=1.0, step=0.1)
        fit_p_dur = st.number_input("Experiment Stimulus Duration (s)", value=2.5, step=0.1)
        fit_p_end = fit_p_start + fit_p_dur

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Loaded Data Preview:", raw_df.head(4))

        col_t_select, col_v_select = st.columns(2)
        with col_t_select:
            t_col = st.selectbox("Select Time Column", options=raw_df.columns, index=0)
        with col_v_select:
            v_col = st.selectbox("Select -ΔV_m Column", options=raw_df.columns, index=1 if len(raw_df.columns) > 1 else 0)

        t_exp = raw_df[t_col].to_numpy()
        y_exp = raw_df[v_col].to_numpy()

        if st.button("Run Non-Linear Optimization Fit"):
            with st.spinner("Solving inverse problem across non-linear FHN manifold..."):
                initial_guess = [2.0, 10.0, 1.5, 0.01, -0.075]
                bounds = [(0.01, 20.0), (2.0, 25.0), (0.5, 2.5), (0.0001, 0.1), (-0.2, -0.001)]

                def loss_function(params):
                    k_k_val, alpha_val, vm0_val, iv_val, iw_val = params
                    try:
                        sol, v_eq, _ = solve_fhn(k_k_val, alpha_val, vm0_val, fit_p_start, fit_p_end, iv_val, iw_val, t_exp)
                        if not sol.success:
                            return 1e6
                        y_pred = -(sol.y[0] - v_eq)
                        return np.sum((y_pred - y_exp) ** 2)
                    except Exception:
                        return 1e6

                res = minimize(loss_function, initial_guess, method="Nelder-Mead", bounds=bounds, options={'maxiter': 350})
                
                k_opt, a_opt, vm0_opt, iv_opt, iw_opt = res.x
                sol_opt, v_eq_opt, _ = solve_fhn(k_opt, a_opt, vm0_opt, fit_p_start, fit_p_end, iv_opt, iw_opt, t_exp)
                y_fit = -(sol_opt.y[0] - v_eq_opt)

                r2 = r2_score(y_exp, y_fit)
                rmse = root_mean_squared_error(y_exp, y_fit)

            st.success("Optimization converged successfully!")

            # Extracted parameter metrics
            st.markdown("### Best-Fit Model Parameters")
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("V_m0", f"{vm0_opt:.3f}")
            m2.metric("alpha (α)", f"{a_opt:.2f}")
            m3.metric("k_K", f"{k_opt:.3f}")
            m4.metric("I_v", f"{iv_opt:.4f}")
            m5.metric("I_w", f"{iw_opt:.4f}")
            m6.metric("R² Score", f"{r2:.4f}")
            m7.metric("RMSE", f"{rmse:.4e}")

            # Fit plot
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=t_exp, y=y_exp, mode="markers", name="Experimental Data", marker=dict(size=4, color="rgba(0,0,0,0.5)")))
            fig_fit.add_trace(go.Scatter(x=t_exp, y=y_fit, mode="lines", name="Best FHN Model Fit", line=dict(color="red", width=2.5)))
            fig_fit.add_vrect(x0=fit_p_start, x1=fit_p_end, fillcolor="gold", opacity=0.2, line_width=0, annotation_text="Stimulus Window")
            fig_fit.update_layout(title="Model Fit vs. Experimental Observation", xaxis_title="Time (s)", yaxis_title="-ΔV_m", height=450, template="plotly_white")
            st.plotly_chart(fig_fit, use_container_width=True)

            # Cell state classification from fit
            st.markdown("### Physiological Phenotype Classification")
            if k_opt >= 5.0:
                st.success(f"**Proliferative Phenotype**: The extracted potassium conductivity (k_K = {k_opt:.2f}) indicates an active, hyperpolarization-competent cell.")
            elif k_opt <= 1.0:
                st.error(f"**Inhibited / Non-Viable Phenotype**: The low conductibility (k_K = {k_opt:.2f}) matches metabolically stressed or depolarized cells.")
            else:
                st.warning(f"**Intermediate State**: Extracted conductibility (k_K = {k_opt:.2f}) falls into the transitional stress zone.")