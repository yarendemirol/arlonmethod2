import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="ARLON Method with Fixed Data")

# --- FIXED DATA FROM YOUR FIRST CODE BLOCK ---
# v_j and p_j values from Table S3 (for calculating criterion weights from MPSI)
v_j_values = {
    'C1': 0.3407, 'C2': 0.5468, 'C3': 0.6627, 'C4': 0.4171, 'C5': 0.0701,
    'C6': 0.3469, 'C7': 0.3658, 'C8': 0.1794, 'C9': 0.3065, 'C10': 0.1969
}
p_j_values = {
    'C1': 4.4499, 'C2': 4.3343, 'C3': 4.0669, 'C4': 2.3679, 'C5': 1.0406,
    'C6': 4.6405, 'C7': 2.3722, 'C8': 1.2492, 'C9': 2.1399, 'C10': 1.4559
}

# Normalized decision matrix using Heron mean from Table S4 (example data)
normalized_heron_mean_data = {
    'Countries': ['Algeria', 'Angola', 'Argentina', 'Australia', 'Bahrain', 'Bangladesh', 'Belgium', 'Benin', 'Brazil', 'Bulgaria'],
    'C1': [0.0113, 0.0121, 0.0119, 0.0140, 0.0090, 0.0097, 0.0175, 0.0127, 0.0138, 0.0081],
    'C2': [0.0117, 0.0121, 0.0131, 0.0136, 0.0126, 0.0127, 0.0146, 0.0105, 0.0141, 0.0131],
    'C3': [0.0127, 0.0113, 0.0124, 0.0142, 0.0122, 0.0130, 0.0133, 0.0116, 0.0138, 0.0133],
    'C4': [0.2936, 0.2936, 0.3016, 0.3000, 0.3000, 0.2946, 0.3033, 0.3043, 0.3102, 0.3064],
    'C5': [0.2972, 0.2974, 0.3083, 0.3115, 0.3066, 0.3026, 0.3108, 0.3014, 0.3072, 0.3087],
    'C6': [0.3058, 0.3043, 0.3026, 0.3072, 0.3055, 0.3058, 0.3063, 0.3073, 0.3030, 0.3054],
    'C7': [0.3041, 0.3064, 0.3053, 0.3079, 0.3065, 0.3060, 0.3055, 0.3052, 0.3064, 0.3059],
    'C8': [0.3038, 0.3071, 0.3051, 0.3078, 0.3064, 0.3059, 0.3057, 0.3049, 0.3062, 0.3057],
    'C9': [0.3076, 0.3050, 0.3059, 0.3066, 0.3068, 0.3091, 0.3058, 0.3042, 0.3053, 0.3056],
    'C10': [0.3075, 0.3049, 0.3057, 0.3065, 0.3067, 0.3092, 0.3058, 0.3040, 0.3052, 0.3056]
}
# IMPORTANT: Your first code block uses 'normalized_heron_mean_df' as the input to the ARLON
# weighted aggregation step. This means the normalization step (2-2) of ARLON is skipped in
# this specific scenario, as the data is *already* normalized by Heron Mean.
# If you intend to perform the full ARLON normalization (logarithmic normalization + Heron Mean)
# on *raw* data, you would need to provide a raw decision matrix here instead.
# For this request, we'll use 'normalized_heron_mean_df' as the starting point for ARLON's
# weighted aggregation as per your provided first code block.
initial_decision_matrix_fixed = pd.DataFrame(normalized_heron_mean_data).set_index('Countries')

# Criterion types
# C1, C2, C3 are benefit criteria, others are cost criteria
benefit_criteria_fixed = ['C1', 'C2', 'C3']
cost_criteria_fixed = ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']

# --- MPSI Weighting (Step 1 of your original script) ---
sum_pj = sum(p_j_values.values())
criterion_weights_fixed = {k: v / sum_pj for k, v in p_j_values.items()}
criterion_weights_series_fixed = pd.Series(criterion_weights_fixed)

# --- ARLON METHOD FUNCTION (Adapted to take already normalized data) ---
def run_arlon_method_with_fixed_data(normalized_heron_mean_df, criterion_weights, benefit_criteria, cost_criteria, zeta_value=0.5):
    """
    Executes the ARLON MCDM method starting from *already Heron-mean normalized* data.
    This skips the initial logarithmic normalization steps (2-2) of the full ARLON method,
    as the input is assumed to be `ℏnorm_ij` from the paper.

    Args:
        normalized_heron_mean_df (pd.DataFrame): The Heron-mean normalized decision matrix (ℏnorm_ij).
        criterion_weights (pd.Series): Normalized weights for each criterion.
        benefit_criteria (list): List of column names (criteria) that are of 'Benefit' type.
        cost_criteria (list): List of column names (criteria) that are of 'Cost' type.
        zeta_value (float): Zeta (ζ) is used in the full ARLON for combining two logarithmic normalizations.
                            Since the input is already Heron mean, this parameter is not directly
                            applied in the *provided fixed data's ARLON calculation path*.
                            However, we keep it for consistency if one were to re-implement full ARLON.

    Returns:
        tuple: A tuple containing:
            - final_ranking (pd.DataFrame): DataFrame with final scores and rankings.
            - weighted_aggregated_normalized_df (pd.DataFrame): Intermediate matrix after applying weights.
            - cost_benefit_sums (pd.DataFrame): Intermediate DataFrame with summed cost and benefit values.
    """
    # In your original script, `normalized_heron_mean_df` is directly used for weighted aggregation.
    # This implies that the 'normalization' (steps 2-2) is considered already done for this input.
    # So, we'll use `normalized_heron_mean_df` as `aggregated_normalized_df` for the next step.
    aggregated_normalized_df_for_weighting = normalized_heron_mean_df.copy()

    ## ARLON Method Step 2-3: Weighted Aggregation
    # Formula: ̂ℏ_ij = (w_j * ℏnorm_ij)
    weighted_aggregated_normalized_df = aggregated_normalized_df_for_weighting.copy()
    for col in aggregated_normalized_df_for_weighting.columns:
        if col in criterion_weights.index:
            weighted_aggregated_normalized_df[col] = aggregated_normalized_df_for_weighting[col] * criterion_weights[col]
        else:
            # Handle cases where a criterion in the matrix might not have a weight (should ideally not happen with fixed data)
            st.warning(f"Kriter '{col}' için ağırlık bulunamadı. Bu kriter ağırlıklandırılmayacaktır.")
            weighted_aggregated_normalized_df[col] = aggregated_normalized_df_for_weighting[col]


    ## ARLON Method Step 2-4: Separate Summation of Cost and Benefit Criteria
    # Formula: ℂ_i = ∑_{j∈C} ̂ℏ_ij (Sum of Cost Criteria)
    # Formula: B_i = ∑_{j∈B} ̂ℏ_ij (Sum of Benefit Criteria)

    current_cost_criteria = [c for c in cost_criteria if c in weighted_aggregated_normalized_df.columns]
    current_benefit_criteria = [c for c in benefit_criteria if c in weighted_aggregated_normalized_df.columns]

    cost_sums = weighted_aggregated_normalized_df[current_cost_criteria].sum(axis=1) if current_cost_criteria else pd.Series(0.0, index=normalized_heron_mean_df.index)
    benefit_sums = weighted_aggregated_normalized_df[current_benefit_criteria].sum(axis=1) if current_benefit_criteria else pd.Series(0.0, index=normalized_heron_mean_df.index)

    result_df = pd.DataFrame({
        'Cost_Sum (ℂ_i)': cost_sums,
        'Benefit_Sum (B_i)': benefit_sums
    }, index=normalized_heron_mean_df.index)

    ## ARLON Method Step 2-5: Final Ranking of Alternatives
    # Formula: ℝ_i = B_i^ψ + ℂ_i^(1-ψ)
    # ψ (psi) value: (number of benefit criteria) / (total number of criteria)
    total_criteria_count = len(current_benefit_criteria) + len(current_cost_criteria)
    if total_criteria_count == 0:
        psi_value = 0.5 # Default if no criteria defined
        st.warning("Hiç kriter tanımlanmadığı için psi değeri varsayılan olarak 0.5 olarak ayarlandı.")
    else:
        psi_value = len(current_benefit_criteria) / total_criteria_count

    final_ranking_scores = []
    for idx in result_df.index:
        b_i = result_df.loc[idx, 'Benefit_Sum (B_i)']
        c_i = result_df.loc[idx, 'Cost_Sum (ℂ_i)']

        # Ensure that base for power calculation is non-negative
        term_benefit = b_i ** psi_value if b_i >= 0 else 0
        term_cost = c_i ** (1 - psi_value) if c_i >= 0 else 0

        final_ranking_scores.append(term_benefit + term_cost)

    result_df['Final_Ranking_Score (ℝ_i)'] = final_ranking_scores

    # Sort alternatives by final ranking score (higher score means better performance)
    final_ranking = result_df.sort_values(by='Final_Ranking_Score (ℝ_i)', ascending=False)
    final_ranking['Ranking'] = np.arange(1, len(final_ranking) + 1)

    return final_ranking[['Final_Ranking_Score (ℝ_i)', 'Ranking']], weighted_aggregated_normalized_df, result_df[['Cost_Sum (ℂ_i)', 'Benefit_Sum (B_i)']]

# --- STREAMLIT APPLICATION START ---
st.title("ARLON Çok Kriterli Karar Verme Yöntemi (Sabit Veri)")
st.markdown("""
Bu uygulama, sağladığınız **sabit veri setini** kullanarak ARLON (Average Relative Logarithm Normalization) yöntemini uygular. 
Kriter ağırlıkları ve başlangıç matrisi sağlanan veriden doğrudan alınmıştır.
""")

st.header("1. Sabit Veri Seti")

# Display Fixed Criterion Weights
st.subheader("Kriter Ağırlıkları (MPSI'dan Hesaplandı)")
st.dataframe(pd.DataFrame({'Kriter': criterion_weights_series_fixed.index, 'Ağırlık': criterion_weights_series_fixed.values}).set_index('Kriter').T, use_container_width=True)

# Display Fixed Normalized Heron Mean Decision Matrix
st.subheader("Heron Ortalaması ile Normalize Edilmiş Karar Matrisi")
st.dataframe(initial_decision_matrix_fixed, use_container_width=True)

# Display Criterion Types
st.subheader("Kriter Tipleri")
st.write(f"**Fayda Kriterleri:** {', '.join(benefit_criteria_fixed)}")
st.write(f"**Maliyet Kriterleri:** {', '.join(cost_criteria_fixed)}")

st.markdown("---")

# --- RUN ARLON METHOD ---
st.header("2. ARLON Analiz Sonuçları")

# ARLON Method with fixed data (no dynamic zeta slider as it's not applied to already normalized data in this setup)
# For the purpose of displaying intermediate steps, we'll slightly adjust the function's return for clarity.
# Since your initial script directly takes `normalized_heron_mean_df` and applies weights,
# we are simulating step 2-3 onwards from the ARLON method.
try:
    final_ranking, weighted_agg_norm_df, cost_benefit_sums = run_arlon_method_with_fixed_data(
        initial_decision_matrix_fixed, criterion_weights_series_fixed, benefit_criteria_fixed, cost_criteria_fixed
    )

    st.subheader("Alternatiflerin Nihai ARLON Sıralaması")
    st.dataframe(final_ranking.rename(columns={'Final_Ranking_Score (ℝ_i)': 'ARLON Skoru', 'Ranking': 'Sıralama'}), use_container_width=True)

    st.markdown("---")

    st.subheader("Ara Sonuçlar")

    # The fixed data `normalized_heron_mean_df` is already the output of Heron Mean aggregation,
    # so we'll display it here as the base for weighted aggregation.
    st.markdown("#### Heron Ortalaması Agregasyon Matrisi (Başlangıç Verisi)")
    st.markdown("Bu, analize başlanan, Heron ortalaması ile normalize edilmiş başlangıç matrisidir (̂ℏnorm_ij).")
    st.dataframe(initial_decision_matrix_fixed, use_container_width=True)


    st.markdown("#### Ağırlıklı Normalleştirilmiş Karar Matrisi (̂ℏ_ij)")
    st.markdown("Bu tablo, Heron ortalaması ile birleştirilmiş normalize edilmiş değerlerin, kriter ağırlıkları ile çarpılması sonucunda elde edilen ̂ℏ_ij değerlerini göstermektedir.")
    st.dataframe(weighted_agg_norm_df, use_container_width=True)

    st.markdown("#### Fayda ve Maliyet Toplamları ($B_i$ ve $C_i$)")
    st.markdown("Bu tablo, alternatifler için fayda ($B_i$) ve maliyet ($C_i$) kriterlerinin ağırlıklı toplamlarını göstermektedir.")
    st.dataframe(cost_benefit_sums, use_container_width=True)

    st.markdown("---")

    st.subheader("Sıralama Özeti")
    st.write("**:green[En iyi 3 ülke:]**")
    st.dataframe(final_ranking.head(3), use_container_width=True)
    st.write("**:red[En kötü 3 ülke:]**")
    st.dataframe(final_ranking.tail(3), use_container_width=True)

    # Sensitivity analysis with Zeta is not applicable here as the input is already normalized with Heron Mean.
    # If you had raw data, you'd perform the full ARLON which involves Zeta.
    st.info("Bu uygulama, sağlanan sabit veri seti için ARLON yöntemini kullanmaktadır. Bu veri seti zaten Heron ortalaması ile normalize edildiğinden, Zeta (ζ) değeri için hassasiyet analizi bu bağlamda uygulanmamaktadır.")

except Exception as e:
    st.error(f"ARLON analizi sırasında bir hata oluştu: {e}. Lütfen sağlanan sabit verileri kontrol edin.")
