import streamlit as st

from pipeline import Pipeline

st.set_page_config(page_title="Credit Risk App", layout="centered")

st.title("Credit Risk App")
st.write(
    "Enter your profile and requested loan amount below. "
    "We'll estimate your default risk and let you know your approved "
    "interest rate and maximum loan amount."
)

st.markdown("---")

with st.form("applicant_form"):
    st.subheader("Your profile")

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input(
            "Annual income ($)", min_value=0, value=50000, step=1000
        )
        person_home_ownership = st.selectbox(
            "Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
        person_emp_length = st.number_input(
            "Years employed", min_value=0.0, max_value=60.0, value=5.0, step=0.5
        )

    with col2:
        cb_person_cred_hist_length = st.number_input(
            "Credit history length (years)", min_value=0, max_value=40, value=5
        )
        cb_person_default_on_file = st.selectbox(
            "Prior default on record?", ["N", "Y"]
        )
        loan_intent = st.selectbox(
            "Loan purpose",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )

    st.subheader("Requested loan")
    loan_amnt = st.number_input(
        "Requested loan amount ($)", min_value=500, max_value=100000, value=10000, step=500
    )

    submitted = st.form_submit_button("Assess my risk", type="primary")

if submitted:
    applicant = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
    }

    try:
        pipeline = Pipeline()
        result = pipeline.assess_applicant(applicant)

        st.markdown("---")
        st.subheader("Your loan offer")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Interest rate", f"{result['interest_rate']:.2f}%")
        with col2:
            st.metric("Maximum approved amount", f"${result['max_approved_amount']:,.2f}")

        if result['approved']:
            st.success(
                f"Your requested amount of \\${result['requested_amount']:,.2f} "
                f"is within your approved limit."
            )
        else:
            st.warning(
                f"Your requested amount of \\${result['requested_amount']:,.2f} "
                f"exceeds your approved limit of \\${result['max_approved_amount']:,.2f}. "
                f"Consider requesting a lower amount."
            )

        with st.expander("For lending manager use only"):
            st.write(f"**Risk grade:** {result['assigned_grade']}")
            st.write(f"**Predicted probability of default:** {result['probability_of_default']*100:.1f}%")

    except Exception as e:
        st.error(f"Something went wrong while processing your application: {e}")

st.markdown("---")
st.caption(
    "Note: risk grade and interest rate are estimated using a model that "
    "learns typical grade/rate patterns from historical data, since a new "
    "applicant doesn't have these assigned yet. "
    "Full project documentation: [GitHub repo](https://github.com/piyush-d-debugger)"
)