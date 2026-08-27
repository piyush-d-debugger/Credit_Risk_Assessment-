import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# Loan-purpose capacity policy
# ================================================================
# These are policy multipliers used ONLY for the borrower-capacity
# calculation. They are deliberately separate from the ML model,
# because the existing application-time model uses loan_amnt and
# loan_percent_income and is therefore better suited to pricing
# the specific requested loan.
#
# The multiplier changes the borrowing-capacity range by purpose:
# - PERSONAL: smaller typical borrowing capacity
# - EDUCATION: moderate capacity
# - MEDICAL: moderate/high capacity
# - VENTURE: highest potential capacity
# - HOMEIMPROVEMENT: high capacity
# - DEBTCONSOLIDATION: moderate capacity
PURPOSE_MULTIPLIER = {
    "PERSONAL": 0.75,
    "EDUCATION": 1.00,
    "MEDICAL": 1.05,
    "VENTURE": 1.35,
    "HOMEIMPROVEMENT": 1.20,
    "DEBTCONSOLIDATION": 0.95,
}


# Maximum income-to-loan-capacity ratio before profile adjustments.
BASE_INCOME_CAP = 0.50

# Hard ceiling: the training data never contained loans above
# $35,000, so recommending anything beyond that would be
# extrapolating past what the model has ever seen priced.
ABSOLUTE_MAX_LOAN = 35000


class Pipeline:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        with open(
            base_dir / "model" / "model_application_time.pkl", "rb"
        ) as f:
            self.model = pickle.load(f)

        with open(
            base_dir / "parameter" / "scaler_application_time.pkl", "rb"
        ) as f:
            self.scaler = pickle.load(f)

        with open(
            base_dir / "parameter" / "train_columns_application_time.pkl",
            "rb",
        ) as f:
            self.train_columns = pickle.load(f)

        with open(
            base_dir / "parameter" / "grade_mapping.pkl", "rb"
        ) as f:
            self.grade_mapping = pickle.load(f)

        # Build probability -> interest-rate interpolation anchors.
        avg_prob_by_grade = self.grade_mapping["avg_prob_by_grade"]
        avg_rate_by_grade = self.grade_mapping["avg_rate_by_grade"]

        pairs = sorted(
            avg_prob_by_grade.items(),
            key=lambda x: x[1],
        )

        self._sorted_grades = [
            grade for grade, _ in pairs
        ]

        self._sorted_probs = np.array(
            [prob for _, prob in pairs],
            dtype=float,
        )

        self._sorted_rates = np.array(
            [
                avg_rate_by_grade[grade]
                for grade in self._sorted_grades
            ],
            dtype=float,
        )

    # ================================================================
    # Existing ML model preprocessing
    # ================================================================
    def transform(self, input_data):
        data = input_data.copy()

        numerical_cols = self.scaler.feature_names_in_

        df_numerical = data[numerical_cols]

        scaled_numerical = self.scaler.transform(
            df_numerical
        )

        df_scaled_numerical = pd.DataFrame(
            scaled_numerical,
            columns=numerical_cols,
            index=data.index,
        )

        cat_cols = data.select_dtypes(
            include=["object", "str"]
        ).columns

        df_categorical = pd.get_dummies(
            data[cat_cols]
        )

        df_prepared = pd.concat(
            [
                df_scaled_numerical,
                df_categorical,
            ],
            axis=1,
        )

        # Match exactly the columns used during training.
        return df_prepared.reindex(
            columns=self.train_columns,
            fill_value=0,
        )

    def predict(self, input_data):
        transformed_data = self.transform(
            input_data
        )

        return self.model.predict_proba(
            transformed_data
        )

    # ================================================================
    # Request-specific ML risk
    # ================================================================
    def _build_application_row(
        self,
        applicant,
        loan_amount,
    ):
        income = applicant["person_income"]

        return {
            "person_age": applicant["person_age"],
            "person_income": income,
            "person_home_ownership": applicant[
                "person_home_ownership"
            ],
            "person_emp_length": applicant[
                "person_emp_length"
            ],
            "loan_intent": applicant["loan_intent"],
            "loan_amnt": loan_amount,
            "loan_percent_income": (
                loan_amount / income
                if income > 0
                else 0
            ),
            "cb_person_default_on_file": applicant[
                "cb_person_default_on_file"
            ],
            "cb_person_cred_hist_length": applicant[
                "cb_person_cred_hist_length"
            ],
        }

    def _predict_request_risk(
        self,
        applicant,
        loan_amount,
    ):
        row = self._build_application_row(
            applicant,
            loan_amount,
        )

        df_row = pd.DataFrame([row])

        return float(
            self.predict(df_row)[0, 1]
        )

    # ================================================================
    # Grade / interest-rate helpers
    # ================================================================
    def _prob_to_grade(self, probability):
        grades = self.grade_mapping["grades_sorted"]
        thresholds = self.grade_mapping["thresholds"]

        for i, threshold in enumerate(thresholds):
            if probability <= threshold:
                return grades[i]

        return grades[-1]

    def _interpolate_rate(self, probability):
        return float(
            np.interp(
                probability,
                self._sorted_probs,
                self._sorted_rates,
            )
        )

    # ================================================================
    # Profile-based borrowing capacity
    # ================================================================
    def _calculate_profile_score(self, applicant):
        """
        Calculate borrower strength WITHOUT using loan_amnt.

        This is important:
        changing the requested amount must not change the applicant's
        underlying borrowing capacity.
        """

        income = float(
            applicant["person_income"]
        )

        employment = float(
            applicant["person_emp_length"]
        )

        credit_history = float(
            applicant["cb_person_cred_hist_length"]
        )

        age = float(
            applicant["person_age"]
        )

        # ------------------------------------------------------------
        # Income
        # ------------------------------------------------------------
        # Income is the strongest capacity factor.
        # $100k+ receives the maximum income score.
        income_score = min(
            income / 100000.0,
            1.0,
        )

        # ------------------------------------------------------------
        # Employment stability
        # ------------------------------------------------------------
        # 10+ years receives the maximum contribution.
        employment_score = min(
            max(employment, 0.0) / 10.0,
            1.0,
        )

        # ------------------------------------------------------------
        # Credit history
        # ------------------------------------------------------------
        # 10+ years receives the maximum contribution.
        credit_history_score = min(
            max(credit_history, 0.0) / 10.0,
            1.0,
        )

        # ------------------------------------------------------------
        # Age
        # ------------------------------------------------------------
        # Kept deliberately small so age does not dominate.
        age_score = min(
            max(age - 18.0, 0.0) / 32.0,
            1.0,
        )

        # ------------------------------------------------------------
        # Home ownership
        # ------------------------------------------------------------
        housing_score = {
            "MORTGAGE": 1.00,
            "OWN": 0.90,
            "RENT": 0.55,
            "OTHER": 0.40,
        }.get(
            applicant["person_home_ownership"],
            0.40,
        )

        # ------------------------------------------------------------
        # Previous default
        # ------------------------------------------------------------
        # A previous default is a major negative factor.
        default_score = (
            0.0
            if applicant["cb_person_default_on_file"] == "Y"
            else 1.0
        )

        # ------------------------------------------------------------
        # Overall profile score
        # ------------------------------------------------------------
        score = (
            0.30 * income_score
            + 0.20 * employment_score
            + 0.20 * credit_history_score
            + 0.05 * age_score
            + 0.10 * housing_score
            + 0.15 * default_score
        )

        return float(
            np.clip(score, 0.0, 1.0)
        )

    def _calculate_max_approved_amount(
        self,
        applicant,
    ):
        """
        Calculate maximum borrowing capacity.

        IMPORTANT:
        loan_amnt is intentionally NOT used here.

        Therefore:
            request = $10,000 -> same capacity
            request = $20,000 -> same capacity
            request = $30,000 -> same capacity

        Only the applicant's profile and loan purpose determine
        this capacity, capped at ABSOLUTE_MAX_LOAN since the model
        was never trained on loans beyond that amount.
        """

        income = float(
            applicant["person_income"]
        )

        purpose = applicant["loan_intent"]

        if purpose not in PURPOSE_MULTIPLIER:
            purpose_multiplier = 1.0
        else:
            purpose_multiplier = PURPOSE_MULTIPLIER[
                purpose
            ]

        profile_score = self._calculate_profile_score(
            applicant
        )

        # A stronger profile gets a larger share of annual income.
        #
        # Profile score 0.0 -> 50% of base income capacity
        # Profile score 1.0 -> 100% of base income capacity
        profile_multiplier = (
            0.50
            + (0.50 * profile_score)
        )

        capacity = (
            income
            * BASE_INCOME_CAP
            * profile_multiplier
            * purpose_multiplier
        )

        # Practical lower floor. This prevents tiny/negative-looking
        # capacity values for low-income profiles.
        minimum_capacity = 1000.0

        # Cap at ABSOLUTE_MAX_LOAN: the model was only ever trained on
        # loans between $500-$35,000, so approving amounts beyond
        # that would be pure extrapolation with no evidence behind it.
        maximum = min(
            ABSOLUTE_MAX_LOAN,
            max(minimum_capacity, capacity),
        )

        return {
            "max_approved_amount": round(
                float(maximum),
                2,
            ),
            "profile_score": round(
                profile_score,
                4,
            ),
            "profile_multiplier": round(
                profile_multiplier,
                4,
            ),
            "purpose_multiplier": round(
                purpose_multiplier,
                4,
            ),
            "income_based_capacity": round(
                float(capacity),
                2,
            ),
        }

    # ================================================================
    # Public assessment
    # ================================================================
    def assess_applicant(self, applicant):
        """
        Perform two independent assessments.

        PASS 1:
            Profile + loan purpose -> maximum borrowing capacity.

        PASS 2:
            Profile + actual requested amount -> default risk + rate.

        The requested amount can therefore affect the interest rate,
        but cannot change the applicant's underlying maximum capacity.
        """

        requested_amount = float(
            applicant["loan_amnt"]
        )

        income = float(
            applicant["person_income"]
        )

        if income <= 0:
            raise ValueError(
                "Annual income must be greater than $0."
            )

        if requested_amount < 500:
            raise ValueError(
                "Requested loan amount must be at least $500."
            )

        # ------------------------------------------------------------
        # PASS 1: Stable borrower capacity
        # ------------------------------------------------------------
        capacity = (
            self._calculate_max_approved_amount(
                applicant
            )
        )

        maximum = capacity[
            "max_approved_amount"
        ]

        # ------------------------------------------------------------
        # PASS 2: Request-specific risk/pricing
        # ------------------------------------------------------------
        probability = (
            self._predict_request_risk(
                applicant,
                requested_amount,
            )
        )

        grade = self._prob_to_grade(
            probability
        )

        interest_rate = self._interpolate_rate(
            probability
        )

        approved = (
            requested_amount <= maximum
        )

        return {
            # Request-specific outputs
            "probability_of_default": probability,
            "assigned_grade": grade,
            "interest_rate": round(
                interest_rate,
                2,
            ),
            "requested_amount": requested_amount,

            # Stable profile/purpose capacity
            "max_approved_amount": maximum,
            "profile_score": capacity[
                "profile_score"
            ],
            "profile_multiplier": capacity[
                "profile_multiplier"
            ],
            "purpose_multiplier": capacity[
                "purpose_multiplier"
            ],
            "income_based_capacity": capacity[
                "income_based_capacity"
            ],

            # Final decision
            "approved": approved,
        }