"""Streamlit app for five-year digital-twin CLV simulation."""
from __future__ import annotations

import asyncio
import hashlib
import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DEFAULT_USERS_FILE = APP_DIR / "users.xlsx"
MODEL_NAME = "gemini-3.1-flash-lite"
CAC_PER_CUSTOMER = 100.0
SIMULATION_VERSION = 2
YEARS = range(1, 6)


class RetentionResponse(BaseModel):
    year_1_probability: int = Field(ge=0, le=100, description="Calibrated probability of returning in year 1")
    year_2_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 2")
    year_3_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 3")
    year_4_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 4")
    year_5_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 5")
    reason: str = Field(description="One concise sentence explaining the decisions")


def clean_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@st.cache_data(show_spinner=False)
def load_default_profiles() -> pd.DataFrame:
    return pd.read_excel(DEFAULT_USERS_FILE)


def load_profiles(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_profiles()
    if Path(uploaded_file.name).suffix.lower() == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def matching_column(dataframe: pd.DataFrame, wanted: str) -> str | None:
    names = {str(column).strip().casefold(): str(column) for column in dataframe.columns}
    return names.get(wanted.casefold())


def age_group_for(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age):
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return text or None
        age = float(numbers[0])
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def filter_profiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Choose customer profiles")
    fields = (
        ("Age group", matching_column(dataframe, "Age"), age_group_for),
        ("Location", matching_column(dataframe, "Location"), clean_value),
        ("Gender", matching_column(dataframe, "Gender"), clean_value),
    )
    mask = pd.Series(True, index=dataframe.index)
    for container, (label, source, transform) in zip(st.columns(3), fields):
        if not source:
            container.caption(f"{label} filter unavailable.")
            continue
        values = dataframe[source].map(transform)
        options = sorted(set(values.dropna()), key=lambda item: str(item).casefold())
        selected = container.multiselect(label, options, placeholder=f"All {label.lower()}s")
        if selected:
            mask &= values.isin(selected)

    matches = dataframe.loc[mask].copy()
    if matches.empty:
        st.warning("No profiles match the selected filters.")
        return matches
    count = int(st.number_input(
        "Number of profiles to simulate", 1, len(matches), len(matches), 1,
        help="A reproducible random sample is used when fewer profiles are selected.",
    ))
    if count < len(matches):
        matches = matches.sample(count, random_state=42).sort_index()
    return matches.reset_index(drop=True)


def configured_api_key() -> str:
    """The API configuration stays fixed and outside the user interface."""
    try:
        return str(st.secrets["GOOGLE_API_KEY"])
    except (KeyError, FileNotFoundError):
        return ""


def make_prompt(firm: str, description: str, persona: str, profile: dict[str, str]) -> str:
    profile_text = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    return f"""Act as a digital twin of a potential customer.

Firm: {firm}
Firm/service description: {description}
Target customer persona supplied by the researcher:
{persona}

Individual customer profile:
{profile_text}

Estimate this customer's probability (an integer from 0 to 100) of using the service again in
each of the next five years. Years 2-5 are conditional probabilities: estimate the chance of
returning that year if the customer was still active in the prior year.

Be realistically calibrated, not promotional. A generally useful service does not imply 100%
retention. Account for ordinary churn, changing needs, price sensitivity, competing services,
relocation, and declining relevance when consistent with the supplied profile. Reserve values
above 90 for unusually strong, explicit evidence of durable loyalty, and use the full range to
distinguish profiles. Do not invent discounts, product changes, or personal facts. Return five
probabilities and one concise reason."""


def stable_draw(profile_id: int, year: int, firm: str) -> float:
    """Return a reproducible pseudo-random percentile for a profile/year."""
    key = f"{firm.casefold().strip()}|{profile_id}|{year}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return integer / (2**64 - 1) * 100


async def query_profile(profile_id: int, firm: str, prompt: str, llm, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        for attempt in range(5):
            try:
                response = await llm.ainvoke(prompt)
                retained = True
                decisions = {}
                for year in YEARS:
                    probability = int(getattr(response, f"year_{year}_probability"))
                    decisions[f"Year {year} probability"] = probability / 100
                    retained = retained and stable_draw(profile_id, year, firm) < probability
                    decisions[f"Year {year} return"] = retained
                return {"Profile ID": profile_id, **decisions, "Reason": response.reason, "Status": "Success"}
            except Exception as exc:
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                else:
                    return {"Profile ID": profile_id, "Status": f"Error: {exc}"}
    return {"Profile ID": profile_id, "Status": "Unknown error"}


async def run_all(tasks: list, progress) -> list[dict]:
    results = []
    for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await task)
        progress.progress(completed / len(tasks), text=f"{completed} of {len(tasks)} profiles simulated")
    return results


def simulate_retention(profiles_df: pd.DataFrame, firm: str, description: str, persona: str, api_key: str) -> pd.DataFrame:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0.5, google_api_key=api_key
    ).with_structured_output(RetentionResponse)
    semaphore = asyncio.Semaphore(40)
    profiles = {}
    tasks = []
    for profile_id, (_, row) in enumerate(profiles_df.iterrows(), start=1):
        profile = {str(column): value for column, raw in row.items() if (value := clean_value(raw)) is not None}
        profiles[profile_id] = profile
        tasks.append(query_profile(profile_id, firm, make_prompt(firm, description, persona, profile), llm, semaphore))
    progress = st.progress(0, text="Preparing profile simulations...")
    responses = asyncio.run(run_all(tasks, progress))
    progress.empty()
    records = [{**profiles[item["Profile ID"]], **item} for item in responses]
    return pd.DataFrame(records).sort_values("Profile ID")


def calculate_clv(results: pd.DataFrame, monthly_sales: float, margin_percent: float) -> pd.DataFrame:
    successful = results.loc[results["Status"] == "Success"]
    annual_sales = monthly_sales * 12
    margin = margin_percent / 100
    return pd.DataFrame([
        {
            "Year": f"Year {year}",
            "Returning profiles": (retained := int(successful[f"Year {year} return"].sum())),
            "Retention rate": retained / len(successful),
            "Annual sales per profile": annual_sales,
            "Margin": margin,
            "Annual CLV contribution": annual_sales * retained * margin,
        }
        for year in YEARS
    ])


def clv_figure(summary: pd.DataFrame, firm: str):
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(summary["Year"], summary["Annual CLV contribution"], color="#2563eb")
    axis.set_title(f"Five-year CLV contribution — {firm}", pad=14)
    axis.set_ylabel("CLV contribution ($)")
    axis.grid(axis="y", alpha=0.2)
    axis.bar_label(bars, labels=[f"${value:,.0f}" for value in summary["Annual CLV contribution"]], padding=3)
    figure.tight_layout()
    return figure


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_") or "firm"


st.set_page_config(page_title="Customer Lifetime Value Simulator", page_icon="📈", layout="wide")
st.title("Customer Lifetime Value Simulator")
st.caption("Simulate repeat use across customer profiles and estimate year-by-year CLV over five years.")

api_key = configured_api_key()
if not api_key:
    st.error("Gemini API key is not configured. Add GOOGLE_API_KEY to the app's Streamlit secrets.")
    st.stop()

st.subheader("Firm and customer assumptions")
firm_name = st.text_input("Firm name", placeholder="e.g., Acme Fitness")
firm_description = st.text_area(
    "Firm and service description",
    placeholder="Describe the firm, its offering, price, customer experience, and value.", height=150,
)
persona = st.text_area(
    "Target customer persona",
    placeholder="Describe needs, behaviors, motivations, pain points, budget, and usage context in detail.",
    height=240,
    help="This broad persona is considered together with each individual profile below.",
)
sales_col, margin_col = st.columns(2)
monthly_sales = sales_col.number_input(
    "Expected monthly sales per returning customer ($)", min_value=0.0, value=100.0, step=10.0
)
margin_percent = margin_col.number_input(
    "Profit margin (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0
)

st.markdown("### Fixed factors")
fixed1, fixed2, fixed3 = st.columns(3)
fixed1.metric("Customer acquisition cost", "$100 per customer")
fixed2.metric("Simulation horizon", "5 years")
fixed3.metric("Annual sales basis", f"${monthly_sales * 12:,.2f} per customer")
st.info("CAC is fixed at $100 per acquired customer. Monthly sales are annualized (× 12); no discount rate is applied.")

with st.expander("Customer profile source"):
    uploaded_file = st.file_uploader(
        "Optional profile file", type=["xlsx", "xls", "csv"],
        help="The built-in profiles are used by default. Every uploaded column becomes part of a profile.",
    )

try:
    all_profiles = load_profiles(uploaded_file).dropna(how="all").reset_index(drop=True)
    selected_profiles = filter_profiles(all_profiles)
    st.caption(f"{len(selected_profiles):,} of {len(all_profiles):,} profiles selected")
    st.dataframe(selected_profiles, use_container_width=True, height=260)
except Exception as exc:
    st.error(f"Could not load customer profiles: {exc}")
    st.stop()

if st.button("Simulate five-year CLV", type="primary", use_container_width=True):
    missing = [label for label, value in (
        ("firm name", firm_name), ("firm and service description", firm_description),
        ("target customer persona", persona),
    ) if not value.strip()]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    elif selected_profiles.empty:
        st.error("Select at least one customer profile.")
    else:
        with st.spinner("Simulating repeat use across customer profiles..."):
            responses = simulate_retention(selected_profiles, firm_name, firm_description, persona, api_key)
        st.session_state["clv_run"] = {
            "responses": responses, "firm": firm_name,
            "monthly_sales": monthly_sales, "margin": margin_percent,
            "version": SIMULATION_VERSION,
        }

if "clv_run" in st.session_state and st.session_state["clv_run"].get("version") != SIMULATION_VERSION:
    del st.session_state["clv_run"]
    st.info("The retention method was updated. Run the simulation again to generate calibrated results.")

if "clv_run" in st.session_state:
    run = st.session_state["clv_run"]
    results = run["responses"]
    successful = results.loc[results["Status"] == "Success"]
    st.subheader("Five-year CLV results")
    if successful.empty:
        st.error("No simulations succeeded. Review the Status column.")
        st.dataframe(results, use_container_width=True)
    else:
        summary = calculate_clv(results, run["monthly_sales"], run["margin"])
        total_clv = float(summary["Annual CLV contribution"].sum())
        acquired = len(successful)
        average_clv = total_clv / acquired
        ratio = average_clv / CAC_PER_CUSTOMER

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("5-year cohort CLV", f"${total_clv:,.2f}")
        m2.metric("Average CLV per profile", f"${average_clv:,.2f}")
        m3.metric("Total acquisition cost", f"${acquired * CAC_PER_CUSTOMER:,.2f}")
        m4.metric("CLV : CAC", f"{ratio:.2f} : 1")
        st.caption("CLV:CAC = average five-year CLV per successfully simulated profile ÷ fixed $100 CAC.")

        chart_tab, annual_tab, profiles_tab = st.tabs(["CLV chart", "Year-by-year calculation", "Profile decisions"])
        with chart_tab:
            figure = clv_figure(summary, run["firm"])
            st.pyplot(figure, use_container_width=True)
            image_buffer = BytesIO()
            figure.savefig(image_buffer, format="png", dpi=200, bbox_inches="tight")
            plt.close(figure)
        with annual_tab:
            shown = summary.copy()
            shown["Retention rate"] = shown["Retention rate"].map(lambda value: f"{value:.1%}")
            shown["Annual sales per profile"] = shown["Annual sales per profile"].map(lambda value: f"${value:,.2f}")
            shown["Margin"] = shown["Margin"].map(lambda value: f"{value:.1%}")
            shown["Annual CLV contribution"] = shown["Annual CLV contribution"].map(lambda value: f"${value:,.2f}")
            st.dataframe(shown, use_container_width=True, hide_index=True)
            st.code("Annual CLV = (monthly sales × 12) × returning profiles × margin", language=None)
        with profiles_tab:
            st.dataframe(results, use_container_width=True, height=480)

        stem = safe_filename(run["firm"])
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download profile decisions", results.to_csv(index=False).encode("utf-8-sig"), f"{stem}_profile_retention.csv", "text/csv")
        d2.download_button("Download CLV calculation", summary.to_csv(index=False).encode("utf-8-sig"), f"{stem}_five_year_clv.csv", "text/csv")
        d3.download_button("Download chart", image_buffer.getvalue(), f"{stem}_five_year_clv.png", "image/png")
        failed = len(results) - len(successful)
        if failed:
            st.warning(f"{failed} failed simulation(s) were excluded from CLV calculations.")
