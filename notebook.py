# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.3.3",
#     "plotly==6.3.1",
#     "polars==1.34.0",
#     "pyarrow==21.0.0",
# ]
# ///

import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dataclasses import dataclass
    from typing import List, Tuple
    from decimal import Decimal
    return List, Tuple, dataclass, go, make_subplots, mo, np, pl


@app.cell
def _(dataclass):
    @dataclass
    class CurrentCompensation:
        base_salary: float
        annual_bonus: float
        annual_rsu: float
        company_401k_match: float
        mega_backdoor_available: bool
        expected_annual_raise_pct: float
        promotion_probability: float
        promotion_comp_increase_pct: float

        @property
        def total_cash_comp(self) -> float:
            return self.base_salary + self.annual_bonus + self.annual_rsu

        @property
        def total_401k_benefit(self) -> float:
            base = self.company_401k_match
            if self.mega_backdoor_available:
                base += 32500
            return base

    @dataclass
    class StartupOffer:
        base_salary: float
        annual_bonus: float
        number_of_shares: float
        current_stock_fmv: float
        strike_price: float
        current_valuation_millions: float
        company_401k_match: float
        mega_backdoor_available: bool

        @property
        def total_cash_comp(self) -> float:
            return self.base_salary + self.annual_bonus

        @property
        def equity_value_at_fmv(self) -> float:
            return self.number_of_shares * self.current_stock_fmv

        @property
        def initial_equity_percentage(self) -> float:
            fully_diluted_shares = (
                self.current_valuation_millions * 1_000_000
            ) / self.current_stock_fmv
            return (self.number_of_shares / fully_diluted_shares) * 100

        @property
        def exercise_cost(self) -> float:
            return self.number_of_shares * self.strike_price

        @property
        def spread_at_exercise(self) -> float:
            return self.number_of_shares * (self.current_stock_fmv - self.strike_price)

        @property
        def total_401k_benefit(self) -> float:
            base = self.company_401k_match
            if self.mega_backdoor_available:
                base += 32500
            return base

    @dataclass
    class ProbabilityFactors:
        base_probability: float
        founder_has_exit: bool
        top_tier_investors: bool
        series_b_or_later: bool
        strong_traction: bool

        @property
        def multiplier(self) -> float:
            m = 1.0
            if self.founder_has_exit:
                m *= 2.0
            if self.top_tier_investors:
                m *= 1.5
            if self.series_b_or_later:
                m *= 1.3
            if self.strong_traction:
                m *= 1.4
            return m

        @property
        def adjusted_probability(self) -> float:
            return min(self.base_probability * self.multiplier, 100.0)

    @dataclass
    class ExitScenario:
        valuation_multiple: float
        exit_probability: float  # Probability of this specific scenario (%)
        exit_valuation_millions: float
        equity_value: float
        after_tax_proceeds: float
        exit_npv: float
        total_npv: float
        net_vs_current: float
        probability_weighted_npv: (
            float  # Expected value given this scenario's probability
        )

    @dataclass
    class FinancialParameters:
        years_to_exit: int
        total_dilution_pct: float
        discount_rate: float = 0.08
        capital_gains_tax: float = 0.238
        retirement_horizon_years: int = 20
        retirement_growth_rate: float = 1.10

        @property
        def final_ownership_multiplier(self) -> float:
            return 1 - (self.total_dilution_pct / 100)
    return (
        CurrentCompensation,
        ExitScenario,
        FinancialParameters,
        ProbabilityFactors,
        StartupOffer,
    )


@app.cell
def _(go, np):
    def calculate_kahneman_multiplier(net_worth: float) -> float:
        if net_worth < 250_000:
            return 3.0
        elif net_worth < 1_000_000:
            return 3.0 - 0.5 * ((net_worth - 250_000) / 750_000)
        elif net_worth < 3_000_000:
            return 2.5 - 0.5 * ((net_worth - 1_000_000) / 2_000_000)
        elif net_worth < 10_000_000:
            return 2.0 - 0.5 * ((net_worth - 3_000_000) / 7_000_000)
        else:
            return max(1.2, 1.5 - 0.3 * np.log10(net_worth / 10_000_000))

    def create_kahneman_curve(net_worth: float, custom_multiplier: float = None):
        multiplier = (
            custom_multiplier
            if custom_multiplier is not None
            else calculate_kahneman_multiplier(net_worth)
        )

        # Simple bar chart showing the asymmetry
        fig = go.Figure()

        # Show that to offset a $100k loss, you need a much bigger gain
        loss_amount = 100
        gain_needed = loss_amount * multiplier

        fig.add_trace(
            go.Bar(
                x=["Loss", "Gain Needed<br>to Feel Equivalent"],
                y=[loss_amount, gain_needed],
                marker_color=["#FF4444", "#2E7D32"],
                text=[f"${loss_amount}k", f"${gain_needed:.0f}k"],
                textposition="outside",
                hovertemplate="$%{y:,.0f}k<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Loss Aversion Multiplier: {multiplier:.1f}x",
            yaxis_title="Dollar Amount (thousands)",
            template="plotly_white",
            height=500,
            showlegend=False,
        )

        return fig
    return calculate_kahneman_multiplier, create_kahneman_curve


@app.cell
def _(mo):
    mo.md("""# Startup Equity Compensation Analysis""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    current_net_worth = mo.ui.number(
        label="Current Net Worth ($)", value=500000, start=0, step=50000
    )

    use_custom_multiplier = mo.ui.checkbox(
        label="Use Custom Loss Aversion Multiplier", value=False
    )

    custom_multiplier_input = mo.ui.number(
        label="Custom Multiplier", value=2.5, start=1.0, stop=5.0, step=0.1
    )

    mo.vstack(
        [
            mo.md("""## Your Financial Profile

    Enter your current net worth to calculate your personal loss aversion multiplier.
    Optionally, override with a custom multiplier if you have a different risk tolerance."""),
            mo.hstack(
                [
                    current_net_worth,
                    use_custom_multiplier,
                    custom_multiplier_input,
                ]
            ),
        ]
    )
    return current_net_worth, custom_multiplier_input, use_custom_multiplier


@app.cell
def _(
    calculate_kahneman_multiplier,
    current_net_worth,
    custom_multiplier_input,
    mo,
    use_custom_multiplier,
):
    if use_custom_multiplier.value:
        kahneman_multiplier = custom_multiplier_input.value
        multiplier_source = "custom"
    else:
        kahneman_multiplier = calculate_kahneman_multiplier(current_net_worth.value)
        multiplier_source = "calculated"

    mo.md(
        f"""
        **Your Loss Aversion Multiplier: {kahneman_multiplier:.2f}x** ({multiplier_source})

        This means you need a potential gain to be {kahneman_multiplier:.1f}x larger than a certain loss to feel psychologically equivalent.
        """
    )
    return (kahneman_multiplier,)


@app.cell
def _(
    create_kahneman_curve,
    current_net_worth,
    custom_multiplier_input,
    mo,
    use_custom_multiplier,
):
    kahneman_fig = create_kahneman_curve(
        current_net_worth.value,
        custom_multiplier_input.value if use_custom_multiplier.value else None,
    )
    mo.vstack([
        mo.md("## Loss Aversion Curve"),
        mo.ui.plotly(kahneman_fig),
        mo.accordion({
            "Understanding Loss Aversion (Kahneman's Nobel Prize Work)": mo.md(
                r"""
        In 2002, Daniel Kahneman won the Nobel Prize in Economics for Prospect Theory, which fundamentally changed how we understand decision-making under risk.

        **The Core Finding:**

        Losses loom larger than gains. The pain of losing $1,000 is psychologically more powerful than the pleasure of gaining $1,000. Specifically, you need to gain roughly **2.5x as much** to offset the psychological impact of a loss.

        **Practical Examples:**

        - Would you play a game with 50% chance to win $1,100 and 50% chance to lose $1,000?
          - **Expected value**: +$50 (rational to play)
          - **Most people refuse** - the potential loss feels too painful

        - Would you play a game with 50% chance to win $2,500 and 50% chance to lose $1,000?
          - **Expected value**: +$750
          - **Most people accept** - the gain is large enough to compensate for loss aversion

        **Why This Matters for Startup Offers:**

        If you're taking a $200k/year pay cut to join a startup, you're experiencing a *guaranteed loss* right now. Your equity needs to be worth much more than $200k/year in expected value to rationally compensate for this certain loss.

        The multiplier depends on your net worth:
        - **Low net worth (<$250k)**: Need ~3x to compensate (can't afford losses)
        - **Medium net worth ($250k-$1M)**: Need ~2.5x (Kahneman's baseline)
        - **High net worth ($1M-$3M)**: Need ~2x (comfortable but cautious)
        - **Very high net worth (>$3M)**: Need ~1.5x (can afford risk)

        The curve above shows your personal loss aversion based on your net worth.
        """
            )
        })
    ])
    return


@app.cell
def _(mo):
    current_base = mo.ui.number(
        label="Base Salary ($)", value=250000, start=0, step=10000
    )

    current_bonus = mo.ui.number(
        label="Annual Bonus ($)", value=25000, start=0, step=5000
    )

    current_rsu_annual = mo.ui.number(
        label="Annual RSU Value ($)", value=125000, start=0, step=25000
    )

    company_401k_match = mo.ui.number(
        label="401k Match ($/year)", value=11500, start=0, step=500
    )

    mega_backdoor_available = mo.ui.checkbox(
        label="Mega Backdoor Roth Available", value=True
    )

    expected_annual_raise = mo.ui.slider(
        label="Expected Annual Raise (%)",
        start=0,
        stop=15,
        step=0.5,
        value=3.0,
        show_value=True,
    )

    promotion_probability = mo.ui.slider(
        label="Promotion Probability (next 3 years, %)",
        start=0,
        stop=100,
        step=5,
        value=30,
        show_value=True,
    )

    promotion_increase = mo.ui.slider(
        label="Promotion Comp Increase (%)",
        start=0,
        stop=50,
        step=5,
        value=20,
        show_value=True,
    )

    mo.accordion(
        {
            "## Current Compensation": mo.hstack(
                [
                    mo.vstack([current_base, current_bonus, current_rsu_annual]),
                    mo.vstack([company_401k_match, mega_backdoor_available]),
                    mo.vstack(
                        [
                            expected_annual_raise,
                            promotion_probability,
                            promotion_increase,
                        ]
                    ),
                ],
                justify="start",
            )
        }
    )
    return (
        company_401k_match,
        current_base,
        current_bonus,
        current_rsu_annual,
        expected_annual_raise,
        mega_backdoor_available,
        promotion_increase,
        promotion_probability,
    )


@app.cell
def _(
    CurrentCompensation,
    company_401k_match,
    current_base,
    current_bonus,
    current_rsu_annual,
    expected_annual_raise,
    mega_backdoor_available,
    mo,
    promotion_increase,
    promotion_probability,
):
    current_comp = CurrentCompensation(
        base_salary=current_base.value,
        annual_bonus=current_bonus.value,
        annual_rsu=current_rsu_annual.value,
        company_401k_match=company_401k_match.value,
        mega_backdoor_available=mega_backdoor_available.value,
        expected_annual_raise_pct=expected_annual_raise.value,
        promotion_probability=promotion_probability.value,
        promotion_comp_increase_pct=promotion_increase.value,
    )

    mo.md(f"**Current Total Compensation: ${current_comp.total_cash_comp:,.0f}/year**")
    return (current_comp,)


@app.cell
def _(mo):
    startup_base = mo.ui.number(
        label="Base Salary ($)", value=250000, start=0, step=10000
    )

    startup_bonus = mo.ui.number(
        label="Annual Bonus ($)", value=20000, start=0, step=5000
    )

    number_of_shares = mo.ui.number(
        label="Number of Shares", value=400000, start=0, step=10000
    )

    current_stock_fmv = mo.ui.number(
        label="Current Stock FMV ($/share)", value=1.50, start=0.01, step=0.01
    )

    strike_price = mo.ui.number(
        label="Strike Price ($/share)", value=1.50, start=0.01, step=0.01
    )

    current_valuation = mo.ui.number(
        label="Current Valuation ($M)", value=150, start=1, step=10
    )

    startup_401k_match = mo.ui.number(
        label="401k Match ($/year)", value=0, start=0, step=500
    )

    startup_mega_backdoor = mo.ui.checkbox(
        label="Mega Backdoor Roth Available", value=False
    )

    mo.accordion(
        {
            "## Startup Offer": mo.hstack(
                [
                    mo.vstack([startup_base, startup_bonus]),
                    mo.vstack([number_of_shares, current_stock_fmv, strike_price]),
                    mo.vstack(
                        [current_valuation, startup_401k_match, startup_mega_backdoor]
                    ),
                ],
                justify="start",
            )
        }
    )
    return (
        current_stock_fmv,
        current_valuation,
        number_of_shares,
        startup_401k_match,
        startup_base,
        startup_bonus,
        startup_mega_backdoor,
        strike_price,
    )


@app.cell
def _(
    StartupOffer,
    current_stock_fmv,
    current_valuation,
    mo,
    number_of_shares,
    startup_401k_match,
    startup_base,
    startup_bonus,
    startup_mega_backdoor,
    strike_price,
):
    startup_offer = StartupOffer(
        base_salary=startup_base.value,
        annual_bonus=startup_bonus.value,
        number_of_shares=number_of_shares.value,
        current_stock_fmv=current_stock_fmv.value,
        strike_price=strike_price.value,
        current_valuation_millions=current_valuation.value,
        company_401k_match=startup_401k_match.value,
        mega_backdoor_available=startup_mega_backdoor.value,
    )

    spread_explanation = ""
    if startup_offer.spread_at_exercise == 0:
        spread_explanation = """

        **Note:** Your spread at exercise is $0 because strike price equals current FMV. This means:
        - You pay ${:,.0f} to exercise (strike price × shares)
        - You receive shares currently worth ${:,.0f} (FMV × shares)
        - **No immediate paper gain**, but you lock in the right to buy at ${:.2f}/share forever
        - **Value comes from future growth**: If the company exits at 10x, your shares become worth ${:,.0f}M while you still only pay ${:,.0f}k to exercise
        """.format(
            startup_offer.exercise_cost,
            startup_offer.equity_value_at_fmv,
            startup_offer.strike_price,
            startup_offer.equity_value_at_fmv * 10 / 1_000_000,
            startup_offer.exercise_cost / 1000,
        )

    mo.md(
        f"""
        **Startup Cash Compensation: ${startup_offer.total_cash_comp:,.0f}/year**

        **Equity Summary:**
        - Number of shares: {startup_offer.number_of_shares:,.0f}
        - Strike price: ${startup_offer.strike_price:.2f}/share
        - Current FMV: ${startup_offer.current_stock_fmv:.2f}/share
        - Initial ownership: {startup_offer.initial_equity_percentage:.3f}%
        - Exercise cost: ${startup_offer.exercise_cost:,.0f}
        - Current equity value at FMV: ${startup_offer.equity_value_at_fmv:,.0f}
        - Spread at exercise (FMV - Strike): ${startup_offer.spread_at_exercise:,.0f}
        {spread_explanation}
        """
    )
    return (startup_offer,)


@app.cell
def _(mo):
    base_exit_probability = mo.ui.slider(
        label="Base Exit Probability (%)",
        start=0.1,
        stop=20,
        step=0.1,
        value=1.0,
        show_value=True,
    )

    founder_successful_exit = mo.ui.checkbox(
        label="Founder has successful exit (2x)", value=False
    )

    top_tier_investors = mo.ui.checkbox(label="Top-tier investors (1.5x)", value=False)

    series_b_or_later = mo.ui.checkbox(label="Series B or later (1.3x)", value=False)

    strong_traction = mo.ui.checkbox(label="Strong traction/PMF (1.4x)", value=False)

    mo.accordion(
        {
            "## Exit Probability": mo.vstack(
                [
                    base_exit_probability,
                    founder_successful_exit,
                    top_tier_investors,
                    series_b_or_later,
                    strong_traction,
                ]
            )
        }
    )
    return (
        base_exit_probability,
        founder_successful_exit,
        series_b_or_later,
        strong_traction,
        top_tier_investors,
    )


@app.cell
def _(
    ProbabilityFactors,
    base_exit_probability,
    founder_successful_exit,
    mo,
    series_b_or_later,
    strong_traction,
    top_tier_investors,
):
    probability_factors = ProbabilityFactors(
        base_probability=base_exit_probability.value,
        founder_has_exit=founder_successful_exit.value,
        top_tier_investors=top_tier_investors.value,
        series_b_or_later=series_b_or_later.value,
        strong_traction=strong_traction.value,
    )

    mo.md(
        f"""
        **Adjusted Exit Probability: {probability_factors.adjusted_probability:.2f}%**

        Base: {probability_factors.base_probability}% × Multiplier: {probability_factors.multiplier:.2f}x
        """
    )
    return (probability_factors,)


@app.cell
def _(mo):
    exit_multipliers_input = mo.ui.array(
        [
            mo.ui.number(
                label=f"Scenario {i + 1} (Multiple)", value=val, start=0.1, step=1
            )
            for i, val in enumerate([2, 5, 10, 20])
        ]
    )

    total_dilution = mo.ui.slider(
        label="Total Dilution (%)", start=0, stop=70, step=5, value=40, show_value=True
    )

    years_to_exit = mo.ui.slider(
        label="Years to Exit", start=1, stop=10, step=1, value=5, show_value=True
    )

    mo.accordion(
        {
            "## Financial Assumptions": mo.vstack(
                [exit_multipliers_input, total_dilution, years_to_exit]
            )
        }
    )
    return exit_multipliers_input, total_dilution, years_to_exit


@app.cell
def _(FinancialParameters, mo, total_dilution, years_to_exit):
    financial_params = FinancialParameters(
        years_to_exit=years_to_exit.value, total_dilution_pct=total_dilution.value
    )

    mo.md(
        f"After dilution, you'll own {100 - total_dilution.value:.0f}% of your initial equity stake."
    )
    return (financial_params,)


@app.cell
def _(ExitScenario, List, Tuple):
    def calculate_exit_scenarios(
        current_comp,
        startup_offer,
        financial_params,
        exit_multipliers: List[float],
    ) -> Tuple[List[ExitScenario], float, float, float]:
        years = financial_params.years_to_exit
        discount_rate = financial_params.discount_rate

        current_npv = 0
        for t in range(1, years + 1):
            year_comp = current_comp.total_cash_comp
            year_comp *= (1 + current_comp.expected_annual_raise_pct / 100) ** (t - 1)
            if t == 2:
                year_comp *= 1 + (current_comp.promotion_probability / 100) * (
                    current_comp.promotion_comp_increase_pct / 100
                )
            current_npv += year_comp / (1 + discount_rate) ** t

        startup_cash_npv = sum(
            [
                startup_offer.total_cash_comp / (1 + discount_rate) ** t
                for t in range(1, years + 1)
            ]
        )

        final_equity_pct = (
            startup_offer.initial_equity_percentage
            * financial_params.final_ownership_multiplier
        )

        scenarios = []
        for multiplier in exit_multipliers:
            exit_val_millions = startup_offer.current_valuation_millions * multiplier
            equity_value = exit_val_millions * 1_000_000 * (final_equity_pct / 100)
            proceeds_after_exercise = equity_value - startup_offer.exercise_cost
            after_tax = proceeds_after_exercise * (
                1 - financial_params.capital_gains_tax
            )

            exit_npv = after_tax / (1 + discount_rate) ** years
            total_npv = startup_cash_npv + exit_npv

            scenarios.append(
                ExitScenario(
                    valuation_multiple=multiplier,
                    exit_probability=0,  # Will be set separately when needed
                    exit_valuation_millions=exit_val_millions,
                    equity_value=equity_value,
                    after_tax_proceeds=after_tax,
                    exit_npv=exit_npv,
                    total_npv=total_npv,
                    net_vs_current=total_npv - current_npv,
                    probability_weighted_npv=0,  # Will be calculated when needed
                )
            )

        return scenarios, current_npv, startup_cash_npv, final_equity_pct

    def calculate_retirement_benefits(
        current_comp, startup_offer, financial_params
    ) -> Tuple[float, float, float]:
        years = financial_params.retirement_horizon_years
        rate = financial_params.retirement_growth_rate

        fv_factor = ((rate**years - 1) / (rate - 1)) * rate

        current_401k_fv = current_comp.total_401k_benefit * fv_factor
        startup_401k_fv = startup_offer.total_401k_benefit * fv_factor

        return current_401k_fv, startup_401k_fv, current_401k_fv - startup_401k_fv
    return calculate_exit_scenarios, calculate_retirement_benefits


@app.cell
def _(
    calculate_exit_scenarios,
    current_comp,
    exit_multipliers_input,
    financial_params,
    startup_offer,
):
    exit_multipliers = list(exit_multipliers_input.value)
    exit_scenarios, current_npv, startup_cash_npv, final_equity_pct = (
        calculate_exit_scenarios(
            current_comp, startup_offer, financial_params, exit_multipliers
        )
    )
    return (
        current_npv,
        exit_multipliers,
        exit_scenarios,
        final_equity_pct,
        startup_cash_npv,
    )


@app.cell
def _(
    calculate_retirement_benefits,
    current_comp,
    financial_params,
    startup_offer,
):
    retirement_current, retirement_startup, retirement_diff = (
        calculate_retirement_benefits(current_comp, startup_offer, financial_params)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(go, make_subplots, np):
    def create_npv_visualizations(
        exit_scenarios,
        current_npv,
        startup_cash_npv,
        kahneman_multiplier,
        financial_params,
        startup_offer,
        final_equity_pct,
    ):
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "NPV Comparison (If Exit)",
                "Break-Even Analysis",
                "After-Tax Equity Value",
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]],
            horizontal_spacing=0.10,
        )

        scenarios_labels = [f"{es.valuation_multiple}x" for es in exit_scenarios]
        startup_npvs = [es.total_npv for es in exit_scenarios]

        fig.add_trace(
            go.Bar(
                x=scenarios_labels,
                y=[current_npv] * len(scenarios_labels),
                name="Current (Certain)",
                marker_color="#0084FF",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=scenarios_labels,
                y=startup_npvs,
                name="Startup (If Exit)",
                marker_color="#E60023",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        multipliers = np.linspace(1, 30, 100)
        breakeven_values = []

        for mult in multipliers:
            exit_val = startup_offer.current_valuation_millions * mult * 1_000_000
            equity_val = exit_val * (final_equity_pct / 100)
            after_tax = (equity_val - startup_offer.exercise_cost) * (
                1 - financial_params.capital_gains_tax
            )
            exit_npv_temp = (
                after_tax
                / (1 + financial_params.discount_rate) ** financial_params.years_to_exit
            )
            breakeven_values.append(startup_cash_npv + exit_npv_temp)

        fig.add_trace(
            go.Scatter(
                x=multipliers,
                y=breakeven_values,
                mode="lines",
                name="Startup NPV",
                line=dict(color="#E60023", width=3),
            ),
            row=1,
            col=2,
        )

        fig.add_hline(
            y=current_npv,
            line_dash="dash",
            line_color="#0084FF",
            annotation_text="Current NPV",
            row=1,
            col=2,
        )

        kahneman_adjusted = current_npv * kahneman_multiplier
        fig.add_hline(
            y=kahneman_adjusted,
            line_dash="dot",
            line_color="#FF6B6B",
            annotation_text=f"Loss-Adjusted ({kahneman_multiplier:.1f}x)",
            row=1,
            col=2,
        )

        equity_values_millions = [
            es.after_tax_proceeds / 1_000_000 for es in exit_scenarios
        ]

        fig.add_trace(
            go.Bar(
                x=scenarios_labels,
                y=equity_values_millions,
                name="After-Tax Equity",
                marker_color="#00C853",
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        opportunity_cost = (current_npv - startup_cash_npv) / 1_000_000
        fig.add_hline(
            y=opportunity_cost,
            line_dash="dash",
            line_color="#FF9800",
            annotation_text="Opportunity Cost",
            row=1,
            col=3,
        )

        fig.update_xaxes(title_text="Exit Multiple", row=1, col=1)
        fig.update_yaxes(title_text="NPV ($)", row=1, col=1)

        fig.update_xaxes(title_text="Exit Multiple", row=1, col=2)
        fig.update_yaxes(title_text="NPV ($)", row=1, col=2)

        fig.update_xaxes(title_text="Exit Multiple", row=1, col=3)
        fig.update_yaxes(title_text="After-Tax Value ($M)", row=1, col=3)

        fig.update_layout(
            height=500,
            showlegend=True,
            template="plotly_white",
            title_text="Financial Analysis",
            title_x=0.5,
        )

        return fig
    return (create_npv_visualizations,)


@app.cell
def _(
    create_npv_visualizations,
    current_npv,
    exit_scenarios,
    final_equity_pct,
    financial_params,
    kahneman_multiplier,
    pl,
    startup_cash_npv,
    startup_offer,
):
    npv_fig = create_npv_visualizations(
        exit_scenarios,
        current_npv,
        startup_cash_npv,
        kahneman_multiplier,
        financial_params,
        startup_offer,
        final_equity_pct,
    )

    _scenario_df = pl.DataFrame(
        {
            "Exit Multiple": [f"{es.valuation_multiple}x" for es in exit_scenarios],
            "Exit Valuation": [
                f"${es.exit_valuation_millions:,.0f}M" for es in exit_scenarios
            ],
            "Gross Equity": [f"${es.equity_value:,.0f}" for es in exit_scenarios],
            "After-Tax": [f"${es.after_tax_proceeds:,.0f}" for es in exit_scenarios],
            "Total NPV (If Exit)": [f"${es.total_npv:,.0f}" for es in exit_scenarios],
            "vs Current": [
                f"${es.net_vs_current:,.0f}"
                if es.net_vs_current >= 0
                else f"-${abs(es.net_vs_current):,.0f}"
                for es in exit_scenarios
            ],
        }
    )
    return


@app.cell
def _(kahneman_multiplier, mo):
    mo.md(
        f"""
    ## Results

    The heatmap shows whether the startup offer is rational at different exit scenarios:

    - **Cell value & color**: Loss-adjusted gain/loss (expected NPV - current NPV × {kahneman_multiplier:.1f})
      - Green = rational decision (compensates for loss aversion)
      - Red = irrational decision (doesn't justify the risk)

    - **Hover for details**: Shows all the NPV values and comparisons:
      - Your expected startup NPV at this exit scenario
      - Current job NPV (flat salary vs with career growth)
      - Three comparisons: vs flat, vs growth, and loss-adjusted
    """
    )
    return


@app.cell
def _(
    create_breakeven_heatmap,
    current_comp,
    current_npv,
    final_equity_pct,
    financial_params,
    kahneman_multiplier,
    mo,
    probability_factors,
    startup_cash_npv,
    startup_offer,
):
    heatmap_fig_summary = create_breakeven_heatmap(
        current_npv,
        kahneman_multiplier,
        startup_offer,
        startup_cash_npv,
        financial_params,
        final_equity_pct,
        probability_factors,
        current_comp,
    )
    mo.ui.plotly(heatmap_fig_summary)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## Breakeven Surface: Exit Multiple vs Probability

    This 3D visualization shows all combinations of exit valuation and exit probability that result in breakeven with your current position.

    **How to read this:**
    - **Blue surface**: Expected NPV of the startup as a function of exit multiple and probability
    - **Red plane**: Your current job's NPV (certain, doesn't vary)
    - **Orange plane**: Loss-adjusted breakeven (accounting for Kahneman loss aversion)
    - **Green contour**: Where startup NPV equals loss-adjusted breakeven

    The startup only makes sense in regions where the blue surface is above the orange plane. This shows you need either:
    - High exit multiple with moderate probability, OR
    - Moderate exit multiple with high probability

    Most startup offers fall below the breakeven surface, which is why they feel like bad deals even when the "math works out."
    """
    )
    return


@app.cell
def _(go, np):
    def create_breakeven_3d_surface(
        startup_offer,
        startup_cash_npv,
        current_npv,
        kahneman_multiplier,
        financial_params,
        final_equity_pct,
    ):
        exit_multiples = np.linspace(0.5, 50, 100)
        exit_probabilities = np.linspace(0, 100, 100)

        X, Y = np.meshgrid(exit_multiples, exit_probabilities)
        Z = np.zeros_like(X)

        for i in range(len(exit_probabilities)):
            for j in range(len(exit_multiples)):
                mult = X[i, j]
                prob = Y[i, j] / 100

                exit_val = startup_offer.current_valuation_millions * mult * 1_000_000
                equity_val = exit_val * (final_equity_pct / 100)
                after_tax = (equity_val - startup_offer.exercise_cost) * (
                    1 - financial_params.capital_gains_tax
                )
                exit_npv = (
                    after_tax
                    / (1 + financial_params.discount_rate)
                    ** financial_params.years_to_exit
                )

                expected_npv = (
                    prob * (startup_cash_npv + exit_npv) + (1 - prob) * startup_cash_npv
                )
                Z[i, j] = expected_npv

        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Blues",
                name="Startup Expected NPV",
                opacity=0.9,
                showscale=True,
                colorbar=dict(title="NPV ($)", x=1.1),
                hovertemplate="Exit Multiple: %{x:.1f}x<br>Exit Probability: %{y:.1f}%%<br>Expected NPV: $%{z:,.0f}<extra></extra>",
            )
        )

        current_plane = np.full_like(Z, current_npv)

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=current_plane,
                colorscale=[[0, "rgba(0,132,255,0.7)"], [1, "rgba(0,132,255,0.7)"]],
                name="Current Job NPV",
                showscale=False,
                opacity=0.8,
                hovertemplate="Exit Multiple: %{x:.1f}x<br>Exit Probability: %{y:.1f}%%<br>Current Job NPV: $%{z:,.0f}<extra></extra>",
            )
        )

        loss_adjusted_plane = np.full_like(Z, current_npv * kahneman_multiplier)

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=loss_adjusted_plane,
                colorscale=[[0, "rgba(255,107,107,0.7)"], [1, "rgba(255,107,107,0.7)"]],
                name=f"Loss-Adjusted ({kahneman_multiplier:.1f}x)",
                showscale=False,
                opacity=0.8,
                hovertemplate="Exit Multiple: %{x:.1f}x<br>Exit Probability: %{y:.1f}%%<br>Loss-Adjusted Target: $%{z:,.0f}<extra></extra>",
            )
        )

        # Find the 3D contour curve where startup NPV equals loss-adjusted breakeven
        contour_level = current_npv * kahneman_multiplier
        contour_x = []
        contour_y = []
        contour_z = []

        # For each exit multiple, find the probability where Z crosses the threshold
        for j in range(len(exit_multiples)):
            mult = exit_multiples[j]
            # Get Z values for this multiple across all probabilities
            z_slice = Z[:, j]
            # Find where it crosses the threshold
            for i in range(len(exit_probabilities) - 1):
                if (z_slice[i] <= contour_level <= z_slice[i + 1]) or (
                    z_slice[i] >= contour_level >= z_slice[i + 1]
                ):
                    # Linear interpolation
                    t = (contour_level - z_slice[i]) / (z_slice[i + 1] - z_slice[i])
                    prob_interp = exit_probabilities[i] + t * (
                        exit_probabilities[i + 1] - exit_probabilities[i]
                    )
                    contour_x.append(mult)
                    contour_y.append(prob_interp)
                    contour_z.append(contour_level)
                    break

        if contour_x:
            fig.add_trace(
                go.Scatter3d(
                    x=contour_x,
                    y=contour_y,
                    z=contour_z,
                    mode="lines+markers",
                    line=dict(color="green", width=6),
                    marker=dict(size=4, color="green"),
                    name="Breakeven Curve",
                    hovertemplate="Exit Multiple: %{x:.1f}x<br>Exit Probability: %{y:.1f}%%<br>Breakeven NPV: $%{z:,.0f}<extra></extra>",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="Exit Valuation Multiple",
                yaxis_title="Exit Probability (%)",
                zaxis_title="Expected NPV ($)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            title=dict(
                text="Breakeven Analysis: Exit Multiple vs Probability<br><sub>Blue surface must exceed orange plane for rational acceptance</sub>",
                x=0.5,
                xanchor="center",
            ),
            height=700,
            showlegend=True,
        )

        return fig
    return (create_breakeven_3d_surface,)


@app.cell
def _(
    create_breakeven_3d_surface,
    current_npv,
    final_equity_pct,
    financial_params,
    kahneman_multiplier,
    mo,
    startup_cash_npv,
    startup_offer,
):
    breakeven_3d_fig = create_breakeven_3d_surface(
        startup_offer,
        startup_cash_npv,
        current_npv,
        kahneman_multiplier,
        financial_params,
        final_equity_pct,
    )
    mo.ui.plotly(breakeven_3d_fig)
    return


@app.cell
def _(go):
    def create_breakeven_heatmap(
        current_npv,
        kahneman_multiplier,
        startup_offer,
        startup_cash_npv,
        financial_params,
        final_equity_pct,
        probability_factors,
        current_comp,
    ):
        exit_multiples_hm = [2, 5, 10, 15, 20, 30, 50]
        probabilities_hm = [1, 2, 5, 10, 20, 30, 50, 75, 100]

        years = financial_params.years_to_exit
        discount_rate = financial_params.discount_rate
        current_npv_flat = sum(
            [
                current_comp.total_cash_comp / (1 + discount_rate) ** t
                for t in range(1, years + 1)
            ]
        )

        Z_heatmap = []
        Z_display = []
        hover_text = []

        for prob in probabilities_hm:
            row_color = []
            row_display = []
            row_hover = []
            for mult in exit_multiples_hm:
                exit_val = startup_offer.current_valuation_millions * mult * 1_000_000
                equity_val = exit_val * (final_equity_pct / 100)
                after_tax = (equity_val - startup_offer.exercise_cost) * (
                    1 - financial_params.capital_gains_tax
                )
                exit_npv = (
                    after_tax
                    / (1 + financial_params.discount_rate)
                    ** financial_params.years_to_exit
                )

                prob_pct = prob / 100
                expected_npv = (
                    prob_pct * (startup_cash_npv + exit_npv)
                    + (1 - prob_pct) * startup_cash_npv
                )

                base_diff = expected_npv - current_npv
                flat_diff = expected_npv - current_npv_flat
                loss_adjusted_flat = expected_npv - (
                    current_npv_flat * kahneman_multiplier
                )
                loss_adjusted_growth = expected_npv - (
                    current_npv * kahneman_multiplier
                )

                row_color.append(loss_adjusted_growth)
                row_display.append(loss_adjusted_growth)

                hover = (
                    f"<b>Exit Scenario: {mult}x @ {prob}% probability</b><br><br>"
                    f"━━━ RISK-ADJUSTED EXPECTED VALUE ({kahneman_multiplier:.1f}x loss aversion) ━━━<br>"
                    f"<b>vs Current job (with career growth):</b> ${loss_adjusted_growth / 1e6:.2f}M<br>"
                    f"<b>vs Current job (no career growth):</b> ${loss_adjusted_flat / 1e6:.2f}M<br><br>"
                    f"━━━ EXPECTED VALUE (no risk adjustment) ━━━<br>"
                    f"<b>vs Current job (with career growth):</b> ${base_diff / 1e6:.2f}M<br>"
                    f"<b>vs Current job (no career growth):</b> ${flat_diff / 1e6:.2f}M<br><br>"
                    f"━━━ UNDERLYING NPV VALUES ━━━<br>"
                    f"Startup expected NPV: ${expected_npv / 1e6:.2f}M<br>"
                    f"Current job (growth): ${current_npv / 1e6:.2f}M<br>"
                    f"Current job (flat): ${current_npv_flat / 1e6:.2f}M"
                )
                row_hover.append(hover)

            Z_heatmap.append(row_color)
            Z_display.append(row_display)
            hover_text.append(row_hover)

        fig = go.Figure(
            data=go.Heatmap(
                z=Z_heatmap,
                x=[f"{m}x" for m in exit_multiples_hm],
                y=[f"{p}%" for p in probabilities_hm],
                colorscale=[
                    [0, "#FF4444"],
                    [0.5, "#FFFFFF"],
                    [1, "#44FF44"],
                ],
                zmid=0,
                text=[[f"${v / 1e6:.2f}M" for v in row] for row in Z_display],
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>",
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Loss-Adjusted Gain ($)"),
            )
        )

        user_prob = probability_factors.adjusted_probability
        fig.add_annotation(
            text=f"Your Estimate: {user_prob:.1f}%",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=14, color="blue"),
        )

        fig.update_layout(
            title="Expected Value vs Current Job (Green = Gain, Red = Loss)",
            xaxis_title="Exit Valuation Multiple",
            yaxis_title="Exit Probability",
            height=500,
            template="plotly_white",
        )

        return fig
    return (create_breakeven_heatmap,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Breakeven Probability Analysis

    The table below shows what exit probability you need at each valuation multiple to make the startup offer rational.
    """
    )
    return


@app.function
def calculate_required_probability(
    exit_multiple,
    startup_offer,
    startup_cash_npv,
    target_npv,
    financial_params,
    final_equity_pct,
):
    """Calculate the probability needed at a given exit multiple to reach target NPV"""
    exit_val = startup_offer.current_valuation_millions * exit_multiple * 1_000_000
    equity_val = exit_val * (final_equity_pct / 100)
    after_tax = (equity_val - startup_offer.exercise_cost) * (
        1 - financial_params.capital_gains_tax
    )
    exit_npv = (
        after_tax
        / (1 + financial_params.discount_rate) ** financial_params.years_to_exit
    )

    # Expected NPV = prob * (startup_cash + exit_npv) + (1-prob) * startup_cash
    # = startup_cash + prob * exit_npv
    # Solve for prob: prob = (target_npv - startup_cash) / exit_npv

    if exit_npv <= 0:
        return None  # Can't reach target with this multiple

    required_prob = (target_npv - startup_cash_npv) / exit_npv
    return max(0, min(100, required_prob * 100))


@app.cell
def _(
    current_npv,
    exit_multipliers,
    final_equity_pct,
    financial_params,
    kahneman_multiplier,
    mo,
    pl,
    probability_factors,
    startup_cash_npv,
    startup_offer,
):
    breakeven_data = []
    user_prob = probability_factors.adjusted_probability

    for mult in exit_multipliers:
        req_prob_current = calculate_required_probability(
            mult,
            startup_offer,
            startup_cash_npv,
            current_npv,
            financial_params,
            final_equity_pct,
        )
        req_prob_loss_adj = calculate_required_probability(
            mult,
            startup_offer,
            startup_cash_npv,
            current_npv * kahneman_multiplier,
            financial_params,
            final_equity_pct,
        )

        # Deal is rational if user's probability >= required loss-adjusted probability
        is_rational = (
            "✓"
            if req_prob_loss_adj is not None and user_prob >= req_prob_loss_adj
            else "✗"
        )

        breakeven_data.append(
            {
                "Exit Multiple": f"{mult}x",
                "Required Prob (Current NPV)": (
                    f"{req_prob_current:.2f}%"
                    if req_prob_current is not None
                    else "N/A"
                ),
                "Required Prob (Loss-Adjusted)": (
                    f"{req_prob_loss_adj:.2f}%"
                    if req_prob_loss_adj is not None
                    else "N/A"
                ),
                "Your Est. Probability": f"{user_prob:.2f}%",
                "Rational?": is_rational,
            }
        )

    breakeven_df = pl.DataFrame(breakeven_data)
    mo.ui.table(breakeven_df.to_pandas())
    return


@app.cell
def _(current_npv, kahneman_multiplier, mo):
    mo.md(
        f"""
    ### Key Insights from the Breakeven Surface

    The green curve shows all (exit multiple, probability) combinations that exactly equal your loss-adjusted breakeven of ${current_npv * kahneman_multiplier:,.0f}.

    Points above and to the right of this curve represent rational deals (after accounting for loss aversion).
    Points below and to the left are irrational - you're taking certain losses for insufficient expected gains.

    The table above shows exactly what probability you need at each exit multiple. If your estimated probability is below the "Required Prob (Loss-Adjusted)", the deal is marked as irrational (✗).
    """
    )
    return


if __name__ == "__main__":
    app.run()
