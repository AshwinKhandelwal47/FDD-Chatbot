from typing import Any


class FinancialRatios:
    """All financial math lives here; the LLM only interprets results."""

    @staticmethod
    def gross_margin(revenue: float, cogs: float) -> float:
        return round((revenue - cogs) / revenue * 100, 2)

    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float) -> float:
        return round(current_assets / current_liabilities, 2)

    @staticmethod
    def dso(accounts_receivable: float, revenue: float) -> float:
        return round((accounts_receivable / revenue) * 365, 1)

    @staticmethod
    def cagr(start_value: float, end_value: float, years: int) -> float:
        return round(((end_value / start_value) ** (1 / years) - 1) * 100, 2)

    def compute_all(self, stmts: list[Any], sheets: list[Any], cashflows: list[Any]) -> dict[str, dict[str, float]]:
        ratios: dict[str, dict[str, float]] = {}
        for stmt in stmts:
            year = str(stmt.year)
            ratios[year] = {
                "gross_margin": self.gross_margin(stmt.revenue, stmt.cogs),
            }

        for sheet in sheets:
            year = str(sheet.year)
            ratios.setdefault(year, {})
            ratios[year].update(
                {
                    "current_ratio": self.current_ratio(
                        sheet.current_assets, sheet.current_liabilities
                    )
                }
            )

        for cf in cashflows:
            year = str(cf.year)
            ratios.setdefault(year, {})
            if hasattr(cf, "accounts_receivable") and cf.accounts_receivable is not None:
                ratios[year]["dso"] = self.dso(cf.accounts_receivable, cf.revenue)

        if len(stmts) >= 2:
            sorted_years = sorted(stmts, key=lambda x: x.year)
            start_value = sorted_years[0].revenue
            end_value = sorted_years[-1].revenue
            years = sorted_years[-1].year - sorted_years[0].year
            if start_value > 0 and years > 0:
                ratios.setdefault(str(sorted_years[-1].year), {})
                ratios[str(sorted_years[-1].year)]["cagr"] = self.cagr(
                    start_value, end_value, years
                )

        return ratios
