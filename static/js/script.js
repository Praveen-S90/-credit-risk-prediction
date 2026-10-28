
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("creditForm");
    const resultSection = document.getElementById("resultSection");
    const riskLabel = document.getElementById("riskLabel");
    const lowRisk = document.getElementById("lowRisk");
    const highRisk = document.getElementById("highRisk");
    const predictButton = document.getElementById("predictButton");
    const explainabilitySection = document.getElementById("explainabilitySection");
    const historyStatus = document.getElementById("historyStatus");
    const historyTableBody = document.getElementById("historyTableBody");
    const refreshHistoryButton = document.getElementById("refreshHistory");

    if (!form) {
        console.error('Missing form with id="creditForm". Check index.html.');
        return;
    }

    function getValue(id, numeric = false) {
        const element = document.getElementById(id);
        if (!element) throw new Error("Missing form field: " + id);
        return numeric ? Number(element.value) : element.value;
    }

    function escapeHTML(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function renderExplanation(items, valueKey, title) {
        if (!Array.isArray(items) || items.length === 0) {
            return '<p class="no-explanation">No valid ' + title +
                ' explanation is available.</p>';
        }

        const validItems = items.map(item => ({
            feature: String(item.feature ?? "Unknown feature"),
            value: Number(item[valueKey])
        })).filter(item => Number.isFinite(item.value));

        if (!validItems.length) {
            return '<p class="no-explanation">No valid ' + title +
                ' explanation is available.</p>';
        }

        const maxImpact = Math.max(
            ...validItems.map(item => Math.abs(item.value)),
            0.000001
        );

        return validItems.map(item => {
            const increasing = item.value >= 0;
            const width = Math.max(
                3, Math.abs(item.value) / maxImpact * 100
            );
            const colorClass = increasing
                ? "impact-increase" : "impact-decrease";
            const direction = increasing
                ? "Increases risk" : "Decreases risk";

            return `
                <details class="impact-card ${colorClass}">
                    <summary class="impact-summary">
                        <div class="impact-topline">
                            <span class="impact-feature">
                                ${escapeHTML(item.feature)}
                            </span>
                            <span class="impact-value">
                                ${item.value > 0 ? "+" : ""}${item.value.toFixed(3)}
                            </span>
                        </div>
                        <div class="impact-track">
                            <div class="impact-fill" style="width:${width}%"></div>
                        </div>
                        <span class="impact-direction">
                            ${increasing ? "↑" : "↓"} ${direction}
                        </span>
                    </summary>
                    <div class="impact-details">
                        This ${title} value is ${item.value.toFixed(4)}.
                        Its sign indicates the direction of its contribution.
                    </div>
                </details>`;
        }).join("");
    }

    async function loadPredictionHistory() {
        if (!historyStatus || !historyTableBody) return;

        historyStatus.textContent = "Loading prediction history...";
        historyTableBody.replaceChildren();

        try {
            const response = await fetch("/history");
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Could not load history.");
            }

            const history = Array.isArray(data.history)
                ? data.history : [];

            if (history.length === 0) {
                historyStatus.textContent = "No saved predictions yet.";
                return;
            }

            history.forEach(item => {
                const row = document.createElement("tr");
                const score = Number(item.risk_score);

                const values = [
                    item.id ?? "—",
                    Number.isFinite(score)
                        ? score.toFixed(2) + "%" : "—",
                    item.risk_level ?? "—",
                    item.created_at ?? "—"
                ];

                values.forEach(value => {
                    const cell = document.createElement("td");
                    cell.textContent = String(value);
                    cell.style.padding = "12px";
                    cell.style.borderBottom = "1px solid #39364c";
                    row.appendChild(cell);
                });

                historyTableBody.appendChild(row);
            });

            historyStatus.textContent =
                `Showing ${history.length} saved prediction(s).`;

        } catch (error) {
            console.error("History loading error:", error);
            historyStatus.textContent =
                "Could not load history. Check that Flask is running.";
        }
    }

    form.addEventListener("submit", async event => {
        event.preventDefault();

        if (predictButton) {
            predictButton.disabled = true;
            predictButton.textContent = "⏳ Analyzing...";
        }

        try {
            const formData = {
                checking_status: getValue("checking_status"),
                duration: getValue("duration", true),
                credit_history: getValue("credit_history"),
                purpose: getValue("purpose"),
                credit_amount: getValue("credit_amount", true),
                savings_status: getValue("savings_status"),
                employment: getValue("employment"),
                installment_rate: getValue("installment_rate", true),
                personal_status: getValue("personal_status"),
                other_debtors: getValue("other_debtors"),
                residence_since: getValue("residence_since", true),
                property: getValue("property"),
                age: getValue("age", true),
                other_installment_plans: getValue("other_installment_plans"),
                housing: getValue("housing"),
                existing_credits: getValue("existing_credits", true),
                job: getValue("job"),
                num_dependents: getValue("num_dependents", true),
                telephone: getValue("telephone"),
                foreign_worker: getValue("foreign_worker")
            };

            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed.");
            }

            const applicantSummary =
                document.getElementById("applicantSummary");

            if (applicantSummary && result.applicant_summary) {
                applicantSummary.replaceChildren();

                Object.entries(result.applicant_summary).forEach(
                    ([key, value]) => {
                        const item = document.createElement("div");
                        item.className = "summary-item";

                        const label = document.createElement("span");
                        label.className = "summary-label";
                        label.textContent = key
                            .replace(/_/g, " ")
                            .replace(/\b\w/g, char => char.toUpperCase());

                        const data = document.createElement("strong");
                        data.className = "summary-value";
                        data.textContent = String(value);

                        item.append(label, data);
                        applicantSummary.appendChild(item);
                    }
                );
            }

            const predictionTime =
                document.getElementById("predictionTime");

            if (predictionTime && result.prediction_time) {
                predictionTime.textContent = result.prediction_time;
            }

            if (resultSection) resultSection.style.display = "block";

            const isHighRisk = Number(result.prediction) === 1;

            if (riskLabel) {
                riskLabel.textContent = isHighRisk
                    ? "🔴 HIGH RISK" : "🟢 LOW RISK";
                riskLabel.className = isHighRisk
                    ? "high-risk" : "low-risk";
            }

            if (lowRisk) {
                lowRisk.textContent =
                    Number(result.low_risk_probability).toFixed(2) + "%";
            }

            if (highRisk) {
                highRisk.textContent =
                    Number(result.high_risk_probability).toFixed(2) + "%";
            }

            const riskScore = document.getElementById("riskScore");
            const riskLevel = document.getElementById("riskLevel");
            const riskMessage = document.getElementById("riskMessage");
            const riskMeterFill = document.getElementById("riskMeterFill");
            const score = Number(result.risk_score);

            if (riskScore) {
                riskScore.textContent = Number.isFinite(score)
                    ? score.toFixed(2) + "%" : "—";
            }

            if (riskLevel) {
                riskLevel.classList.remove("risk-medium", "risk-high");
                riskLevel.textContent =
                    `${result.risk_level ?? (isHighRisk ? "HIGH" : "LOW")} RISK`;

                if (score >= 60) {
                    riskLevel.classList.add("risk-high");
                } else if (score >= 30) {
                    riskLevel.classList.add("risk-medium");
                }
            }

            if (riskMessage) {
                riskMessage.textContent = result.risk_message ?? "";
            }

            if (riskMeterFill) {
                const safeScore = Number.isFinite(score)
                    ? Math.min(100, Math.max(0, score)) : 0;

                riskMeterFill.style.width = "0%";
                requestAnimationFrame(() => {
                    riskMeterFill.style.width = safeScore + "%";
                });
            }

            if (explainabilitySection) {
                const shapHTML = renderExplanation(
                    result.shap_explanation, "shap_value", "SHAP"
                );
                const limeHTML = renderExplanation(
                    result.lime_explanation, "weight", "LIME"
                );

                explainabilitySection.style.display = "block";
                explainabilitySection.innerHTML = `
                    <div class="explanation-result interactive-explanation">
                        <div class="explanation-heading">
                            <span class="explanation-icon">📊</span>
                            <div>
                                <h3>SHAP Explanation</h3>
                                <p>Discover which features influence this prediction.</p>
                            </div>
                        </div>
                        <div class="explanation-legend">
                            <span class="legend-up">↑ Increases risk</span>
                            <span class="legend-down">↓ Decreases risk</span>
                        </div>
                        <div class="shap-list">${shapHTML}</div>

                        <div class="explanation-heading lime-heading">
                            <span class="explanation-icon">🔎</span>
                            <div>
                                <h3>LIME Explanation</h3>
                                <p>Explore the rules behind this individual prediction.</p>
                            </div>
                        </div>
                        <div class="lime-list">${limeHTML}</div>
                        <p class="explanation-note">
                            These explanations describe how the model reached
                            its prediction; they do not guarantee loan outcomes.
                        </p>
                    </div>`;
            }

            if (resultSection) {
                resultSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

            await loadPredictionHistory();

        } catch (error) {
            console.error("Prediction error:", error);
            alert("Prediction failed: " + error.message);
        } finally {
            if (predictButton) {
                predictButton.disabled = false;
                predictButton.innerHTML =
                    "<span>✦</span> Predict Credit Risk <span>→</span>";
            }
        }
    });

    if (refreshHistoryButton) {
        refreshHistoryButton.addEventListener(
            "click", loadPredictionHistory
        );
    }

    loadPredictionHistory();
});
