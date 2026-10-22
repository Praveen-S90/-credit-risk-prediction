document.addEventListener("DOMContentLoaded", () => {

    // =====================================================
    // GET HTML ELEMENTS
    // =====================================================

    const form = document.getElementById("creditForm");
    const resultSection = document.getElementById("resultSection");
    const riskLabel = document.getElementById("riskLabel");
    const lowRisk = document.getElementById("lowRisk");
    const highRisk = document.getElementById("highRisk");
    const explainabilitySection =
        document.getElementById("explainabilitySection");
    const predictButton =
        document.getElementById("predictButton");


    // =====================================================
    // CHECK REQUIRED ELEMENTS
    // =====================================================

    if (!form) {
        console.error("creditForm not found in index.html");
        return;
    }

    if (!predictButton) {
        console.error("predictButton not found in index.html");
        return;
    }


    // =====================================================
    // FORM SUBMIT
    // =====================================================

    form.addEventListener("submit", async (event) => {

        event.preventDefault();


        // -------------------------------------------------
        // BUTTON LOADING STATE
        // -------------------------------------------------

        predictButton.disabled = true;
        predictButton.innerHTML = "⏳ Analyzing...";


        try {

            // =================================================
            // COLLECT FORM DATA
            // =================================================

            const formData = {

                checking_status:
                    document.getElementById("checking_status").value,

                duration:
                    Number(document.getElementById("duration").value),

                credit_history:
                    document.getElementById("credit_history").value,

                purpose:
                    document.getElementById("purpose").value,

                credit_amount:
                    Number(document.getElementById("credit_amount").value),

                savings_status:
                    document.getElementById("savings_status").value,

                employment:
                    document.getElementById("employment").value,

                installment_rate:
                    Number(document.getElementById("installment_rate").value),

                personal_status:
                    document.getElementById("personal_status").value,

                other_debtors:
                    document.getElementById("other_debtors").value,

                residence_since:
                    Number(document.getElementById("residence_since").value),

                property:
                    document.getElementById("property").value,

                age:
                    Number(document.getElementById("age").value),

                other_installment_plans:
                    document.getElementById("other_installment_plans").value,

                housing:
                    document.getElementById("housing").value,

                existing_credits:
                    Number(document.getElementById("existing_credits").value),

                job:
                    document.getElementById("job").value,

                num_dependents:
                    Number(document.getElementById("num_dependents").value),

                telephone:
                    document.getElementById("telephone").value,

                foreign_worker:
                    document.getElementById("foreign_worker").value
            };


            console.log("Sending data:", formData);


            // =================================================
            // SEND DATA TO FLASK
            // =================================================

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(formData)
            });


            // =================================================
            // READ SERVER RESPONSE
            // =================================================

            const result = await response.json();
            // =========================================================
            // APPLICANT SUMMARY
            // =========================================================

            const applicantSummary =
                document.getElementById("applicantSummary");

            if (applicantSummary && result.applicant_summary) {

                const summary = result.applicant_summary;

                applicantSummary.innerHTML = "";

                Object.entries(summary).forEach(([key, value]) => {

                    const item = document.createElement("div");
                    item.className = "summary-item";

                    const label = document.createElement("span");
                    label.className = "summary-label";

                    label.textContent = key
                        .replace(/_/g, " ")
                        .replace(/\b\w/g, char => char.toUpperCase());

                    const data = document.createElement("strong");
                    data.className = "summary-value";
                    data.textContent = value;

                    item.appendChild(label);
                    item.appendChild(data);

                    applicantSummary.appendChild(item);
                });
            }
            const predictionTime =
                document.getElementById("predictionTime");

            if (predictionTime && result.prediction_time) {
                predictionTime.textContent =
                    result.prediction_time;
            }

            console.log("Server response:", result);


            if (!response.ok) {

                throw new Error(
                    result.error || "Prediction failed"
                );
            }


            // =================================================
            // SHOW RESULT SECTION
            // =================================================

            if (resultSection) {
                resultSection.style.display = "block";
            }


            // =================================================
            // SHOW RISK PREDICTION
            // =================================================

            if (Number(result.prediction) === 1) {

                riskLabel.textContent = "🔴 HIGH RISK";

                riskLabel.className = "high-risk";

            } else {

                riskLabel.textContent = "🟢 LOW RISK";

                riskLabel.className = "low-risk";
            }


            // =================================================
            // SHOW PROBABILITIES
            // =================================================

            if (lowRisk) {

                lowRisk.textContent =
                    Number(
                        result.low_risk_probability
                    ).toFixed(2) + "%";
            }


            if (highRisk) {

                highRisk.textContent =
                    Number(
                        result.high_risk_probability
                    ).toFixed(2) + "%";
            }
            // Update Risk Score
            const riskScore = document.getElementById("riskScore");
            const riskLevel = document.getElementById("riskLevel");
            const riskMessage = document.getElementById("riskMessage");
            const riskMeterFill = document.getElementById("riskMeterFill");

            if (riskScore) {
                riskScore.textContent =
                    Number(result.risk_score).toFixed(2) + "%";
            }

            if (riskLevel) {

                const score = Number(result.risk_score);

                riskLevel.classList.remove(
                    "risk-medium",
                    "risk-high"
                );

                riskLevel.textContent =
                    result.risk_level + " RISK";

                if (score >= 60) {
                    riskLevel.classList.add("risk-high");
                }
                else if (score >= 30) {
                    riskLevel.classList.add("risk-medium");
                }
            }

            if (riskMessage) {
                riskMessage.textContent =
                    result.risk_message;
            }

            if (riskMeterFill) {

                const score = Number(result.risk_score);

                // Start from 0%
                riskMeterFill.style.width = "0%";

                // Animate to the actual risk score
                setTimeout(() => {
                    riskMeterFill.style.width = score + "%";
                }, 100);
            }


            // =================================================
            // SHOW EXPLAINABILITY
            // =================================================

            if (explainabilitySection) {

                explainabilitySection.style.display = "block";


                // =================================================
                // SHAP DATA
                // =================================================


                const shapHTML =
                    Array.isArray(result.shap_explanation) &&
                        result.shap_explanation.length > 0
                        ? (() => {
                            const items = result.shap_explanation
                                .map(item => ({
                                    feature: String(item.feature ?? "Unknown feature"),
                                    value: Number(item.shap_value)
                                }))
                                .filter(item => Number.isFinite(item.value));

                            const maxImpact = Math.max(
                                ...items.map(item => Math.abs(item.value)),
                                0.000001
                            );

                            return items.map(item => {
                                const value = item.value;
                                const increasing = value >= 0;
                                const width = Math.max(
                                    3,
                                    (Math.abs(value) / maxImpact) * 100
                                );

                                const direction = increasing
                                    ? "Increases risk"
                                    : "Decreases risk";

                                const colorClass = increasing
                                    ? "impact-increase"
                                    : "impact-decrease";

                                return `
                    <details class="impact-card ${colorClass}">
                        <summary class="impact-summary">
                            <div class="impact-topline">
                                <span class="impact-feature">
                                    ${item.feature.replace(/&/g, "&amp;")
                                        .replace(/</g, "&lt;")
                                        .replace(/>/g, "&gt;")
                                        .replace(/"/g, "&quot;")}
                                </span>
                                <span class="impact-value">
                                    ${value > 0 ? "+" : ""}${value.toFixed(3)}
                                </span>
                            </div>

                            <div class="impact-track">
                                <div
                                    class="impact-fill"
                                    style="width:${width}%"
                                ></div>
                            </div>

                            <span class="impact-direction">
                                ${increasing ? "↑" : "↓"} ${direction}
                            </span>
                        </summary>

                        <div class="impact-details">
                            This feature's SHAP value for this prediction is
                            ${value.toFixed(4)}. Its sign indicates the
                            direction of its contribution in the model's
                            explanation.
                        </div>
                    </details>
                `;
                            }).join("");
                        })()
                        : `<p class="no-explanation">No valid SHAP explanation is available.</p>`;



                // =================================================
                // LIME DATA
                // =================================================


                const limeHTML =
                    Array.isArray(result.lime_explanation) &&
                        result.lime_explanation.length > 0
                        ? (() => {
                            const items = result.lime_explanation
                                .map(item => ({
                                    feature: String(item.feature ?? "Unknown feature"),
                                    value: Number(item.weight)
                                }))
                                .filter(item => Number.isFinite(item.value));

                            const maxImpact = Math.max(
                                ...items.map(item => Math.abs(item.value)),
                                0.000001
                            );

                            return items.map(item => {
                                const value = item.value;
                                const increasing = value >= 0;

                                const width = Math.max(
                                    3,
                                    (Math.abs(value) / maxImpact) * 100
                                );

                                const direction = increasing
                                    ? "Increases risk"
                                    : "Decreases risk";

                                const colorClass = increasing
                                    ? "impact-increase"
                                    : "impact-decrease";

                                const safeFeature = item.feature
                                    .replace(/&/g, "&amp;")
                                    .replace(/</g, "&lt;")
                                    .replace(/>/g, "&gt;")
                                    .replace(/"/g, "&quot;");

                                return `
                    <details class="impact-card ${colorClass}">
                        <summary class="impact-summary">
                            <div class="impact-topline">
                                <span class="impact-feature">
                                    ${safeFeature}
                                </span>

                                <span class="impact-value">
                                    ${value > 0 ? "+" : ""}${value.toFixed(3)}
                                </span>
                            </div>

                            <div class="impact-track">
                                <div
                                    class="impact-fill"
                                    style="width:${width}%"
                                ></div>
                            </div>

                            <span class="impact-direction">
                                ${increasing ? "↑" : "↓"} ${direction}
                            </span>
                        </summary>

                        <div class="impact-details">
                            This LIME weight is ${value.toFixed(4)} for this
                            feature's local explanation. Its sign indicates
                            the direction of the contribution according to
                            the model's local explanation.
                        </div>
                    </details>
                `;
                            }).join("");
                        })()
                        : `<p class="no-explanation">No valid LIME explanation is available.</p>`;



                // =================================================
                // DISPLAY SHAP + LIME
                // =================================================


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

        <div class="shap-list">
            ${shapHTML}
        </div>

        <div class="explanation-heading lime-heading">
            <span class="explanation-icon">🔎</span>
            <div>
                <h3>LIME Explanation</h3>
                <p>Explore the rules behind this individual prediction.</p>
            </div>
        </div>

        <div class="lime-list">
            ${limeHTML}
        </div>

        <p class="explanation-note">
            These explanations describe how the model reached its prediction;
            they do not guarantee that a loan will default or be repaid.
        </p>

    </div>
`;

            }


            // =================================================
            // SCROLL TO RESULT
            // =================================================

            if (resultSection) {

                resultSection.scrollIntoView({

                    behavior: "smooth",

                    block: "start"

                });
            }


        } catch (error) {

            // =================================================
            // ERROR HANDLING
            // =================================================

            console.error(
                "Prediction error:",
                error
            );

            alert(
                "Prediction failed: " +
                error.message
            );

        }


        // =================================================
        // RESET BUTTON
        // =================================================

        predictButton.disabled = false;

        predictButton.innerHTML =
            "<span>✦</span> Predict Credit Risk <span>→</span>";

    });

});