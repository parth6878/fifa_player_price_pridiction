/**
 * FIFA Player Price Predictor - Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // Form & Controls
    const form = document.getElementById("prediction-form");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnSpinner = submitBtn.querySelector(".btn-spinner");
    const alertBox = document.getElementById("alert-box");
    const alertMessage = document.getElementById("alert-message");

    // Display Elements
    const priceDisplay = document.getElementById("price-display");
    const valuationDetails = document.getElementById("valuation-details");
    const upsideVal = document.getElementById("upside-val");
    const tierVal = document.getElementById("tier-val");

    // Health Indicator
    const healthDot = document.getElementById("health-dot");
    const healthLabel = document.getElementById("health-label");

    // Live FUT Card Elements
    const cardOva = document.getElementById("card-ova");
    const cardPos = document.getElementById("card-pos");
    const cardFoot = document.getElementById("card-foot");
    const cardClub = document.getElementById("card-club");
    const cardAtt = document.getElementById("c-att");
    const cardSkl = document.getElementById("c-skl");
    const cardMov = document.getElementById("c-mov");
    const cardPow = document.getElementById("c-pow");
    const cardDef = document.getElementById("c-def");
    const cardPot = document.getElementById("c-pot");

    // Inputs
    const clubInput = document.getElementById("club-input");
    const posSelect = document.getElementById("position-select");
    const footSelect = document.getElementById("foot-select");

    // Tier colors, tied to the warm/light palette (kept in sync with style.css tokens)
    const TIER_COLORS = {
        superstar: "#8A5A16",   // --gold-deep
        topFlight: "#2E6B4E",   // --pitch
        regular: "#B3562B",     // warm bronze
        prospect: "#AC9C80"     // --ink-faint
    };

    // Stat mapping for sync: [inputId, sliderId, cardElem]
    const numericFields = [
        { id: "ova", card: cardOva },
        { id: "pot", card: cardPot },
        { id: "age", card: null },
        { id: "height", card: null },
        { id: "weight", card: null },
        { id: "attacking", card: cardAtt },
        { id: "skill", card: cardSkl },
        { id: "movement", card: cardMov },
        { id: "power", card: cardPow },
        { id: "mentality", card: null },
        { id: "defending", card: cardDef },
        { id: "goalkeeping", card: null }
    ];

    // -------------------------------------------------------------
    // 1. Sync Sliders and Number Boxes
    // -------------------------------------------------------------
    numericFields.forEach(field => {
        const numInput = document.getElementById(`input-${field.id}`);
        const slider = document.getElementById(`slider-${field.id}`);

        if (numInput && slider) {
            // Slider to Input
            slider.addEventListener("input", (e) => {
                numInput.value = e.target.value;
                if (field.card) {
                    field.card.textContent = e.target.value;
                }
            });

            // Input to Slider
            numInput.addEventListener("input", (e) => {
                let val = parseFloat(e.target.value);
                if (!isNaN(val)) {
                    slider.value = val;
                    if (field.card) {
                        field.card.textContent = val;
                    }
                }
            });
        }
    });

    // -------------------------------------------------------------
    // 2. Sync Profile Meta (Club, Pos, Foot) to Card Preview
    // -------------------------------------------------------------
    clubInput.addEventListener("change", (e) => {
        cardClub.textContent = e.target.value || "Free Agent";
    });

    posSelect.addEventListener("change", (e) => {
        cardPos.textContent = e.target.value;
    });

    footSelect.addEventListener("change", (e) => {
        cardFoot.textContent = e.target.value === "Left" ? "L" : "R";
    });

    // -------------------------------------------------------------
    // 3. Load Clubs from Backend API into the Club Dropdown
    // -------------------------------------------------------------
    async function loadClubs() {
        try {
            const res = await fetch("/api/clubs");
            if (res.ok) {
                const data = await res.json();
                clubInput.innerHTML = "";
                data.clubs.forEach(club => {
                    const opt = document.createElement("option");
                    opt.value = club;
                    opt.textContent = club;
                    clubInput.appendChild(opt);
                });
            } else {
                throw new Error(`Unexpected status ${res.status}`);
            }
        } catch (err) {
            console.warn("Could not load clubs list:", err);
            clubInput.innerHTML = '<option value="" disabled selected>Could not load clubs — refresh to retry</option>';
        }
    }

    // Ensures a preset's club exists as an option even if it wasn't
    // among the clubs the backend returned, so applying a preset never
    // leaves the dropdown empty.
    function ensureClubOption(clubName) {
        const exists = Array.from(clubInput.options).some(opt => opt.value === clubName);
        if (!exists) {
            const opt = document.createElement("option");
            opt.value = clubName;
            opt.textContent = clubName;
            clubInput.appendChild(opt);
        }
    }

    // -------------------------------------------------------------
    // 4. Live Model/Backend Health Check
    // -------------------------------------------------------------
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (res.ok && data.model_loaded) {
                healthDot.classList.remove("offline");
                healthLabel.textContent = "Valuation model is live";
            } else {
                healthDot.classList.add("offline");
                healthLabel.textContent = "Model unavailable — predictions disabled";
            }
        } catch (err) {
            healthDot.classList.add("offline");
            healthLabel.textContent = "Can't reach the backend";
        }
    }

    // -------------------------------------------------------------
    // 5. Presets
    // -------------------------------------------------------------
    const presets = {
        wonderkid: {
            Club: "FC Barcelona",
            "Best Position": "RW",
            "Preferred Foot": "Left",
            OVA: 86,
            POT: 95,
            Age: 18,
            Height: 180,
            Weight: 70,
            Attacking: 450,
            Skill: 430,
            Movement: 440,
            Power: 380,
            Mentality: 390,
            Defending: 120,
            Goalkeeping: 55
        },
        striker: {
            Club: "Manchester City",
            "Best Position": "ST",
            "Preferred Foot": "Left",
            OVA: 91,
            POT: 94,
            Age: 23,
            Height: 194,
            Weight: 88,
            Attacking: 440,
            Skill: 390,
            Movement: 410,
            Power: 450,
            Mentality: 395,
            Defending: 110,
            Goalkeeping: 50
        },
        playmaker: {
            Club: "Manchester City",
            "Best Position": "CAM",
            "Preferred Foot": "Right",
            OVA: 91,
            POT: 91,
            Age: 29,
            Height: 181,
            Weight: 70,
            Attacking: 415,
            Skill: 450,
            Movement: 390,
            Power: 405,
            Mentality: 420,
            Defending: 170,
            Goalkeeping: 56
        },
        goalkeeper: {
            Club: "Real Madrid",
            "Best Position": "GK",
            "Preferred Foot": "Left",
            OVA: 89,
            POT: 90,
            Age: 28,
            Height: 199,
            Weight: 96,
            Attacking: 95,
            Skill: 110,
            Movement: 290,
            Power: 260,
            Mentality: 150,
            Defending: 55,
            Goalkeeping: 435
        }
    };

    function applyPreset(presetKey) {
        const data = presets[presetKey];
        if (!data) return;

        ensureClubOption(data.Club);
        clubInput.value = data.Club;
        posSelect.value = data["Best Position"];
        footSelect.value = data["Preferred Foot"];

        // Update numeric inputs & sliders
        for (const [key, val] of Object.entries(data)) {
            const lowerKey = key.toLowerCase();
            const numInput = document.getElementById(`input-${lowerKey}`);
            const slider = document.getElementById(`slider-${lowerKey}`);
            if (numInput) numInput.value = val;
            if (slider) slider.value = val;
        }

        // Update preview card
        cardClub.textContent = data.Club;
        cardPos.textContent = data["Best Position"];
        cardFoot.textContent = data["Preferred Foot"] === "Left" ? "L" : "R";
        cardOva.textContent = data.OVA;
        cardPot.textContent = data.POT;
        cardAtt.textContent = data.Attacking;
        cardSkl.textContent = data.Skill;
        cardMov.textContent = data.Movement;
        cardPow.textContent = data.Power;
        cardDef.textContent = data.Defending;

        hideAlert();
    }

    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const presetKey = btn.getAttribute("data-preset");
            applyPreset(presetKey);
        });
    });

    // -------------------------------------------------------------
    // 6. Form Submission & Prediction
    // -------------------------------------------------------------
    function showAlert(msg) {
        alertMessage.textContent = msg;
        alertBox.classList.remove("hidden");
    }

    function hideAlert() {
        alertBox.classList.add("hidden");
        alertMessage.textContent = "";
    }

    function formatCurrency(val) {
        return new Intl.NumberFormat("en-US", {
            style: "decimal",
            maximumFractionDigits: 0
        }).format(val);
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideAlert();

        // Gather payload matching exact FEATURE_COLS
        const payload = {
            "Club": clubInput.value,
            "Best Position": posSelect.value,
            "Preferred Foot": footSelect.value,
            "OVA": parseFloat(document.getElementById("input-ova").value),
            "POT": parseFloat(document.getElementById("input-pot").value),
            "Age": parseFloat(document.getElementById("input-age").value),
            "Height": parseFloat(document.getElementById("input-height").value),
            "Weight": parseFloat(document.getElementById("input-weight").value),
            "Attacking": parseFloat(document.getElementById("input-attacking").value),
            "Skill": parseFloat(document.getElementById("input-skill").value),
            "Movement": parseFloat(document.getElementById("input-movement").value),
            "Power": parseFloat(document.getElementById("input-power").value),
            "Mentality": parseFloat(document.getElementById("input-mentality").value),
            "Defending": parseFloat(document.getElementById("input-defending").value),
            "Goalkeeping": parseFloat(document.getElementById("input-goalkeeping").value)
        };

        // Client-side quick check for a selected club
        if (!payload.Club) {
            showAlert("Please select a club for the player.");
            clubInput.focus();
            return;
        }

        // Show Loading State
        submitBtn.disabled = true;
        btnText.textContent = "Estimating value...";
        btnSpinner.classList.remove("hidden");
        priceDisplay.style.opacity = "0.5";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                // Friendly error message for 422, 503, or 500
                const errorText = result.error || `Server responded with status ${response.status}`;
                showAlert(errorText);
                priceDisplay.textContent = "--";
                valuationDetails.classList.add("hidden");
                return;
            }

            // Success Display
            const predictedValue = result.predicted_value_eur;
            priceDisplay.textContent = formatCurrency(predictedValue);
            priceDisplay.style.opacity = "1";

            // Update details
            const diff = payload.POT - payload.OVA;
            upsideVal.textContent = diff > 0 ? `+${diff} OVR` : `${diff} OVR`;

            if (predictedValue >= 70_000_000) {
                tierVal.textContent = "Superstar tier";
                tierVal.style.color = TIER_COLORS.superstar;
            } else if (predictedValue >= 30_000_000) {
                tierVal.textContent = "Top flight";
                tierVal.style.color = TIER_COLORS.topFlight;
            } else if (predictedValue >= 10_000_000) {
                tierVal.textContent = "First team regular";
                tierVal.style.color = TIER_COLORS.regular;
            } else {
                tierVal.textContent = "Squad player / prospect";
                tierVal.style.color = TIER_COLORS.prospect;
            }
            valuationDetails.classList.remove("hidden");

        } catch (err) {
            showAlert("Unable to connect to the prediction server. Please ensure the backend is running.");
            console.error("Prediction fetch error:", err);
            priceDisplay.textContent = "--";
            valuationDetails.classList.add("hidden");
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = "Calculate market valuation";
            btnSpinner.classList.add("hidden");
            priceDisplay.style.opacity = "1";
        }
    });

    // -------------------------------------------------------------
    // Initialize: load clubs, check health, then apply default preset
    // -------------------------------------------------------------
    (async () => {
        await loadClubs();
        checkHealth();
        applyPreset("wonderkid");
    })();
});