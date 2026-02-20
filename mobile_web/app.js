const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const searchInput = document.getElementById("searchInput");
const results = document.getElementById("results");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const originalScoreEl = document.getElementById("originalScore");
const healthiestScoreEl = document.getElementById("healthiestScore");
const cheapestCostEl = document.getElementById("cheapestCost");
const balancedScoreEl = document.getElementById("balancedScore");
const downloadLink = document.getElementById("downloadLink");
const categoryPanel = document.getElementById("categoryPanel");
const categoryList = document.getElementById("categoryList");
const runBtn = document.getElementById("runBtn");
const runRecipeBtn = document.getElementById("runRecipeBtn");
const componentsEl = document.getElementById("components");
const recipeEl = document.getElementById("recipe");
const recipeNameEl = document.getElementById("recipeName");
const recipeInstructionsEl = document.getElementById("recipeInstructions");
const recipeIngredientsEl = document.getElementById("recipeIngredients");
const stickyEl = document.getElementById("sticky");
const stickyCostEl = document.getElementById("stickyCost");
const stickyCostDeltaEl = document.getElementById("stickyCostDelta");
const stickyHealthEl = document.getElementById("stickyHealth");
const stickyHealthDeltaEl = document.getElementById("stickyHealthDelta");
const stickyComponentsEl = document.getElementById("stickyComponents");
const stickyNoteEl = document.getElementById("stickyNote");
const focusViewEl = document.getElementById("focusView");
const focusBodyEl = document.getElementById("focusBody");
const focusTitleEl = document.getElementById("focusTitle");
const focusSubtitleEl = document.getElementById("focusSubtitle");
const focusProgressEl = document.getElementById("focusProgress");
const navTabs = document.querySelectorAll(".top-nav__tab");

let rows = [];
let initialRows = [];
let uploadedFile = null;
let baseTotals = { cost: 0, score: 0 };
let initialTotals = { cost: 0, score: 0 };
let baseComponents = {};
let initialComponents = {};
let currentSelection = null; // { rowIndex, optionKey, ghostTotals }
let currentView = "list"; // list | focus
let currentIndex = 0;

const COMPONENT_LABELS = {
  HEI2015C1_TOTALVEG: "Vegetables",
  HEI2015C3_TOTALFRUIT: "Fruits",
  HEI2015C5_WHOLEGRAIN: "Grain",
  HEI2015C6_TOTALDAIRY: "Dairy",
  HEI2015C7_TOTPROT: "Protein",
  HEI2015C9_FATTYACID: "Unsaturated Fat",
  HEI2015C10_SODIUM: "Sodium",
  HEI2015C11_REFINEDGRAIN: "Refined Grains",
  HEI2015C12_SFAT: "Saturated Fat",
  HEI2015C13_ADDSUG: "Sugar",
};

const COMPONENT_ORDER = [
  "HEI2015C1_TOTALVEG",
  "HEI2015C3_TOTALFRUIT",
  "HEI2015C5_WHOLEGRAIN",
  "HEI2015C6_TOTALDAIRY",
  "HEI2015C7_TOTPROT",
  "HEI2015C9_FATTYACID",
  "HEI2015C10_SODIUM",
  "HEI2015C11_REFINEDGRAIN",
  "HEI2015C12_SFAT",
  "HEI2015C13_ADDSUG",
];

const MODERATION_COMPONENTS = new Set([]);

const SLIDER_ORDER = [
  "Keyword",
  "wweia_food_category_description",
  "Level_2_Category",
  "Level_1_Category",
];

const OPTION_ICONS = {
  balanced: "✨",
  healthiest: "🌿",
  cheapest: "💳",
};

const DEFAULT_OPTION_LABELS = {
  balanced: "Balanced",
  healthiest: "Healthier",
  cheapest: "Cheaper",
};

const OPTION_LABEL_ORDER = ["Healthier", "Balanced", "Cheaper"];

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b44b3d" : "";
};

const safeImage = (url, alt) => {
  if (!url) return "";
  return `<img src="${url}" alt="${alt}" />`;
};

const formatPrice = (value) => {
  const num = Number(value);
  if (Number.isFinite(num)) {
    return `$${num.toFixed(2)}`;
  }
  return value || "-";
};

const formatScore = (value) => {
  const num = Number(value);
  if (Number.isFinite(num)) {
    return num.toFixed(1);
  }
  return "-";
};

const buildChips = (components = {}) => {
  const chips = COMPONENT_ORDER.filter((key) => Number(components[key] || 0) > 0).map((key) => {
    const value = Number(components[key]).toFixed(1);
    return `<span class="chip">${COMPONENT_LABELS[key]} ${value}</span>`;
  });
  if (!chips.length) {
    return "";
  }
  return `<div class="chip-group">${chips.join("")}</div>`;
};

const sumComponents = (items) => {
  const totals = {};
  COMPONENT_ORDER.forEach((key) => {
    totals[key] = 0;
  });
  (items || []).forEach((row) => {
    const components = row.original?.components || {};
    COMPONENT_ORDER.forEach((key) => {
      totals[key] += Number(components[key] || 0);
    });
  });
  return totals;
};

const applyComponentDelta = (base, from, to) => {
  const next = { ...base };
  COMPONENT_ORDER.forEach((key) => {
    next[key] = Number(base[key] || 0) + (Number(to?.[key] || 0) - Number(from?.[key] || 0));
  });
  return next;
};

const isDeltaGoodSticky = (_key, delta) => delta >= 0;

const isDeltaGoodSimple = (delta) => delta > 0;

const buildDeltaChips = (from = {}, to = {}) => {
  const chips = COMPONENT_ORDER.map((key) => {
    const delta = Number(to[key] || 0) - Number(from[key] || 0);
    if (Math.abs(delta) < 0.01) {
      return null;
    }
    const arrow = delta > 0 ? "↗" : "↘";
    const className = isDeltaGoodSimple(delta) ? "is-good" : "is-bad";
    return `<span class="chip chip--delta ${className}">${COMPONENT_LABELS[key]} ${arrow}${Math.abs(delta).toFixed(1)}</span>`;
  }).filter(Boolean);
  if (!chips.length) {
    return "";
  }
  return `<div class="chip-group chip-group--compact">${chips.join("")}</div>`;
};

const buildStickyComponents = (current = {}, base = {}, showDelta = false) => {
  const chips = COMPONENT_ORDER.map((key) => {
    const value = Number(current[key] || 0);
    if (!showDelta && value <= 0) {
      return null;
    }
    const delta = value - Number(base[key] || 0);
    if (showDelta && Math.abs(delta) < 0.01) {
      return null;
    }
    const arrow = delta > 0 ? "↗" : delta < 0 ? "↘" : "→";
    const className = showDelta ? (isDeltaGoodSticky(key, delta) ? "is-good" : "is-bad") : "is-neutral";
    const label = showDelta
      ? `${COMPONENT_LABELS[key]} ${value.toFixed(1)} ${arrow}${Math.abs(delta).toFixed(1)}`
      : `${COMPONENT_LABELS[key]} ${value.toFixed(1)}`;
    return `<span class="chip chip--delta ${className}">${label}</span>`;
  }).filter(Boolean);

  if (!chips.length) {
    return "";
  }
  return `<div class="chip-group chip-group--compact">${chips.join("")}</div>`;
};

const getOutcomeLabels = (row) => {
  const labels = { ...DEFAULT_OPTION_LABELS };
  if (!row?.options) return labels;

  const candidates = ["balanced", "healthiest", "cheapest"]
    .map((key) => ({
      key,
      data: row.options[key],
    }))
    .filter((item) => item.data && item.data.name);

  if (candidates.length < 2) {
    return labels;
  }

  candidates.forEach((item) => {
    item.deltaScore =
      Number(item.data.total_score_after ?? item.data.hei ?? 0) - Number(baseTotals.score || 0);
    item.deltaCost =
      Number(item.data.total_cost_after ?? item.data.price ?? 0) - Number(baseTotals.cost || 0);
  });

  candidates.sort((a, b) => {
    if (b.deltaScore !== a.deltaScore) return b.deltaScore - a.deltaScore;
    return a.deltaCost - b.deltaCost;
  });
  const healthiest = candidates.shift();
  if (healthiest) {
    labels[healthiest.key] = "Healthier";
  }

  if (candidates.length) {
    candidates.sort((a, b) => {
      if (a.deltaCost !== b.deltaCost) return a.deltaCost - b.deltaCost;
      return b.deltaScore - a.deltaScore;
    });
    const cheapest = candidates.shift();
    if (cheapest) {
      labels[cheapest.key] = "Cheaper";
    }
  }

  candidates.forEach((item) => {
    labels[item.key] = "Balanced";
  });

  return labels;
};

const getOutcomeOptions = (row) => {
  const labels = getOutcomeLabels(row);
  const items = ["healthiest", "balanced", "cheapest"].map((key) => ({
    key,
    label: labels[key] || DEFAULT_OPTION_LABELS[key],
    option: row.options?.[key],
  }));
  items.sort((a, b) => OPTION_LABEL_ORDER.indexOf(a.label) - OPTION_LABEL_ORDER.indexOf(b.label));
  return items;
};

const updateSticky = (ghostTotals = null) => {
  if (!stickyEl) return;
  stickyEl.hidden = rows.length === 0;

  if (!ghostTotals) {
    stickyCostEl.textContent = formatPrice(baseTotals.cost);
    stickyHealthEl.textContent = formatScore(baseTotals.score);
    stickyCostDeltaEl.textContent = "";
    stickyHealthDeltaEl.textContent = "";
    stickyNoteEl.textContent = "Live impact";
    stickyCostDeltaEl.className = "sticky__delta";
    stickyHealthDeltaEl.className = "sticky__delta";
    if (stickyComponentsEl) {
      stickyComponentsEl.innerHTML = buildStickyComponents(baseComponents, baseComponents, false);
    }
    return;
  }

  const costDelta = ghostTotals.cost - baseTotals.cost;
  const healthDelta = ghostTotals.score - baseTotals.score;
  const ghostComponents = ghostTotals.components || baseComponents;

  stickyCostEl.textContent = formatPrice(ghostTotals.cost);
  stickyHealthEl.textContent = formatScore(ghostTotals.score);

  const costArrow = costDelta > 0 ? "↗" : costDelta < 0 ? "↘" : "→";
  const healthArrow = healthDelta > 0 ? "↗" : healthDelta < 0 ? "↘" : "→";

  stickyCostDeltaEl.textContent = `${costArrow} ${formatPrice(Math.abs(costDelta))}`;
  stickyHealthDeltaEl.textContent = `${healthArrow} ${formatScore(Math.abs(healthDelta))}`;

  stickyCostDeltaEl.className = `sticky__delta ${costDelta <= 0 ? "is-good" : "is-bad"}`;
  stickyHealthDeltaEl.className = `sticky__delta ${healthDelta >= 0 ? "is-good" : "is-bad"}`;
  stickyNoteEl.textContent = "Showing trade-offs";
  if (stickyComponentsEl) {
    stickyComponentsEl.innerHTML = buildStickyComponents(ghostComponents, baseComponents, true);
  }
};

const renderOptionCard = (rowIndex, label, optionKey, option, original) => {
  const optionKeyLower = optionKey.toLowerCase();
  const safeOption = option || {};
  const originalName = original?.name || "";
  const normalizedOriginal = originalName.toLowerCase().trim();
  const normalizedOption = (safeOption.name || "").toLowerCase().trim();
  const isSame = normalizedOriginal && normalizedOption && normalizedOriginal === normalizedOption;
  const selected = currentSelection && currentSelection.rowIndex === rowIndex && currentSelection.optionKey === optionKeyLower;

  if (isSame) {
    return `
      <div class="option-card option-card--good">
        <div>
          <div class="option-card__tag">
            <span class="option-card__icon">${OPTION_ICONS[optionKeyLower] || "✨"}</span>
            ${label}
          </div>
          <div class="option-card__good">This was a good choice!</div>
        </div>
      </div>
    `;
  }

  const comparisonComponents = safeOption.total_components_after || safeOption.components || {};
  const chips = buildDeltaChips(baseComponents, comparisonComponents);
  const deltaPrice = Number(safeOption.total_cost_after ?? safeOption.price ?? 0) - Number(baseTotals.cost || 0);
  const deltaScore = Number(safeOption.total_score_after ?? safeOption.hei ?? 0) - Number(baseTotals.score || 0);
  const priceLabel = `${deltaPrice >= 0 ? "+" : "-"}${formatPrice(Math.abs(deltaPrice))}`;
  const scoreLabel = `${deltaScore >= 0 ? "+" : "-"}${formatScore(Math.abs(deltaScore))} HEI`;
  const priceClass = deltaPrice <= 0 ? "is-good" : "is-bad";
  const scoreClass = deltaScore >= 0 ? "is-good" : "is-bad";

  return `
    <div class="option-card ${selected ? "is-selected" : ""}" data-action="select" data-row="${rowIndex}" data-option="${optionKeyLower}">
      <div class="option-card__media">
        ${safeImage(safeOption.image, safeOption.name || "Recommended product")}
      </div>
      <div>
        <div class="option-card__tag">
          <span class="option-card__icon">${OPTION_ICONS[optionKeyLower] || "✨"}</span>
          ${label}
        </div>
        <div class="option-card__name">${safeOption.name || "(missing)"}</div>
      </div>
      <div class="option-card__meta">
        <div class="option-card__price ${priceClass}">${priceLabel}</div>
        <div class="option-card__score ${scoreClass}">${scoreLabel}</div>
        ${chips}
      </div>
      ${selected ? `<button class="option-card__confirm" data-action="confirm" data-row="${rowIndex}" data-option="${optionKeyLower}">Confirm Swap</button>` : ""}
    </div>
  `;
};

const renderCards = (filtered) => {
  results.innerHTML = filtered
    .map(({ row, index }) => {
      const original = row.original || {};
      const originalName = original.name || "";
      const originalChips = buildChips(original.components);
      const outcomeOptions = getOutcomeOptions(row);

      const targetLabel = row.target_category || row.target_category_column || "";

      return `
        <article class="product-card">
          <div class="product-card__head">
            <div class="product-card__anchor">
              ${safeImage(original.image, originalName || "Original product")}
              <div>
                <div class="anchor__label">Original</div>
                <div class="anchor__name">${originalName || "(missing)"}</div>
                <div class="anchor__price">${formatPrice(original.price)}</div>
                ${originalChips}
              </div>
            </div>
            <div class="anchor__category">${targetLabel}</div>
          </div>
          <div class="product-card__options">
            <div class="options__title">Better alternatives</div>
            ${outcomeOptions
              .map((item) => renderOptionCard(index, item.label, item.key, item.option, original))
              .join("")}
          </div>
        </article>
      `;
    })
    .join("");
};

const renderFocus = () => {
  if (!focusViewEl) return;
  if (!rows.length) {
    focusViewEl.classList.remove("is-summary");
    focusTitleEl.textContent = "Focus mode";
    focusSubtitleEl.textContent = "Upload and run recommendations to begin.";
    focusProgressEl.textContent = "0%";
    focusBodyEl.innerHTML = `<div class="focus-empty">Upload a cart and run recommendations to start reviewing items.</div>`;
    return;
  }

  if (currentIndex >= rows.length) {
    focusViewEl.classList.add("is-summary");
    focusTitleEl.textContent = "Selection Complete!";
    focusSubtitleEl.textContent = "";
    focusProgressEl.textContent = "100%";

    const listItems = rows
      .map((row) => {
        return `<li><span>${row.original.name}</span><span>${formatPrice(row.original.price)}</span></li>`;
      })
      .join("");

    focusBodyEl.innerHTML = `
      <div class="summary-card">
        <div class="summary-card__icon">✓</div>
        <h3>Selection Complete!</h3>
        <p class="muted">Here is the summary of your curated basket.</p>
        <div class="summary-card__stats">
          <div>
            <p class="summary__label">Total Cost</p>
            <p class="summary__value">${formatPrice(baseTotals.cost)}</p>
          </div>
          <div>
            <p class="summary__label">Nutrition Score</p>
            <p class="summary__value">${formatScore(baseTotals.score)}</p>
          </div>
        </div>
        <ul class="summary-card__list">${listItems}</ul>
        <button class="run-btn" id="restartBtn">Start Over</button>
      </div>
    `;

    const restartBtn = document.getElementById("restartBtn");
    if (restartBtn) {
      restartBtn.addEventListener("click", () => {
        rows = JSON.parse(JSON.stringify(initialRows));
        baseTotals = { ...initialTotals };
        baseComponents = { ...initialComponents };
        currentIndex = 0;
        currentSelection = null;
        render();
      });
    }
    return;
  }

  focusViewEl.classList.remove("is-summary");

  const row = rows[currentIndex];
  const original = row.original;
  const progress = Math.round(((currentIndex + 1) / rows.length) * 100);
  const focusLabel = row.target_category || row.target_category_column || "item";
  focusTitleEl.textContent = `Reviewing ${focusLabel}`;
  focusSubtitleEl.textContent = `Item ${currentIndex + 1} of ${rows.length}`;
  focusProgressEl.textContent = `${progress}%`;

  const originalChips = buildChips(original.components);
  const outcomeOptions = getOutcomeOptions(row);

  const buildOption = (label, optionKey, option) => {
    const lower = optionKey.toLowerCase();
    if (!option) {
      return `
        <div class="swap-card swap-card--good">
          <div class="swap-card__icon">•</div>
          <div class="swap-card__label">${label}</div>
          <div class="swap-card__name">No option available</div>
        </div>
      `;
    }
    const selected = currentSelection && currentSelection.rowIndex === currentIndex && currentSelection.optionKey === lower;
    const comparisonComponents = option.total_components_after || option.components || {};
    const deltaChips = buildDeltaChips(baseComponents, comparisonComponents);
    const deltaCost = Number(option.total_cost_after ?? option.price ?? 0) - Number(baseTotals.cost || 0);
    const deltaScore = Number(option.total_score_after ?? option.hei ?? 0) - Number(baseTotals.score || 0);
    const costLabel = `${deltaCost >= 0 ? "+" : "-"}${formatPrice(Math.abs(deltaCost))}`;
    const scoreLabel = `${deltaScore >= 0 ? "+" : "-"}${formatScore(Math.abs(deltaScore))} HEI`;
    const normalizedOriginal = (original.name || "").toLowerCase().trim();
    const normalizedOption = (option.name || "").toLowerCase().trim();
    const isSame = normalizedOriginal && normalizedOption && normalizedOriginal === normalizedOption;

    if (isSame) {
      return `
        <div class="swap-card swap-card--good">
          <div class="swap-card__icon">✓</div>
          <div class="swap-card__label">${label}</div>
          <div class="swap-card__name">This was a good choice!</div>
        </div>
      `;
    }

    return `
      <div class="swap-card ${selected ? "is-selected" : ""}" data-action="select" data-row="${currentIndex}" data-option="${lower}">
        <div class="swap-card__icon">${OPTION_ICONS[lower] || "✨"}</div>
        <div class="swap-card__media">
          ${safeImage(option.image, option.name || "Recommended product")}
        </div>
        <div class="swap-card__label">${label}</div>
        <div class="swap-card__name">${option.name || "(missing)"}</div>
        <div class="swap-card__delta ${deltaCost <= 0 ? "is-good" : "is-bad"}">${costLabel}</div>
        <div class="swap-card__delta ${deltaScore >= 0 ? "is-good" : "is-bad"}">${scoreLabel}</div>
        ${deltaChips}
        ${selected ? `<button class="swap-card__confirm" data-action="confirm" data-row="${currentIndex}" data-option="${lower}">Confirm Swap</button>` : ""}
      </div>
    `;
  };

  focusBodyEl.innerHTML = `
    <div class="hero-card">
      <div class="hero-card__title">Currently selected</div>
      <div class="hero-card__content">
        ${safeImage(original.image, original.name)}
        <div>
          <div class="hero-card__name">${original.name}</div>
          <div class="hero-card__price">${formatPrice(original.price)}</div>
          ${originalChips}
        </div>
      </div>
      <button class="hero-card__keep" data-action="keep">Keep This Item</button>
    </div>
    <div class="swap-divider">OR SWAP FOR</div>
    <div class="swap-grid">
      ${outcomeOptions
        .map((item) => buildOption(item.label, item.key, item.option))
        .join("")}
    </div>
  `;
};

const render = () => {
  const query = searchInput.value.trim().toLowerCase();
  const filtered = rows
    .map((row, index) => ({ row, index }))
    .filter(({ row }) => {
      const originalName = row.original?.name?.toLowerCase() || "";
      const options = row.options || {};
      return (
        originalName.includes(query) ||
        (options.healthiest?.name || "").toLowerCase().includes(query) ||
        (options.cheapest?.name || "").toLowerCase().includes(query) ||
        (options.balanced?.name || "").toLowerCase().includes(query) ||
        (row.target_category || "").toLowerCase().includes(query)
      );
    });

  renderCards(filtered);
  renderFocus();
  updateSticky(currentSelection ? currentSelection.ghostTotals : null);
};

const renderRecipe = (recipeData, recipeInfo) => {
  if (!recipeData && !recipeInfo) {
    recipeEl.hidden = true;
    return;
  }

  if (recipeData) {
    recipeNameEl.textContent = recipeData.recipe_name || "";
    recipeInstructionsEl.textContent = (recipeData.instructions || "").replace(/\\n/g, "\n");
    recipeIngredientsEl.innerHTML = "";
    (recipeData.missing_ingredients || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item.name || "Ingredient";
      recipeIngredientsEl.appendChild(li);
    });
  } else {
    recipeNameEl.textContent = "";
    recipeInstructionsEl.textContent = recipeInfo || "";
    recipeIngredientsEl.innerHTML = "";
  }

  recipeEl.hidden = false;
};

const renderComponents = () => {
  if (componentsEl) {
    componentsEl.hidden = true;
    componentsEl.innerHTML = "";
  }
};

const updateSliderLabels = (labelsContainer, index) => {
  labelsContainer.querySelectorAll("span").forEach((label, idx) => {
    if (idx === index) {
      label.classList.add("is-active");
    } else {
      label.classList.remove("is-active");
    }
  });
};

const buildCategoryOptions = (items) => {
  categoryList.innerHTML = "";

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "category-row";

    const values = item.values || {};

    const labels = document.createElement("div");
    labels.className = "slider-labels";
    SLIDER_ORDER.forEach((col) => {
      const span = document.createElement("span");
      span.textContent = values[col] || "—";
      labels.appendChild(span);
    });

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = 0;
    slider.max = SLIDER_ORDER.length - 1;
    slider.step = 1;
    slider.value = Number.isFinite(item.default_index) ? item.default_index : 0;
    slider.dataset.index = item.index;
    slider.dataset.values = JSON.stringify(values);

    updateSliderLabels(labels, Number(slider.value));

    slider.addEventListener("input", () => {
      updateSliderLabels(labels, Number(slider.value));
    });

    row.innerHTML = `<strong>${item.product_name}</strong>`;
    row.appendChild(labels);
    row.appendChild(slider);
    categoryList.appendChild(row);
  });

  categoryPanel.hidden = false;
};

const parseJsonResponse = async (res) => {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    return { error: text || "Unexpected server response." };
  }
};

const fetchOptions = (file) => {
  const formData = new FormData();
  formData.append("diet", file);

  setStatus("Loading category options…");

  fetch("/api/options", {
    method: "POST",
    body: formData,
  })
    .then(parseJsonResponse)
    .then((data) => {
      if (data.error) {
        setStatus(data.error, true);
        return;
      }
      buildCategoryOptions(data.items || []);
      setStatus("Select categories, then run recommendations.");
    })
    .catch((err) => {
      setStatus(`Request failed: ${err.message}`, true);
    });
};

const runRecommendations = (useOpenAI = false) => {
  if (!uploadedFile) return;

  const overrides = [];
  categoryList.querySelectorAll("input[type=range]").forEach((slider) => {
    const values = JSON.parse(slider.dataset.values || "{}");
    const idx = Number(slider.value);
    const column = SLIDER_ORDER[idx];
    const value = values[column];
    if (!value) {
      return;
    }
    overrides.push({
      index: Number(slider.dataset.index),
      column,
      value,
      description: "",
    });
  });

  const formData = new FormData();
  formData.append("diet", uploadedFile);
  if (overrides.length) {
    formData.append("overrides", JSON.stringify(overrides));
  }
  if (useOpenAI) {
    formData.append("use_openai", "1");
  }

  setStatus("Running optimizer… this can take a minute.");
  recipeEl.hidden = true;

  fetch("/api/recommend", {
    method: "POST",
    body: formData,
  })
    .then(parseJsonResponse)
    .then((data) => {
      if (data.error) {
        setStatus(data.error, true);
        return;
      }

      rows = data.rows || [];
      initialRows = JSON.parse(JSON.stringify(rows));
      const summary = data.summary || {};
      const summaryCost = Number(summary.original_cost);
      const baseComponentsData = data.base_components || {};
      const baseScoreValue = Number(baseComponentsData.HEI2015_TOTAL_SCORE ?? summary.original_score);
      baseTotals = {
        cost: Number.isFinite(summaryCost)
          ? summaryCost
          : rows.reduce((sum, row) => sum + Number(row.original?.price || 0), 0),
        score: Number.isFinite(baseScoreValue)
          ? baseScoreValue
          : rows.reduce((sum, row) => sum + Number(row.original?.hei || 0), 0),
      };
      initialTotals = { ...baseTotals };
      baseComponents = baseComponentsData;
      if (!Object.keys(baseComponents).length) {
        baseComponents = sumComponents(rows);
      }
      initialComponents = { ...baseComponents };

      currentSelection = null;
      currentIndex = 0;

      originalScoreEl.textContent = formatScore(summary.original_score);
      healthiestScoreEl.textContent = formatScore(summary.healthiest_score);
      cheapestCostEl.textContent = formatPrice(summary.cheapest_cost);
      balancedScoreEl.textContent = formatScore(summary.balanced_score);
      summaryEl.hidden = false;

      renderComponents(data.original_components, data.recommended_components);
      renderRecipe(data.recipe_data, data.recipe_info);

      if (data.csv) {
        const blob = new Blob([data.csv], { type: "text/csv" });
        downloadLink.href = URL.createObjectURL(blob);
      }

      setStatus(`Loaded ${rows.length} recommendations.`);
      render();
    })
    .catch((err) => {
      setStatus(`Request failed: ${err.message}`, true);
    });
};

const handleOptionSelect = (rowIndex, optionKey) => {
  const row = rows[rowIndex];
  if (!row) return;
  const option = row.options[optionKey.toLowerCase()];
  if (!option) return;

  if (currentSelection && currentSelection.rowIndex === rowIndex && currentSelection.optionKey === optionKey.toLowerCase()) {
    currentSelection = null;
    render();
    return;
  }

  const ghostTotals = {
    cost: Number(option.total_cost_after ?? baseTotals.cost + (option.price - row.original.price)),
    score: Number(option.total_score_after ?? baseTotals.score + (option.hei - row.original.hei)),
    components: option.total_components_after || applyComponentDelta(baseComponents, row.original.components, option.components),
  };

  currentSelection = {
    rowIndex,
    optionKey: optionKey.toLowerCase(),
    ghostTotals,
  };
  render();
};

const handleConfirm = (rowIndex, optionKey) => {
  const row = rows[rowIndex];
  if (!row) return;
  const key = optionKey.toLowerCase();
  const option = row.options[key];
  if (!option) return;

  const prevTotals = { ...baseTotals };
  const prevComponents = { ...baseComponents };

  const oldOriginal = row.original;
  row.original = {
    name: option.name,
    price: option.price,
    image: option.image,
    components: option.components,
    hei: option.hei,
  };

  row.options[key] = {
    name: oldOriginal.name,
    price: oldOriginal.price,
    image: oldOriginal.image,
    components: oldOriginal.components,
    hei: oldOriginal.hei,
    total_cost_after: prevTotals.cost,
    total_score_after: prevTotals.score,
    total_components_after: prevComponents,
  };

  baseTotals = {
    cost: Number(option.total_cost_after ?? baseTotals.cost + (option.price - oldOriginal.price)),
    score: Number(option.total_score_after ?? baseTotals.score + (option.hei - oldOriginal.hei)),
  };
  baseComponents = option.total_components_after || applyComponentDelta(baseComponents, oldOriginal.components, option.components);

  currentSelection = null;
  stickyEl.classList.remove("flash");
  void stickyEl.offsetWidth;
  stickyEl.classList.add("flash");

  if (currentView === "focus") {
    currentIndex += 1;
  }

  render();
};

const handleKeep = () => {
  currentSelection = null;
  if (currentView === "focus") {
    currentIndex += 1;
    render();
  }
};

const setView = (view) => {
  currentView = view;
  navTabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.view === view);
  });
  results.hidden = view !== "list";
  focusViewEl.hidden = view !== "focus";
  document.body.classList.toggle("is-focus", view === "focus");
  render();
};

const loadFile = (file) => {
  if (!file) return;
  uploadedFile = file;
  rows = [];
  initialRows = [];
  baseTotals = { cost: 0, score: 0 };
  baseComponents = {};
  initialTotals = { cost: 0, score: 0 };
  initialComponents = {};
  results.innerHTML = "";
  summaryEl.hidden = true;
  componentsEl.hidden = true;
  recipeEl.hidden = true;
  currentSelection = null;
  currentIndex = 0;
  fetchOptions(file);
};

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  loadFile(file);
});

runBtn.addEventListener("click", () => runRecommendations(false));
runRecipeBtn.addEventListener("click", () => runRecommendations(true));

searchInput.addEventListener("input", render);

results.addEventListener("click", (event) => {
  const confirmBtn = event.target.closest("[data-action='confirm']");
  if (confirmBtn) {
    const rowIndex = Number(confirmBtn.dataset.row);
    const optionKey = confirmBtn.dataset.option;
    handleConfirm(rowIndex, optionKey);
    return;
  }

  const optionCard = event.target.closest("[data-action='select']");
  if (optionCard) {
    const rowIndex = Number(optionCard.dataset.row);
    const optionKey = optionCard.dataset.option;
    handleOptionSelect(rowIndex, optionKey);
  }
});

focusBodyEl.addEventListener("click", (event) => {
  const keepBtn = event.target.closest("[data-action='keep']");
  if (keepBtn) {
    handleKeep();
    return;
  }

  const confirmBtn = event.target.closest("[data-action='confirm']");
  if (confirmBtn) {
    const rowIndex = Number(confirmBtn.dataset.row);
    const optionKey = confirmBtn.dataset.option;
    handleConfirm(rowIndex, optionKey);
    return;
  }

  const optionCard = event.target.closest("[data-action='select']");
  if (optionCard) {
    const rowIndex = Number(optionCard.dataset.row);
    const optionKey = optionCard.dataset.option;
    handleOptionSelect(rowIndex, optionKey);
  }
});

navTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setView(tab.dataset.view);
  });
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-active");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  fileInput.files = event.dataTransfer.files;
  loadFile(file);
});

setView("list");
