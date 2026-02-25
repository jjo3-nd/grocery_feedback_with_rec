const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const searchInput = document.getElementById("searchInput");
const results = document.getElementById("results");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const originalScoreEl = document.getElementById("originalScore");
const recommendedScoreEl = document.getElementById("recommendedScore");
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
const cuisineInputs = document.querySelectorAll("input[name='cuisine']");

let rows = [];
let uploadedFile = null;

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
  HEI2015_TOTAL_SCORE: "Total HEI score",
};

const COMPONENT_COLORS = {
  HEI2015C1_TOTALVEG: "#5bbf62",
  HEI2015C3_TOTALFRUIT: "#e44d4d",
  HEI2015C5_WHOLEGRAIN: "#f28b2c",
  HEI2015C6_TOTALDAIRY: "#3b84c5",
  HEI2015C7_TOTPROT: "#6b4ca5",
  HEI2015C9_FATTYACID: "#f2b632",
  HEI2015C10_SODIUM: "#46b2c3",
  HEI2015C11_REFINEDGRAIN: "#cc3aa3",
  HEI2015C12_SFAT: "#f2a93b",
  HEI2015C13_ADDSUG: "#8e63a8",
  HEI2015_TOTAL_SCORE: "#1c1b19",
};

const COMPONENT_MAX = {
  HEI2015C1_TOTALVEG: 5,
  HEI2015C3_TOTALFRUIT: 5,
  HEI2015C5_WHOLEGRAIN: 10,
  HEI2015C6_TOTALDAIRY: 10,
  HEI2015C7_TOTPROT: 5,
  HEI2015C9_FATTYACID: 10,
  HEI2015C10_SODIUM: 10,
  HEI2015C11_REFINEDGRAIN: 10,
  HEI2015C12_SFAT: 10,
  HEI2015C13_ADDSUG: 10,
  HEI2015_TOTAL_SCORE: 100,
};

const MYPLATE_KEYS = [
  "HEI2015C1_TOTALVEG",
  "HEI2015C3_TOTALFRUIT",
  "HEI2015C5_WHOLEGRAIN",
  "HEI2015C6_TOTALDAIRY",
  "HEI2015C7_TOTPROT",
];

const MODERATE_KEYS = [
  "HEI2015C9_FATTYACID",
  "HEI2015C10_SODIUM",
  "HEI2015C11_REFINEDGRAIN",
  "HEI2015C12_SFAT",
  "HEI2015C13_ADDSUG",
];

const ICONS = {
  HEI2015C1_TOTALVEG: "assets/icons/vegetables.png",
  HEI2015C3_TOTALFRUIT: "assets/icons/fruits.png",
  HEI2015C5_WHOLEGRAIN: "assets/icons/grains.png",
  HEI2015C6_TOTALDAIRY: "assets/icons/dairy.png",
  HEI2015C7_TOTPROT: "assets/icons/protein.png",
  HEI2015C9_FATTYACID: "assets/icons/unsaturated-fat.png",
  HEI2015C10_SODIUM: "assets/icons/sodium.png",
  HEI2015C11_REFINEDGRAIN: "assets/icons/refined-grains.png",
  HEI2015C12_SFAT: "assets/icons/saturated-fat.png",
  HEI2015C13_ADDSUG: "assets/icons/sugar.png",
};

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b44b3d" : "";
};

const getSelectedCuisine = () => {
  const selected = Array.from(cuisineInputs).find((input) => input.checked);
  return selected?.value || "";
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

const renderPaired = (filtered) => {
  results.innerHTML = "";
  filtered.forEach((row) => {
    const card = document.createElement("article");
    card.className = "card";

    const aiPrefix = "AI Rec:";
    const isAiRow = (row.Original_Food || "").startsWith(aiPrefix);
    const originalLabel = isAiRow ? "New product" : "Original";
    const originalName = isAiRow
      ? row.Original_Food.replace(aiPrefix, "").trim()
      : row.Original_Food;
    const newName = row.New_Food || "";
    const normalizedOriginal = (originalName || "").toLowerCase().trim();
    const normalizedNew = newName.toLowerCase().trim();
    const isSameProduct =
      normalizedOriginal &&
      normalizedNew &&
      normalizedOriginal === normalizedNew;
    const showRecommended = !isSameProduct;

    const recommendedBlock = showRecommended
      ? `
        <div class="card__item">
          ${safeImage(row.New_Image_URL, row.New_Food || "Recommended product")}
          <div>
            <span class="card__label">Recommended</span>
            <div>${row.New_Food || "(missing)"}</div>
          </div>
        </div>
      `
      : "";

    const recommendationPill = showRecommended
      ? `<span class="card__pill">New: ${formatPrice(row.New_Price)}</span>`
      : `<span class="card__pill">This was a good choice!</span>`;
    const reasonText = row.Recommendation_Reason || "";
    const reasonBlock = reasonText
      ? `<div class="card__reason"><span class="reason-badge">Reason</span><span>${reasonText}</span></div>`
      : "";

    card.innerHTML = `
      <div class="card__pair">
        <div class="card__item">
          ${safeImage(row.Original_Image_URL, row.Original_Food || "Original product")}
          <div>
            <span class="card__label">${originalLabel}</span>
            <div>${originalName || "(missing)"}</div>
          </div>
        </div>
        ${recommendedBlock}
      </div>
      <div class="card__meta">
        <span class="card__pill">${row.Target_Category || "Category unknown"}</span>
        <span class="card__pill">Original: ${formatPrice(row.Original_Price)}</span>
        ${recommendationPill}
      </div>
      ${reasonBlock}
    `;
    results.appendChild(card);
  });
};

const buildSearchIndex = (row) => {
  const values = Object.values(row || {})
    .map((val) => {
      if (val === null || val === undefined) return "";
      if (typeof val === "string") return val;
      if (typeof val === "number") return String(val);
      return "";
    })
    .join(" ");
  return values.toLowerCase();
};

const render = () => {
  const query = searchInput.value.trim().toLowerCase();
  const filtered = rows.filter((row) => {
    if (!query) return true;
    return (row._search || "").includes(query);
  });

  renderPaired(filtered);
};

const renderComponentRow = (key, original, recommended) => {
  const max = COMPONENT_MAX[key] || 1;
  const before = Number(original?.[key] ?? 0);
  const after = Number(recommended?.[key] ?? 0);
  const beforePct = Math.min(100, Math.max(0, (before / max) * 100));
  const afterPct = Math.min(100, Math.max(0, (after / max) * 100));
  const color = COMPONENT_COLORS[key] || "#1c1b19";
  const iconPath = ICONS[key];
  const icon = iconPath ? `<img src="${iconPath}" alt="" />` : "";

  const left = Math.min(beforePct, afterPct);
  const width = Math.max(4, Math.abs(afterPct - beforePct));

  const row = document.createElement("div");
  row.className = "score-row";
  row.innerHTML = `
    <div class="score-row__label" style="color:${color}">
      <div class="score-row__icon">${icon}</div>
      <div>${COMPONENT_LABELS[key]}</div>
    </div>
    <div class="score-row__track">
      <div class="score-row__line"></div>
      <div class="score-row__segment" style="left:${left}%; width:${width}%; background:${color}"></div>
      <div class="score-row__dot is-before" style="left:${beforePct}%; border-color:${color}; color:${color}">
        <span>${beforePct.toFixed(0)}%</span>
      </div>
      <div class="score-row__dot is-after" style="left:${afterPct}%; background:${color}; color:${color}">
        <span>${afterPct.toFixed(0)}%</span>
      </div>
    </div>
  `;
  return row;
};

const renderComponents = (original, recommended) => {
  componentsEl.innerHTML = "";

  const plateBox = document.createElement("div");
  plateBox.className = "score-box";
  plateBox.innerHTML = `<h3>MyPlate Breakdown</h3>`;
  MYPLATE_KEYS.forEach((key) => {
    plateBox.appendChild(renderComponentRow(key, original, recommended));
  });

  const modBox = document.createElement("div");
  modBox.className = "score-box";
  modBox.innerHTML = `<h3>Components to Moderate</h3>`;
  MODERATE_KEYS.forEach((key) => {
    modBox.appendChild(renderComponentRow(key, original, recommended));
  });

  const totalBox = document.createElement("div");
  totalBox.className = "score-box";
  totalBox.innerHTML = `<h3>Total HEI Score</h3>`;
  totalBox.appendChild(
    renderComponentRow("HEI2015_TOTAL_SCORE", original, recommended),
  );

  componentsEl.appendChild(plateBox);
  componentsEl.appendChild(modBox);
  componentsEl.appendChild(totalBox);
  componentsEl.hidden = false;
};

const renderRecipe = (recipeData, recipeInfo) => {
  if (!recipeData && !recipeInfo) {
    recipeEl.hidden = true;
    return;
  }

  if (recipeData) {
    recipeNameEl.textContent = recipeData.recipe_name || "";
    const instructions = recipeData.instructions || "";
    const steps = instructions
      .split(/\n+/)
      .map((line) => line.replace(/^\s*[-*\d.)]+\s*/, "").trim())
      .filter((line) => line.length);
    if (steps.length) {
      recipeInstructionsEl.innerHTML = `<ol>${steps
        .map((step) => `<li>${step}</li>`)
        .join("")}</ol>`;
    } else {
      recipeInstructionsEl.textContent = instructions.replace(/\\n/g, "\n");
    }
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

const buildCategoryOptions = (items) => {
  categoryList.innerHTML = "";

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "category-row";

    const select = document.createElement("select");
    select.dataset.index = item.index;

    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Select a category";
    select.appendChild(defaultOption);

    item.options.forEach((option) => {
      const opt = document.createElement("option");
      opt.value = JSON.stringify({
        column: option.column,
        value: option.value,
        description: option.description || "",
      });
      opt.textContent = option.label;
      select.appendChild(opt);
    });

    row.innerHTML = `<strong>${item.product_name}</strong>`;
    row.appendChild(select);
    categoryList.appendChild(row);

    if (item.default) {
      const targetValue = JSON.stringify({
        column: item.default.column,
        value: item.default.value,
        description: item.default.description || "",
      });
      select.value = targetValue;
    }
  });

  categoryPanel.hidden = false;
};

const fetchOptions = (file) => {
  const formData = new FormData();
  formData.append("diet", file);

  setStatus("Loading category options…");

  fetch("/api/options", {
    method: "POST",
    body: formData,
  })
    .then((resp) => {
      if (!resp.ok) {
        throw new Error(`${resp.status}`);
      }
      return resp.json();
    })
    .then((data) => {
      if (data?.error) {
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

const runRecommendations = (useOpenAI = true) => {
  if (!uploadedFile) return;

  const overrides = [];
  categoryList.querySelectorAll("select").forEach((select) => {
    if (!select.value) return;
    const parsed = JSON.parse(select.value);
    overrides.push({
      index: Number(select.dataset.index),
      column: parsed.column,
      value: parsed.value,
      description: parsed.description,
    });
  });

  const formData = new FormData();
  formData.append("diet", uploadedFile);
  const cuisine = getSelectedCuisine();
  if (cuisine) {
    formData.append("cuisine", cuisine);
  }
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
    .then((resp) => {
      if (!resp.ok) {
        return { error: `${resp.text().then((r) => r)}` };
      }
      return resp.json();
    })
    .then((data) => {
      if (data?.error) {
        setStatus(data.error, true);
        return;
      }

      rows = (data.rows || []).map((row) => ({
        ...row,
        _search: buildSearchIndex(row),
      }));
      originalScoreEl.textContent = data.original_score?.toFixed(2) ?? "-";
      recommendedScoreEl.textContent =
        data.recommended_score?.toFixed(2) ?? "-";
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

const loadFile = (file) => {
  if (!file) return;
  uploadedFile = file;
  rows = [];
  results.innerHTML = "";
  summaryEl.hidden = true;
  componentsEl.hidden = true;
  recipeEl.hidden = true;
  fetchOptions(file);
};

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  loadFile(file);
});

runBtn.addEventListener("click", () => runRecommendations(true));
runRecipeBtn.addEventListener("click", () => runRecommendations(true));

searchInput.addEventListener("input", render);

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
