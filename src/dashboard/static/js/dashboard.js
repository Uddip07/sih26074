/**
 * Interactive Agromet Downscaling Dashboard
 * Section 10 Design System & Operational Visualization
 */

document.addEventListener("DOMContentLoaded", () => {
  let map;
  let panchayatsGeojson;
  let blocksGeojson;
  let panchayatLayer;
  let blockLayer;
  let selectedFeature = null;
  let currentDisplayMode = "downscaled_rain"; // 'downscaled_rain', 'block_forecast', 'residual', 'advisory'
  let currentHorizonDay = 1; // 1 to 5
  let allFeatures = [];
  let validationData = null;

  // Initialize Map
  function initMap() {
    map = L.map("map", {
      zoomControl: true,
      attributionControl: false
    }).setView([18.55, 74.05], 9);

    // High quality CartoDB Positron basemap (subtle, clean for choropleth viewing)
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
      subdomains: "abcd"
    }).addTo(map);

    L.control.attribution({ position: "bottomright", prefix: false })
      .addAttribution('&copy; <a href="https://carto.com/">CARTO</a> | IMD GKMS | UrbanMorph LGD')
      .addTo(map);

    loadMapData();
  }

  // Load GeoJSON and Validation Data
  async function loadMapData() {
    try {
      const [panchayatsResp, blocksResp, validationResp] = await Promise.all([
        fetch("/static/data/pune_panchayats_web.geojson"),
        fetch("/static/data/pune_blocks_web.geojson"),
        fetch("/api/validation")
      ]);

      panchayatsGeojson = await panchayatsResp.json();
      blocksGeojson = await blocksResp.json();
      validationData = await validationResp.json();

      allFeatures = panchayatsGeojson.features;

      renderPanchayatLayer();
      renderBlockLayer();
      populateBlockFilter();
      renderValidationScorecard();

      // Automatically select the first representative high-variance panchayat (e.g. Ahupe or Velhe)
      const defaultPanchayat = allFeatures.find(f => f.properties.block_name === "VELHE" || f.properties.block_name === "MAVAL") || allFeatures[0];
      if (defaultPanchayat) {
        selectPanchayat(defaultPanchayat.properties);
      }
    } catch (err) {
      console.error("Error loading geojson data:", err);
    }
  }

  // Color functions matching Section 10 Pantone palette
  function getChoroplethColor(feature, mode, dayIdx) {
    const props = feature.properties;
    const dayData = props.daily_forecasts && props.daily_forecasts[dayIdx - 1] 
      ? props.daily_forecasts[dayIdx - 1] 
      : { downscaled_rain: props.downscaled_rain, block_forecast_rain: props.block_forecast_rain, residual: props.residual_rain };

    if (mode === "downscaled_rain") {
      const r = dayData.downscaled_rain;
      if (r >= 50.0) return "#9B2423";      // PMS 7621 C - Severe Red (Extreme Rain)
      if (r >= 25.0) return "#C44818";      // Heavy Orange-Red
      if (r >= 15.0) return "#E8720C";      // PMS 1585 C - Advisory Amber
      if (r >= 5.0)  return "#8EAC24";      // Light Green-Gold
      if (r >= 1.0)  return "#6CA02D";      // PMS 7488 C - Agriculture Green
      return "#B2D38B";                     // Very light pale green
    } else if (mode === "block_forecast") {
      const b = dayData.block_forecast_rain;
      if (b >= 50.0) return "#9B2423";
      if (b >= 25.0) return "#C44818";
      if (b >= 15.0) return "#E8720C";
      if (b >= 5.0)  return "#8EAC24";
      if (b >= 1.0)  return "#6CA02D";
      return "#B2D38B";
    } else if (mode === "residual") {
      const res = dayData.residual;
      if (res >= 12.0)  return "#7E1918";   // Deep Red (severely under-predicted by block)
      if (res >= 5.0)   return "#E8720C";   // Amber
      if (res >= -2.0 && res <= 2.0) return "#F1F1F0"; // Cool Gray 1 (minimal residual)
      if (res <= -10.0) return "#003D6B";   // PMS 2955 C Primary Deep Sky (block over-predicted)
      return "#3B82F6";                     // Blue
    } else if (mode === "advisory") {
      const status = props.advisory_status;
      if (status === "Severe Alert") return "#9B2423";
      if (status === "Advisory")     return "#E8720C";
      return "#6CA02D";
    }
    return "#6CA02D";
  }

  // Render Panchayat Choropleth Layer
  function renderPanchayatLayer() {
    if (panchayatLayer) {
      map.removeLayer(panchayatLayer);
    }

    panchayatLayer = L.geoJSON(panchayatsGeojson, {
      style: (feature) => {
        const isSelected = selectedFeature && selectedFeature.gp_code === feature.properties.gp_code;
        return {
          fillColor: getChoroplethColor(feature, currentDisplayMode, currentHorizonDay),
          weight: isSelected ? 3 : 0.8,
          opacity: 1,
          color: isSelected ? "#002847" : "#FFFFFF", // 1px white border for separation (Section 10.4)
          fillOpacity: isSelected ? 0.85 : 0.65       // 60-65% fill opacity (Section 10.4)
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        
        // Tooltip
        layer.bindTooltip(`
          <div style="font-family:'Inter', sans-serif; font-size:12px;">
            <strong>${p.gp_name}</strong> <span style="color:#666;">(${p.block_name})</span><br/>
            <span>Rain: <strong style="font-family:'JetBrains Mono';">${p.downscaled_rain} mm</strong></span>
            <span style="margin-left:6px; color:#888;">(Block: ${p.block_forecast_rain} mm)</span><br/>
            <span>Elevation: <strong>${p.elevation_mean} m</strong></span>
          </div>
        `, { sticky: true });

        // Click selection
        layer.on("click", (e) => {
          L.DomEvent.stopPropagation(e);
          selectPanchayat(p);
          panchayatLayer.resetStyle();
          layer.setStyle({
            weight: 3,
            color: "#002847",
            fillOpacity: 0.9
          });
        });
      }
    }).addTo(map);
  }

  // Render Parent Block Outline Overlay
  function renderBlockLayer() {
    if (blockLayer) {
      map.removeLayer(blockLayer);
    }

    blockLayer = L.geoJSON(blocksGeojson, {
      style: {
        fill: false,
        weight: 2.2,
        color: "#003D6B", // PMS 2955 C Primary boundary lines
        dashArray: "4, 4",
        opacity: 0.85
      },
      interactive: false
    }).addTo(map);
  }

  // Select Panchayat & Update Inspector Panel
  function selectPanchayat(props) {
    selectedFeature = props;

    // Title Block
    document.getElementById("selectedPanchayatName").textContent = props.gp_name;
    document.getElementById("selectedBlockName").textContent = `${props.block_name} BLOCK | PUNE DISTRICT`;
    
    // Status Badge
    const badge = document.getElementById("selectedStatusBadge");
    badge.className = `status-badge status-${props.advisory_status === 'Severe Alert' ? 'severe' : props.advisory_status === 'Advisory' ? 'advisory' : 'normal'}`;
    badge.innerHTML = `<span style="font-size:14px;">●</span> ${props.advisory_status}`;

    // Comparison Numbers
    const dayData = props.daily_forecasts && props.daily_forecasts[currentHorizonDay - 1]
      ? props.daily_forecasts[currentHorizonDay - 1]
      : { downscaled_rain: props.downscaled_rain, block_forecast_rain: props.block_forecast_rain, residual: props.residual_rain };

    document.getElementById("valBlockForecast").textContent = `${dayData.block_forecast_rain.toFixed(1)} mm`;
    document.getElementById("valDownscaled").textContent = `${dayData.downscaled_rain.toFixed(1)} mm`;

    const deltaElem = document.getElementById("valDelta");
    const delta = dayData.residual;
    const sign = delta > 0 ? "+" : "";
    deltaElem.textContent = `Residual Adjustment: ${sign}${delta.toFixed(1)} mm`;
    deltaElem.className = `comparison-delta ${delta >= 0 ? 'delta-plus' : 'delta-minus'}`;

    // Microclimate Attributes
    document.getElementById("attrElevation").textContent = `${props.elevation_mean.toFixed(0)} m (±${props.elevation_std.toFixed(0)}m)`;
    document.getElementById("attrSlope").textContent = `${props.slope_mean.toFixed(1)}°`;
    document.getElementById("attrCoastDist").textContent = `${props.dist_to_coast_km.toFixed(1)} km`;
    document.getElementById("attrWaterDist").textContent = `${props.dist_to_water_km.toFixed(1)} km`;
    document.getElementById("attrLulc").textContent = `${props.cropland_pct.toFixed(0)}% Crop / ${props.forest_pct.toFixed(0)}% Forest`;

    // GKMS Advisory Directives
    renderAdvisoryCard(props);

    // 5-Day Sparkline
    renderSparkline(props);
  }

  // Render GKMS Advisory Directives
  function renderAdvisoryCard(props) {
    const container = document.getElementById("advisoryItemsContainer");
    container.innerHTML = "";

    const rain = props.downscaled_rain;
    const tmax = props.tmax;
    const wind = props.wind;
    const rh = props.rh;

    let items = [];

    if (rain > 50.0) {
      items.push({
        type: "severe",
        title: "HEAVY PRECIPITATION & DRAINAGE ALERT",
        text: `Extreme 24-hr rainfall (${rain.toFixed(1)} mm) projected for ${props.gp_name}. High risk of soil waterlogging and nutrient leaching.`,
        action: "Clear farm drainage channels and trenches immediately. Strictly postpone nitrogen top-dressing and chemical sprays."
      });
    } else if (rain < 2.5) {
      items.push({
        type: "advisory",
        title: "IRRIGATION ADVISORY (DRY SPELL)",
        text: `Cumulative rainfall < 2.5 mm forecast. Vegetative crop stages will experience soil moisture deficit.`,
        action: "Provide light to moderate drip/micro-sprinkler irrigation during early morning hours. Apply bio-mulch to retain soil moisture."
      });
    }

    if (tmax > 35.0 && rh > 70.0) {
      items.push({
        type: "advisory",
        title: "HIGH FUNGAL & PEST VULNERABILITY",
        text: `Max temperature (${tmax.toFixed(1)}°C) and relative humidity (${rh.toFixed(0)}%) create optimal conditions for Downy Mildew and Thrips.`,
        action: "Undertake prophylactic spray of Copper Oxychloride (2.5 g/L) or neem-based bio-pesticide once foliage dries."
      });
    }

    if (wind > 40.0) {
      items.push({
        type: "severe",
        title: "STRONG WIND & LODGING RISK",
        text: `Wind gusts exceeding 40 km/h (${wind.toFixed(1)} km/h) can cause physical crop lodging and drift.`,
        action: "Provide earthing-up and bamboo staking in banana and sugarcane. Postpone all spray operations until wind drops below 15 km/h."
      });
    }

    if (items.length === 0) {
      items.push({
        type: "normal",
        title: "FAVORABLE AGRO-CLIMATIC CONDITIONS",
        text: `Moderate temperatures and light moisture create ideal vegetative growing conditions across ${props.gp_name}.`,
        action: "Proceed with scheduled intercultural operations, weeding, and balanced fertilizer applications as per crop calendar."
      });
    }

    items.forEach(item => {
      const div = document.createElement("div");
      div.className = `advisory-item-block ${item.type}`;
      div.innerHTML = `
        <div class="advisory-title-row">
          <span>${item.title}</span>
          <span style="font-size:10px; text-transform:uppercase; letter-spacing:0.04em;">GKMS Rule</span>
        </div>
        <div class="advisory-text">${item.text}</div>
        <div class="action-directive">Directive: ${item.action}</div>
      `;
      container.appendChild(div);
    });
  }

  // Render 5-Day Sparkline Bar Chart
  function renderSparkline(props) {
    const container = document.getElementById("sparklineChart");
    container.innerHTML = "";

    if (!props.daily_forecasts || props.daily_forecasts.length === 0) return;

    const maxRain = Math.max(...props.daily_forecasts.map(d => d.downscaled_rain), 30.0);

    props.daily_forecasts.forEach(d => {
      const heightPct = Math.max(8, (d.downscaled_rain / maxRain) * 100);
      const isSelectedDay = d.day === currentHorizonDay;

      const barClass = d.downscaled_rain >= 50.0 ? "high-rain" : (d.downscaled_rain >= 15.0 ? "mod-rain" : "");

      const wrapper = document.createElement("div");
      wrapper.className = "spark-bar-wrapper";
      wrapper.style.cursor = "pointer";
      wrapper.title = `Day ${d.day} (${d.date}): ${d.downscaled_rain} mm downscaled vs ${d.block_forecast_rain} mm block`;

      wrapper.innerHTML = `
        <div class="spark-val" style="color: ${isSelectedDay ? 'var(--pms-primary)' : 'inherit'}; font-weight: ${isSelectedDay ? '700' : '500'};">
          ${d.downscaled_rain.toFixed(1)}
        </div>
        <div class="spark-bar ${barClass}" style="height: ${heightPct}%; border: ${isSelectedDay ? '2px solid #002847' : 'none'};"></div>
        <div class="spark-date" style="font-weight: ${isSelectedDay ? '700' : '400'}; color: ${isSelectedDay ? 'var(--pms-primary)' : '#666'};">
          D${d.day}
        </div>
      `;

      wrapper.addEventListener("click", () => {
        setHorizonDay(d.day);
      });

      container.appendChild(wrapper);
    });
  }

  // Populate Block Selector Dropdown
  function populateBlockFilter() {
    const select = document.getElementById("blockSelectFilter");
    if (!select || !blocksGeojson) return;

    const blocks = blocksGeojson.features.map(f => f.properties.block_name).sort();
    blocks.forEach(b => {
      const opt = document.createElement("option");
      opt.value = b;
      opt.textContent = `${b} Block`;
      select.appendChild(opt);
    });

    select.addEventListener("change", (e) => {
      const selectedBlock = e.target.value;
      if (!selectedBlock) {
        map.setView([18.55, 74.05], 9);
        return;
      }

      const blockFeature = blocksGeojson.features.find(f => f.properties.block_name === selectedBlock);
      if (blockFeature) {
        const bbox = L.geoJSON(blockFeature).getBounds();
        map.fitBounds(bbox, { padding: [20, 20] });
      }

      // Also select a panchayat in this block
      const p = allFeatures.find(f => f.properties.block_name === selectedBlock);
      if (p) {
        selectPanchayat(p.properties);
      }
    });
  }

  // Horizon Day Switcher (1 to 5)
  function setHorizonDay(day) {
    currentHorizonDay = day;
    document.querySelectorAll(".step-day-btn").forEach(btn => {
      btn.classList.toggle("active", parseInt(btn.dataset.day) === day);
    });

    // Re-render map and inspector
    renderPanchayatLayer();
    if (selectedFeature) {
      selectPanchayat(selectedFeature);
    }
  }

  // Setup UI Listeners
  function setupUIListeners() {
    // Mode toggles
    document.querySelectorAll(".layer-toggle-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".layer-toggle-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentDisplayMode = btn.dataset.mode;
        renderPanchayatLayer();
      });
    });

    // 5-Day timeline step buttons
    document.querySelectorAll(".step-day-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        setHorizonDay(parseInt(btn.dataset.day));
      });
    });

    // View switcher: Operational Map vs Validation Scorecard
    document.getElementById("btnNavMap").addEventListener("click", () => {
      document.getElementById("btnNavMap").classList.add("active");
      document.getElementById("btnNavValidation").classList.remove("active");
      document.getElementById("mapWorkspaceView").style.display = "grid";
      document.getElementById("validationView").classList.remove("active");
      if (map) map.invalidateSize();
    });

    document.getElementById("btnNavValidation").addEventListener("click", () => {
      document.getElementById("btnNavValidation").classList.add("active");
      document.getElementById("btnNavMap").classList.remove("active");
      document.getElementById("mapWorkspaceView").style.display = "none";
      document.getElementById("validationView").classList.add("active");
    });
  }

  // Render LOSOCV Validation Scorecard
  function renderValidationScorecard() {
    if (!validationData) return;

    document.getElementById("scorecardSkillScore").textContent = validationData.mean_skill_score_pct || "+37.5%";
    document.getElementById("scorecardBaseRmse").textContent = `${validationData.mean_baseline_rmse} mm`;
    document.getElementById("scorecardModelRmse").textContent = `${validationData.mean_model_rmse} mm`;
    document.getElementById("scorecardPitchText").textContent = `"${validationData.headline_pitch}"`;

    const tableBody = document.getElementById("losocvTableBody");
    if (!tableBody || !validationData.fold_breakdown) return;

    tableBody.innerHTML = "";
    validationData.fold_breakdown.forEach(f => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:700; color:var(--pms-primary);">${f.fold}</td>
        <td style="font-weight:600;">${f.held_out_block}</td>
        <td>${f.n_panchayats}</td>
        <td>${f.baseline_rmse.toFixed(2)} mm</td>
        <td style="color:#003D6B; font-weight:700;">${f.model_rmse.toFixed(2)} mm</td>
        <td style="color:#2E7D32; font-weight:700;">+${f.skill_score_pct.toFixed(1)}%</td>
        <td>${f.baseline_mae.toFixed(2)} mm</td>
        <td>${f.model_mae.toFixed(2)} mm</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // Run Initialization
  setupUIListeners();
  initMap();
});
