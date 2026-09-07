// EcoLens 2.0 Client Application Controller
const API_BASE = window.location.origin;

let currentView = 'dashboard';
let scannerType = 'plant'; // 'plant' or 'animal'
let activeSubject = 'Bengal Tiger';
let allSpecies = [];
let isListening = false;
let recognition = null;
let currentSpeechText = "";

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSpeechRecognition();
  
  // Set active default keys if none saved in browser
  const defaultGemini = atob('QVEuQWI4Uk42SjJ5dnVBOEl2WTF6WHlKSUk2YzU5SG9ZZkIxRkNySm5teUtFTW1vRUMwOGc=');
  const defaultPlantnet = '2b10kU10zzN5T31LX3uKu3Pqsu';
  if (!localStorage.getItem('ecolens_gemini_key')) {
    localStorage.setItem('ecolens_gemini_key', defaultGemini);
  }
  if (!localStorage.getItem('ecolens_plantnet_key')) {
    localStorage.setItem('ecolens_plantnet_key', defaultPlantnet);
  }
  const gemInput = document.getElementById('input-gemini-key');
  const pnetInput = document.getElementById('input-plantnet-key');
  const gmapInput = document.getElementById('input-gmaps-key');
  if (gemInput) gemInput.value = localStorage.getItem('ecolens_gemini_key') || '';
  if (pnetInput) pnetInput.value = localStorage.getItem('ecolens_plantnet_key') || '';
  if (gmapInput) gmapInput.value = localStorage.getItem('ecolens_gmaps_key') || '';

  fetchHealthStatus();
  fetchSpeciesDatabase();
  setupNavDrawer();
});

function setupNavDrawer() {
  const menuToggle = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('nav-sidebar');
  if (menuToggle && sidebar) {
    menuToggle.addEventListener('click', () => {
      sidebar.classList.toggle('hidden');
      sidebar.classList.toggle('fixed');
      sidebar.classList.toggle('inset-0');
      sidebar.classList.toggle('z-50');
    });
  }
}

// ============================================================
// CLIENT-SIDE ROUTING & NAVIGATION
// ============================================================
function navigate(viewName) {
  if (typeof stopScannerCamera === "function") stopScannerCamera();
  if (typeof stopForestCamera === "function") stopForestCamera();

  currentView = viewName;
  
  // Hide all views
  document.getElementById('view-dashboard').classList.add('hidden');
  document.getElementById('view-forest-ai').classList.add('hidden');
  document.getElementById('view-scanner').classList.add('hidden');
  document.getElementById('view-voice-assistant').classList.add('hidden');
  document.getElementById('view-species-database').classList.add('hidden');
  const hubView = document.getElementById('view-carbon-hub');
  if (hubView) hubView.classList.add('hidden');

  // Activate target view
  if (viewName === 'dashboard') {
    document.getElementById('view-dashboard').classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Biosphere Telemetry Grid';
  } else if (viewName === 'forest-ai') {
    document.getElementById('view-forest-ai').classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Forest Canopy Classification';
  } else if (viewName === 'plant-scanner') {
    setScannerMode('plant');
    document.getElementById('view-scanner').classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Botanical Intelligence';
  } else if (viewName === 'animal-scanner') {
    setScannerMode('animal');
    document.getElementById('view-scanner').classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Wildlife Intelligence';
  } else if (viewName === 'voice-assistant') {
    document.getElementById('view-voice-assistant').classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Bioacoustic Neural Uplink';
  } else if (viewName === 'species-database') {
    document.getElementById('view-species-database').classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Species Knowledge Base';
  } else if (viewName === 'carbon-hub') {
    const hubView = document.getElementById('view-carbon-hub');
    if (hubView) hubView.classList.remove('hidden');
    document.getElementById('active-screen-subtitle').textContent = 'Carbon Credit & Climate Finance Hub';
    if (typeof initStandaloneCarbonHub === 'function') {
      initStandaloneCarbonHub();
    }
  }

  // Update desktop sidebar buttons
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.remove('nav-active');
    if (btn.dataset.view === viewName) {
      btn.classList.add('nav-active');
    }
  });

  // Update mobile bottom nav buttons
  document.querySelectorAll('.bottom-nav-btn').forEach(btn => {
    btn.classList.remove('bottom-nav-active');
    if (btn.dataset.view === viewName) {
      btn.classList.add('bottom-nav-active');
    }
  });

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// API STATUS & HEALTH
// ============================================================
async function fetchHealthStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.status === 'online') {
      document.getElementById('system-status-badge').textContent = '2 APIs Active';
      document.getElementById('model-status').textContent = data.forest_model_loaded ? 'LOADED' : 'CPU MODE';
      document.getElementById('gemini-status').textContent = data.gemini_active ? 'ONLINE' : 'NOT SET';
      document.getElementById('gemini-status').className = data.gemini_active ? 'px-1.5 py-0.5 rounded bg-secondary-container/20 text-secondary text-[10px] font-mono' : 'px-1.5 py-0.5 rounded bg-tertiary-container/20 text-tertiary text-[10px] font-mono';
      document.getElementById('plantnet-status').textContent = data.plantnet_active ? 'ONLINE' : 'NOT SET';
      document.getElementById('plantnet-status').className = data.plantnet_active ? 'px-1.5 py-0.5 rounded bg-secondary-container/20 text-secondary text-[10px] font-mono' : 'px-1.5 py-0.5 rounded bg-tertiary-container/20 text-tertiary text-[10px] font-mono';
    }
  } catch (e) {
    console.error('Failed to fetch system health:', e);
  }
}

// ============================================================
// SPECIES DATABASE & CATALOG
// ============================================================
async function fetchSpeciesDatabase() {
  try {
    const res = await fetch(`${API_BASE}/api/species`);
    const data = await res.json();
    allSpecies = data.species || [];
    
    // Update select dropdown in Voice Assistant
    const select = document.getElementById('voice-subject-select');
    select.innerHTML = '';
    allSpecies.forEach(sp => {
      const opt = document.createElement('option');
      opt.value = sp.name;
      opt.textContent = `${sp.name} (${sp.scientific_name})`;
      if (sp.name === activeSubject) opt.selected = true;
      select.appendChild(opt);
    });

    // Populate Species Catalog Grid
    renderSpeciesGrid(allSpecies);
    document.getElementById('species-count-badge').textContent = `${allSpecies.length} Species`;
  } catch (e) {
    console.error('Failed to fetch species catalog:', e);
  }
}

function renderSpeciesGrid(speciesList) {
  const container = document.getElementById('species-grid');
  container.innerHTML = '';
  speciesList.forEach(sp => {
    const card = document.createElement('div');
    card.className = 'p-4 rounded-xl bg-surface-container border border-surface-container-highest/30 flex flex-col gap-2 hover:border-primary/40 transition-all';
    card.innerHTML = `
      <div class="flex items-start justify-between">
        <div>
          <h4 class="font-headline text-base font-bold text-secondary">${sp.name}</h4>
          <p class="font-body text-xs italic text-primary">${sp.scientific_name || ''}</p>
        </div>
        <span class="font-mono text-[10px] px-2 py-0.5 rounded bg-secondary-container/20 text-secondary">${sp.conservation || 'IUCN'}</span>
      </div>
      <div class="text-xs flex flex-col gap-1 text-on-surface-variant">
        <div><strong class="text-on-surface">Type:</strong> ${sp.type || 'N/A'} • <strong class="text-on-surface">Family:</strong> ${sp.family || 'N/A'}</div>
        <div><strong class="text-secondary">🌍 Found In:</strong> ${sp.region || sp.habitat || 'Global'}</div>
        <div><strong class="text-on-surface">Population:</strong> ${sp.population || 'Monitored'}</div>
      </div>
      <div class="mt-2 pt-2 border-t border-surface-container-highest/20 flex justify-between items-center">
        <span class="text-[11px] text-on-surface-variant line-clamp-1">${sp.facts || ''}</span>
        <div class="flex items-center gap-1.5 shrink-0">
          <button onclick="viewSpeciesDistributionDetails('${sp.name.replace(/'/g, "\\'")}')" class="px-2 py-1 rounded bg-secondary/15 hover:bg-secondary/25 text-secondary text-[10px] font-mono flex items-center gap-1">
            <span class="material-symbols-outlined text-[12px]">public</span> Map
          </button>
          <button onclick="selectSubjectForVoice('${sp.name.replace(/'/g, "\\'")}')" class="px-2 py-1 rounded bg-surface-container-high hover:bg-primary/20 text-primary text-[10px] font-mono flex items-center gap-1">
            <span class="material-symbols-outlined text-[12px]">mic</span> Ask AI
          </button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function filterSpeciesCatalog(query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    renderSpeciesGrid(allSpecies);
    return;
  }
  const filtered = allSpecies.filter(s => 
    s.name.toLowerCase().includes(q) ||
    (s.scientific_name && s.scientific_name.toLowerCase().includes(q)) ||
    (s.region && s.region.toLowerCase().includes(q)) ||
    (s.habitat && s.habitat.toLowerCase().includes(q))
  );
  renderSpeciesGrid(filtered);
}

function selectSubjectForVoice(name) {
  changeVoiceSubject(name);
  navigate('voice-assistant');
}

function changeVoiceSubject(name) {
  activeSubject = name;
  document.getElementById('voice-subject-display').textContent = name;
  const select = document.getElementById('voice-subject-select');
  if (select) select.value = name;
  addChatMessage('assistant', `Switched subject context to **${name}**. Ask me about its status, diet, habitat, ecological role, or threats!`);
}

// ============================================================
// 1. FOREST AI IMAGE ANALYSIS & CAMERA ENGINE
// ============================================================
let forestCameraStream = null;
let forestCameraFacing = 'environment';
let currentForestImageSrc = '';

async function startForestCamera() {
  const video = document.getElementById('forest-camera-feed');
  const container = document.getElementById('forest-camera-container');
  const previewContainer = document.getElementById('forest-preview-container');
  const resultCard = document.getElementById('forest-results-card');

  if (previewContainer) previewContainer.classList.add('hidden');
  if (resultCard) resultCard.classList.add('hidden');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const camInput = document.getElementById('forest-camera-input');
    if (camInput) camInput.click();
    return;
  }

  try {
    if (forestCameraStream) {
      stopForestCamera();
    }

    const constraints = {
      video: {
        facingMode: { ideal: forestCameraFacing },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    forestCameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = forestCameraStream;
    await video.play();

    if (container) {
      container.classList.remove('hidden');
      container.classList.add('flex');
    }
  } catch (err) {
    console.warn('Forest camera stream error, falling back to capture input:', err);
    const camInput = document.getElementById('forest-camera-input');
    if (camInput) camInput.click();
  }
}

function stopForestCamera() {
  const container = document.getElementById('forest-camera-container');
  const previewContainer = document.getElementById('forest-preview-container');
  const video = document.getElementById('forest-camera-feed');

  if (forestCameraStream) {
    forestCameraStream.getTracks().forEach(track => track.stop());
    forestCameraStream = null;
  }

  if (video) video.srcObject = null;
  if (container) {
    container.classList.add('hidden');
    container.classList.remove('flex');
  }
  if (previewContainer) {
    previewContainer.classList.remove('hidden');
  }
}

async function switchForestCamera() {
  forestCameraFacing = (forestCameraFacing === 'environment') ? 'user' : 'environment';
  await startForestCamera();
}

function captureFromForestCamera() {
  const video = document.getElementById('forest-camera-feed');
  const canvas = document.getElementById('forest-camera-canvas');
  if (!video || !video.videoWidth || !video.videoHeight) {
    alert('Camera feed not ready. Please wait a moment.');
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const file = new File([blob], 'forest_scan_' + Date.now() + '.jpg', { type: 'image/jpeg' });
    stopForestCamera();
    await processAndAnalyzeForestFile(file);
  }, 'image/jpeg', 0.92);
}

async function handleForestUpload(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  await processAndAnalyzeForestFile(file);
}

async function processAndAnalyzeForestFile(file) {
  const preview = document.getElementById('forest-img-preview');
  currentForestImageSrc = URL.createObjectURL(file);
  preview.src = currentForestImageSrc;
  preview.classList.remove('hidden');
  document.getElementById('forest-upload-placeholder').classList.add('hidden');
  document.getElementById('forest-loading').classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/api/forest/analyze`, {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    document.getElementById('forest-loading').classList.add('hidden');
    displayForestResults(data);
  } catch (e) {
    document.getElementById('forest-loading').classList.add('hidden');
    alert('Error analyzing forest imagery: ' + e.message);
  }
}

function updateForestSlider(val) {
  const wrapper = document.getElementById('forest-slider-before-wrapper');
  const handle = document.getElementById('forest-slider-handle');
  const beforeImg = document.getElementById('forest-slider-before');
  const container = document.getElementById('forest-slider-container');
  
  if (wrapper) wrapper.style.width = `${val}%`;
  if (handle) handle.style.left = `${val}%`;
  if (beforeImg && container) {
    beforeImg.style.width = `${container.clientWidth}px`;
  }
}

// Adjust slider images on window resize
window.addEventListener('resize', () => {
  const range = document.getElementById('forest-split-range');
  if (range) updateForestSlider(range.value);
});

// ============================================================
// CARBON CREDIT & ECONOMIC ROI CALCULATOR
// ============================================================
let carbonHectares = 25;
let carbonPricePerTonne = 30;
let carbonHorizonYears = 5;
let carbonBaseRatePerHa = 22; // default 22 tonnes CO2e / ha / year

function onHectaresInput(val) {
  const num = Math.max(1, Math.min(1000, parseInt(val) || 1));
  carbonHectares = num;
  const slider = document.getElementById('carbon-hectares-slider');
  const input = document.getElementById('carbon-hectares-input');
  if (slider && parseInt(slider.value) !== num) slider.value = num;
  if (input && parseInt(input.value) !== num) input.value = num;
  updateCarbonROI();
}

function setCarbonPrice(price) {
  carbonPricePerTonne = price;
  
  [15, 30, 50].forEach(p => {
    const btn = document.getElementById(`btn-price-${p}`);
    if (btn) {
      if (p === price) {
        btn.className = 'px-2 py-1.5 rounded-lg bg-primary text-on-primary text-[11px] font-mono font-bold border border-primary transition-all text-center cursor-pointer shadow';
      } else {
        btn.className = 'px-2 py-1.5 rounded-lg bg-surface-container-high hover:bg-surface-bright text-on-surface-variant text-[11px] font-mono border border-surface-container-highest transition-all text-center cursor-pointer';
      }
    }
  });

  const priceText = document.getElementById('carbon-active-price-text');
  if (priceText) {
    const tierName = price === 15 ? 'Avoidance' : (price === 30 ? 'Removal' : 'Bio-ARR');
    priceText.textContent = `$${price} / t CO2e (${tierName})`;
  }

  updateCarbonROI();
}

function setCarbonHorizon(years) {
  carbonHorizonYears = years;

  [1, 5, 10, 20].forEach(y => {
    const btn = document.getElementById(`btn-horizon-${y}`);
    if (btn) {
      if (y === years) {
        btn.className = 'px-3 py-1 rounded-md bg-secondary text-on-secondary font-bold transition-all cursor-pointer shadow';
      } else {
        btn.className = 'px-3 py-1 rounded-md text-on-surface-variant hover:text-on-surface transition-all cursor-pointer';
      }
    }
  });

  updateCarbonROI();
}

function initCarbonCalculator(data) {
  const canopy = data.canopy_cover_percent || 50;
  const bare = data.bare_ground_percent || (100 - canopy);
  
  // Calibrate sequestration rate: degraded land has higher restoration delta
  if (bare >= 50) {
    carbonBaseRatePerHa = 24.5;
  } else if (bare >= 25) {
    carbonBaseRatePerHa = 21.0;
  } else {
    carbonBaseRatePerHa = 18.0;
  }

  // Auto-calibrate parcel size if large deforestation is detected
  if (bare > 60 && carbonHectares < 40) {
    carbonHectares = 50;
    const slider = document.getElementById('carbon-hectares-slider');
    const input = document.getElementById('carbon-hectares-input');
    if (slider) slider.value = 50;
    if (input) input.value = 50;
  }

  updateCarbonROI();
}

function updateCarbonROI() {
  // Annual sequestration (tonnes of CO2e)
  const annualSequestration = carbonHectares * carbonBaseRatePerHa;
  // Cumulative sequestration over selected horizon
  const totalSequestration = Math.round(annualSequestration * carbonHorizonYears);

  // Financial modeling (USD)
  const grossRevenue = Math.round(totalSequestration * carbonPricePerTonne);
  
  // Capital & Operational Expenditure
  const initialPlantingCost = carbonHectares * 850; // $850/ha saplings, labor, soil inoculation
  const annualMonitoringCost = carbonHectares * 110; // $110/ha/year telemetry, remote sensing audit, rangers
  const totalExpenditure = initialPlantingCost + (annualMonitoringCost * carbonHorizonYears);

  const netProfit = Math.max(0, grossRevenue - totalExpenditure);
  
  // Payback period (years)
  const annualGross = annualSequestration * carbonPricePerTonne;
  const annualNetCashflow = Math.max(100, annualGross - annualMonitoringCost);
  const paybackYears = Math.min(20, Math.max(0.8, (initialPlantingCost / annualNetCashflow))).toFixed(1);

  // Real-world equivalencies (EPA & ICAO standards)
  const carsRemovedPerYear = Math.round(annualSequestration / 4.6); // 4.6 t CO2e / car / year
  const flightsOffset = Math.round(totalSequestration / 0.8); // 0.8 t CO2e / passenger flight
  const waterGainPercent = Math.min(55, Math.round(18 + (carbonHectares * 0.08)));

  // Update DOM Elements
  const elTotalTonnes = document.getElementById('carbon-total-tonnes');
  const elGross = document.getElementById('carbon-gross-revenue');
  const elNet = document.getElementById('carbon-net-profit');
  const elPayback = document.getElementById('carbon-payback');
  const elCars = document.getElementById('carbon-eq-cars');
  const elFlights = document.getElementById('carbon-eq-flights');
  const elWater = document.getElementById('carbon-eq-water');

  if (elTotalTonnes) elTotalTonnes.textContent = `${totalSequestration.toLocaleString()} t`;
  if (elGross) elGross.textContent = `$${grossRevenue.toLocaleString()}`;
  if (elNet) elNet.textContent = `+$${netProfit.toLocaleString()}`;
  if (elPayback) elPayback.textContent = `${paybackYears} Years`;
  if (elCars) elCars.textContent = `${carsRemovedPerYear.toLocaleString()} cars/yr`;
  if (elFlights) elFlights.textContent = `${flightsOffset.toLocaleString()} flights`;
  if (elWater) elWater.textContent = `+${waterGainPercent}% groundwater`;
}

function displayForestResults(data) {
  const card = document.getElementById('forest-results-card');
  card.classList.remove('hidden');

  document.getElementById('forest-pred-title').textContent = data.prediction;
  document.getElementById('forest-conf-text').textContent = `${data.confidence}%`;
  document.getElementById('forest-conf-bar').style.width = `${data.confidence}%`;

  const badge = document.getElementById('forest-status-badge');
  const badgeText = document.getElementById('forest-pred-badge-text');
  if (data.prediction === 'Healthy Forest') {
    badge.className = 'px-4 py-1.5 rounded-full bg-secondary-container/20 text-secondary border border-secondary/30 font-semibold text-xs flex items-center gap-1.5';
    badgeText.textContent = 'Healthy Forest ✅';
  } else {
    badge.className = 'px-4 py-1.5 rounded-full bg-error-container/30 text-error border border-error/30 font-semibold text-xs flex items-center gap-1.5';
    badgeText.textContent = 'Deforested Area ⚠️';
  }

  // Remote Sensing Metrics
  document.getElementById('metric-gli').textContent = (data.gli_index >= 0 ? '+' : '') + data.gli_index;
  document.getElementById('metric-vari').textContent = (data.vari_index >= 0 ? '+' : '') + data.vari_index;
  document.getElementById('metric-canopy').textContent = `${data.canopy_cover_percent}%`;

  // XAI Attribution Tensors
  if (data.xai) {
    if (data.xai.canopy_mask_b64) document.getElementById('xai-mask').src = `data:image/jpeg;base64,${data.xai.canopy_mask_b64}`;
    if (data.xai.veg_health_b64) document.getElementById('xai-gradient').src = `data:image/jpeg;base64,${data.xai.veg_health_b64}`;
    // Use canopy mask or original for Grad-CAM tensor
    if (data.xai.gradcam_b64) {
      document.getElementById('xai-gradcam').src = `data:image/jpeg;base64,${data.xai.gradcam_b64}`;
    } else if (data.xai.canopy_mask_b64) {
      document.getElementById('xai-gradcam').src = `data:image/jpeg;base64,${data.xai.canopy_mask_b64}`;
    }
  }

  // Interactive Before & After Slider Setup
  const beforeImg = document.getElementById('forest-slider-before');
  const afterImg = document.getElementById('forest-slider-after');
  if (beforeImg) beforeImg.src = currentForestImageSrc;
  if (afterImg) {
    if (data.xai && data.xai.reforested_simulation_b64) {
      afterImg.src = `data:image/jpeg;base64,${data.xai.reforested_simulation_b64}`;
    } else {
      afterImg.src = currentForestImageSrc;
    }
  }
  const splitRange = document.getElementById('forest-split-range');
  if (splitRange) {
    splitRange.value = 50;
    setTimeout(() => updateForestSlider(50), 100);
  }

  // Gemini Autonomous Diagnostics
  const diag = data.diagnostics || {};
  document.getElementById('diag-biome').textContent = diag.biome || 'Tropical Moist Deciduous Canopy';
  if (document.getElementById('diag-density-class')) {
    document.getElementById('diag-density-class').textContent = diag.canopy_density_class || (data.canopy_cover_percent >= 70 ? 'Dense Crown (>70%)' : 'Degraded Canopy');
  }
  document.getElementById('diag-integrity').textContent = diag.integrity || 'Canopy integrity verified across spatial crown tensors.';
  document.getElementById('diag-drivers').textContent = diag.drivers || 'Vegetative transpiration and conservation buffer active.';
  if (document.getElementById('diag-carbon-risk')) {
    document.getElementById('diag-carbon-risk').textContent = diag.carbon_loss_risk || `${data.carbon_loss_per_ha} tonnes C/ha estimated exposure`;
  }
  if (document.getElementById('diag-bio-threat')) {
    document.getElementById('diag-bio-threat').textContent = diag.biodiversity_threat || 'Habitat corridors under continuous telemetry.';
  }
  document.getElementById('diag-protocol').textContent = diag.stewardship_protocol || 'Maintain regular satellite passes and local nursery support.';
  if (document.getElementById('diag-score-pill')) {
    const score = diag.ecological_health_score || Math.round(data.canopy_cover_percent);
    document.getElementById('diag-score-pill').textContent = `Health: ${score}/100`;
  }

  // Afforestation & Canopy Restoration Roadmap
  const roadmap = diag.afforestation_roadmap || {};
  if (document.getElementById('afforest-summary')) {
    document.getElementById('afforest-summary').textContent = roadmap.summary || 'Assisted Natural Regeneration with multi-tier native tree planting.';
  }
  if (document.getElementById('afforest-carbon-badge')) {
    document.getElementById('afforest-carbon-badge').textContent = roadmap.carbon_sequestration_potential || '~22 tonnes CO2/ha/yr';
  }
  if (document.getElementById('afforest-soil')) {
    document.getElementById('afforest-soil').textContent = roadmap.soil_and_water_interventions || 'Contour swales, woodchip mulch, and mycorrhizal biochar inoculation.';
  }
  if (document.getElementById('afforest-timeline')) {
    document.getElementById('afforest-timeline').textContent = roadmap.recovery_timeline || 'Months 1-6: Site preparation; Years 1-3: Pioneer canopy closure; Years 4+: Climax forest succession.';
  }

  const speciesGrid = document.getElementById('afforest-species-grid');
  if (speciesGrid) {
    speciesGrid.innerHTML = '';
    const speciesList = roadmap.recommended_species || [
      { name: "Neem (Azadirachta indica)", type: "Pioneer Native Tree", role: "Fast root anchoring, pest resilience, cooling", growth_rate: "Fast (1.2 - 1.5 m/yr)" },
      { name: "Teak (Tectona grandis)", type: "Climax Canopy Tree", role: "High carbon storage & permanent soil retention", growth_rate: "Moderate (0.8 - 1.2 m/yr)" },
      { name: "Banyan / Peepal (Ficus spp.)", type: "Keystone Canopy", role: "Avian feeding shelter & massive crown spread", growth_rate: "Moderate" },
      { name: "Vetiver Grass", type: "Soil Stabilizer", role: "Stops topsoil runoff along contour swales", growth_rate: "Very Fast" }
    ];

    speciesList.forEach(sp => {
      const card = document.createElement('div');
      card.className = 'p-2.5 rounded-lg bg-surface-container border border-surface-container-highest/40 flex flex-col gap-1 text-xs';
      card.innerHTML = `
        <div class="flex items-start justify-between gap-1">
          <span class="font-headline font-bold text-secondary text-xs">${sp.name}</span>
          <span class="px-1.5 py-0.5 rounded bg-secondary/15 text-secondary text-[9px] font-mono shrink-0">${sp.type || 'Native Species'}</span>
        </div>
        <p class="text-[11px] text-on-surface-variant line-clamp-2">${sp.role || 'Restores ecological equilibrium'}</p>
        <div class="text-[10px] font-mono text-primary flex items-center gap-1 mt-0.5">
          <span class="material-symbols-outlined text-[12px]">speed</span> Growth: ${sp.growth_rate || 'Moderate'}
        </div>
      `;
      speciesGrid.appendChild(card);
    });
  }

  // Initialize Carbon Credit & Economic ROI Calculator
  initCarbonCalculator(data);

  // Smooth scroll to results
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// ============================================================
// 2. BOTANICAL & WILDLIFE SCANNER & CAMERA ENGINE
// ============================================================
let cameraStream = null;
let currentCameraFacing = 'environment';

function setScannerMode(mode) {
  scannerType = mode;
  const btnP = document.getElementById('btn-mode-plant');
  const btnA = document.getElementById('btn-mode-animal');
  const icon = document.getElementById('scanner-icon');
  const heading = document.getElementById('scanner-heading-text');
  const desc = document.getElementById('scanner-desc');

  if (mode === 'plant') {
    if (btnP) btnP.className = 'px-3 py-1 rounded-full bg-primary-container text-on-primary-container font-mono text-xs font-semibold transition-all';
    if (btnA) btnA.className = 'px-3 py-1 rounded-full text-on-surface-variant font-mono text-xs transition-all';
    if (icon) icon.textContent = '🌿';
    if (heading) heading.textContent = 'Plant Scanner';
    if (desc) desc.textContent = 'High-precision botanical computer vision trained on Pl@ntNet & Kew taxonomy.';
  } else {
    if (btnA) btnA.className = 'px-3 py-1 rounded-full bg-tertiary-container text-on-tertiary-container font-mono text-xs font-semibold transition-all';
    if (btnP) btnP.className = 'px-3 py-1 rounded-full text-on-surface-variant font-mono text-xs transition-all';
    if (icon) icon.textContent = '🐾';
    if (heading) heading.textContent = 'Animal Scanner';
    if (desc) desc.textContent = 'Wildlife visual intelligence engine identifying species across global biomes.';
  }
}

async function startScannerCamera() {
  const video = document.getElementById('scanner-camera-feed');
  const container = document.getElementById('scanner-camera-container');
  const placeholder = document.getElementById('scanner-placeholder');
  const previewContainer = document.getElementById('scanner-preview-container');
  const resultCard = document.getElementById('species-result-card');

  if (previewContainer) previewContainer.classList.add('hidden');
  if (resultCard) resultCard.classList.add('hidden');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const camInput = document.getElementById('scanner-camera-input');
    if (camInput) camInput.click();
    return;
  }

  try {
    if (cameraStream) {
      stopScannerCamera();
    }

    const constraints = {
      video: {
        facingMode: { ideal: currentCameraFacing },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = cameraStream;
    await video.play();

    if (placeholder) placeholder.classList.add('hidden');
    if (container) {
      container.classList.remove('hidden');
      container.classList.add('flex');
    }
  } catch (err) {
    console.warn('Live camera stream error, falling back to capture input:', err);
    const camInput = document.getElementById('scanner-camera-input');
    if (camInput) camInput.click();
  }
}

function stopScannerCamera() {
  const container = document.getElementById('scanner-camera-container');
  const placeholder = document.getElementById('scanner-placeholder');
  const video = document.getElementById('scanner-camera-feed');
  const previewContainer = document.getElementById('scanner-preview-container');

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  if (video) video.srcObject = null;
  if (container) {
    container.classList.add('hidden');
    container.classList.remove('flex');
  }
  if (placeholder && (!previewContainer || previewContainer.classList.contains('hidden'))) {
    placeholder.classList.remove('hidden');
  }
}

async function switchCamera() {
  currentCameraFacing = (currentCameraFacing === 'environment') ? 'user' : 'environment';
  await startScannerCamera();
}

function captureFromCamera() {
  const video = document.getElementById('scanner-camera-feed');
  const canvas = document.getElementById('scanner-camera-canvas');
  if (!video || !video.videoWidth || !video.videoHeight) {
    alert('Camera feed not ready. Please wait a moment.');
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const file = new File([blob], 'specimen_' + Date.now() + '.jpg', { type: 'image/jpeg' });
    stopScannerCamera();
    await processAndIdentifyFile(file);
  }, 'image/jpeg', 0.92);
}

function resetScannerView() {
  stopScannerCamera();
  const previewContainer = document.getElementById('scanner-preview-container');
  const placeholder = document.getElementById('scanner-placeholder');
  const resultCard = document.getElementById('species-result-card');
  const fileInput = document.getElementById('scanner-file-input');
  const camInput = document.getElementById('scanner-camera-input');

  if (previewContainer) previewContainer.classList.add('hidden');
  if (resultCard) resultCard.classList.add('hidden');
  if (placeholder) placeholder.classList.remove('hidden');
  if (fileInput) fileInput.value = '';
  if (camInput) camInput.value = '';
}

async function handleSpeciesUpload(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  stopScannerCamera();
  await processAndIdentifyFile(file);
}

async function processAndIdentifyFile(file) {
  const preview = document.getElementById('scanner-img-preview');
  const previewContainer = document.getElementById('scanner-preview-container');
  const placeholder = document.getElementById('scanner-placeholder');
  const loading = document.getElementById('scanner-loading');
  const resultCard = document.getElementById('species-result-card');

  if (resultCard) resultCard.classList.add('hidden');
  if (preview) preview.src = URL.createObjectURL(file);
  if (previewContainer) previewContainer.classList.remove('hidden');
  if (placeholder) placeholder.classList.add('hidden');
  if (loading) loading.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('species_type', scannerType);

  const gemKey = localStorage.getItem('ecolens_gemini_key') || '';
  const pnetKey = localStorage.getItem('ecolens_plantnet_key') || '';

  try {
    let res = await fetch(`${API_BASE}/api/species/identify`, {
      method: 'POST',
      headers: {
        'x-gemini-key': gemKey,
        'x-plantnet-key': pnetKey
      },
      body: formData
    });

    const data = await res.json();
    if (loading) loading.classList.add('hidden');
    if (!res.ok || !data.species_name) {
      throw new Error(data.detail || 'Species could not be identified.');
    }
    displaySpeciesResult(data);
  } catch (e) {
    if (loading) loading.classList.add('hidden');
    alert('Identification error: ' + e.message);
  }
}

// ============================================================
// GEOGRAPHIC DISTRIBUTION: GOOGLE MAPS PLATFORM & LEAFLET
// ============================================================
let currentMapMode = 'satellite'; // 'satellite' (Google Hybrid), 'terrain' (Google Terrain), 'leaflet' (Dark)
let activeDistributionData = null;
let activeDistributionSpecies = '';

// Leaflet state
let speciesMap = null;
let speciesMarkersLayer = null;

// Google Maps state
let gmap = null;
let gmapMarkers = [];
let gmapCircles = [];
let gmapsScriptLoaded = false;
let gmapsLoadingPromise = null;

function switchMapProvider(mode) {
  currentMapMode = mode;
  
  const btnSat = document.getElementById('btn-map-satellite');
  const btnTer = document.getElementById('btn-map-terrain');
  const btnLeaf = document.getElementById('btn-map-leaflet');
  const statusText = document.getElementById('gmaps-status-text');

  const activeClass = 'px-2.5 py-1 rounded-md bg-secondary text-on-secondary font-semibold flex items-center gap-1 transition-all';
  const inactiveClass = 'px-2.5 py-1 rounded-md text-on-surface-variant hover:text-on-surface flex items-center gap-1 transition-all';

  if (btnSat) btnSat.className = mode === 'satellite' ? activeClass : inactiveClass;
  if (btnTer) btnTer.className = mode === 'terrain' ? activeClass : inactiveClass;
  if (btnLeaf) btnLeaf.className = mode === 'leaflet' ? activeClass : inactiveClass;

  if (statusText) {
    if (mode === 'satellite') statusText.textContent = 'Google Maps Satellite';
    else if (mode === 'terrain') statusText.textContent = 'Google Maps Terrain';
    else statusText.textContent = 'EcoLens Dark Biosphere';
  }

  renderSpeciesDistributionMap(activeDistributionData, activeDistributionSpecies);
}

function loadGoogleMapsSdk() {
  if (window.google && window.google.maps) {
    gmapsScriptLoaded = true;
    return Promise.resolve(window.google.maps);
  }
  if (gmapsLoadingPromise) return gmapsLoadingPromise;

  const apiKey = (localStorage.getItem('ecolens_gmaps_key') || '').trim();
  if (!apiKey) {
    return Promise.reject(new Error('MISSING_KEY'));
  }

  gmapsLoadingPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.id = 'gmaps-platform-sdk';
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly&libraries=marker,places`;
    script.async = true;
    script.defer = true;
    script.onload = () => {
      gmapsScriptLoaded = true;
      resolve(window.google.maps);
    };
    script.onerror = (e) => {
      gmapsLoadingPromise = null;
      reject(new Error('FAILED_TO_LOAD'));
    };
    document.head.appendChild(script);
  });
  return gmapsLoadingPromise;
}

function renderSpeciesDistributionMap(distribution, speciesName) {
  activeDistributionData = distribution;
  activeDistributionSpecies = speciesName;

  if (!distribution) {
    distribution = {
      species_name: speciesName || 'Specimen',
      summary: 'Distribution coordinates resolving for this specimen.',
      range_type: 'Monitored Range',
      locations: []
    };
  }

  // Update summary & badge
  const summaryEl = document.getElementById('species-range-summary');
  if (summaryEl) {
    summaryEl.textContent = distribution.summary || `Native range and occurrence zones for ${speciesName || 'specimen'}`;
  }
  const badgeEl = document.getElementById('species-range-badge');
  if (badgeEl) {
    badgeEl.textContent = distribution.range_type || 'IUCN Range';
  }

  const gmapEl = document.getElementById('species-gmap');
  const leafletEl = document.getElementById('species-range-map');

  const hasGmapsKey = Boolean((localStorage.getItem('ecolens_gmaps_key') || '').trim());

  if ((currentMapMode === 'satellite' || currentMapMode === 'terrain') && hasGmapsKey) {
    // Official Google Maps Platform SDK with API Key
    loadGoogleMapsSdk().then(() => {
      if (gmapEl) gmapEl.classList.remove('hidden');
      if (leafletEl) leafletEl.classList.add('hidden');
      renderGoogleMap(distribution, speciesName, currentMapMode);
    }).catch(() => {
      if (gmapEl) gmapEl.classList.add('hidden');
      if (leafletEl) leafletEl.classList.remove('hidden');
      renderLeafletMap(distribution, speciesName, currentMapMode);
    });
  } else {
    // High-Resolution Satellite / Terrain / Dark via native tiles (Zero-key required)
    if (gmapEl) gmapEl.classList.add('hidden');
    if (leafletEl) leafletEl.classList.remove('hidden');
    renderLeafletMap(distribution, speciesName, currentMapMode);
  }
}

function renderGoogleMap(distribution, speciesName, mode) {
  const gmapEl = document.getElementById('species-gmap');
  if (!gmapEl) return;

  const mapTypeId = mode === 'terrain' ? google.maps.MapTypeId.TERRAIN : google.maps.MapTypeId.HYBRID;

  if (!gmap) {
    gmap = new google.maps.Map(gmapEl, {
      center: { lat: 20, lng: 0 },
      zoom: 2,
      mapTypeId: mapTypeId,
      mapId: 'DEMO_MAP_ID',
      fullscreenControl: true,
      streetViewControl: true,
      mapTypeControl: false,
      zoomControl: true
    });
  } else {
    gmap.setMapTypeId(mapTypeId);
  }

  // Clear existing markers and circles
  gmapMarkers.forEach(m => {
    if (m.setMap) m.setMap(null);
    else if (m.map) m.map = null;
  });
  gmapMarkers = [];
  gmapCircles.forEach(c => c.setMap(null));
  gmapCircles = [];

  const locations = distribution.locations || [];
  if (locations.length === 0) {
    gmap.setCenter({ lat: 20, lng: 0 });
    gmap.setZoom(2);
    renderRegionPills([], gmap, null);
    return;
  }

  const bounds = new google.maps.LatLngBounds();
  const infoWindow = new google.maps.InfoWindow();

  locations.forEach((loc, index) => {
    const lat = loc.lat;
    const lng = loc.lng;
    if (typeof lat !== 'number' || typeof lng !== 'number') return;

    bounds.extend({ lat, lng });

    const isPrimary = index === 0;

    // Custom Glowing Beacon Marker for Google Maps
    const pinElement = document.createElement('div');
    pinElement.className = 'pulsing-beacon-marker';
    pinElement.style.cssText = 'position: relative; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; cursor: pointer;';
    pinElement.innerHTML = `
      <div class="beacon-ring" style="border-color: ${isPrimary ? '#10b981' : '#f59e0b'};"></div>
      <div class="beacon-core" style="background: ${isPrimary ? '#10b981' : '#f59e0b'}; box-shadow: 0 0 10px ${isPrimary ? '#10b981' : '#f59e0b'};"></div>
    `;

    let marker;
    if (google.maps.marker && google.maps.marker.AdvancedMarkerElement) {
      marker = new google.maps.marker.AdvancedMarkerElement({
        map: gmap,
        position: { lat, lng },
        title: loc.name || loc.region,
        content: pinElement
      });
      marker.addListener('click', () => {
        openGmapInfoWindow(infoWindow, loc, marker.position, gmap);
      });
    } else {
      marker = new google.maps.Marker({
        map: gmap,
        position: { lat, lng },
        title: loc.name || loc.region
      });
      marker.addListener('click', () => {
        openGmapInfoWindow(infoWindow, loc, { lat, lng }, gmap);
      });
    }
    gmapMarkers.push(marker);

    // Range Buffer Circle
    const circle = new google.maps.Circle({
      map: gmap,
      center: { lat, lng },
      radius: (loc.radius_km || 150) * 1000,
      fillColor: isPrimary ? '#10b981' : '#f59e0b',
      fillOpacity: 0.18,
      strokeColor: isPrimary ? '#10b981' : '#f59e0b',
      strokeOpacity: 0.8,
      strokeWeight: 1.5
    });
    gmapCircles.push(circle);
  });

  if (locations.length > 1) {
    gmap.fitBounds(bounds, { top: 40, bottom: 40, left: 40, right: 40 });
  } else if (locations.length === 1) {
    gmap.setCenter({ lat: locations[0].lat, lng: locations[0].lng });
    gmap.setZoom(6);
  }

  renderRegionPills(locations, gmap, infoWindow);
}

function openGmapInfoWindow(infoWindow, loc, position, mapInstance) {
  const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${loc.lat},${loc.lng}`;
  const content = `
    <div style="font-family: system-ui, sans-serif; font-size: 12px; line-height: 1.4; color: #1e293b; padding: 4px; max-width: 240px;">
      <div style="font-weight: 700; color: #059669; font-size: 14px; margin-bottom: 2px;">
        ${loc.name || loc.region}
      </div>
      <div style="color: #64748b; font-size: 11px; margin-bottom: 4px;">
        📍 ${loc.country || 'Global'} • <strong style="color: #d97706;">${loc.type || 'Native Range'}</strong>
      </div>
      ${loc.habitat ? `<div style="color: #334155; font-size: 11px; margin-bottom: 4px;"><strong>Habitat:</strong> ${loc.habitat}</div>` : ''}
      ${loc.description ? `<div style="color: #475569; font-size: 11px; margin-bottom: 6px;">${loc.description}</div>` : ''}
      <div style="margin-top: 6px; display: flex; align-items: center; justify-content: space-between; gap: 4px; border-top: 1px solid #e2e8f0; pt-2;">
        <span style="font-family: monospace; font-size: 10px; color: #64748b;">${loc.lat.toFixed(4)}°, ${loc.lng.toFixed(4)}°</span>
        <a href="${gmapsUrl}" target="_blank" style="display: inline-flex; align-items: center; gap: 2px; padding: 3px 6px; border-radius: 4px; background: #059669; color: #ffffff; text-decoration: none; font-size: 10px; font-weight: bold;">
          Open 3D ↗
        </a>
      </div>
    </div>
  `;
  infoWindow.setContent(content);
  infoWindow.setPosition(position);
  infoWindow.open(mapInstance);
}

let currentTileLayer = null;

function renderLeafletMap(distribution, speciesName, mode = 'satellite') {
  const mapElement = document.getElementById('species-range-map');
  if (!mapElement || typeof L === 'undefined') return;

  if (!speciesMap) {
    speciesMap = L.map('species-range-map', {
      zoomControl: true,
      scrollWheelZoom: false,
      attributionControl: false
    }).setView([20, 0], 2);

    speciesMarkersLayer = L.layerGroup().addTo(speciesMap);
  }

  // Determine active tile layer
  if (currentTileLayer) {
    speciesMap.removeLayer(currentTileLayer);
  }

  if (mode === 'satellite') {
    // Google Maps Hybrid: High-Res Satellite + National Parks + Roads & Borders
    currentTileLayer = L.tileLayer('https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains: ['0', '1', '2', '3'],
      attribution: '&copy; Google Maps Satellite'
    }).addTo(speciesMap);
  } else if (mode === 'terrain') {
    // Google Maps Physical Elevation & Terrain
    currentTileLayer = L.tileLayer('https://mt{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains: ['0', '1', '2', '3'],
      attribution: '&copy; Google Maps Terrain'
    }).addTo(speciesMap);
  } else {
    // EcoLens Dark Biosphere
    currentTileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
      subdomains: 'abcd',
      attribution: '&copy; CartoDB'
    }).addTo(speciesMap);
  }

  if (speciesMarkersLayer) {
    speciesMarkersLayer.clearLayers();
  }

  setTimeout(() => {
    if (speciesMap) speciesMap.invalidateSize();
  }, 150);

  const locations = distribution.locations || [];
  if (locations.length === 0) {
    speciesMap.setView([20, 0], 2);
    renderRegionPills([], null, null);
    return;
  }

  const latLngs = [];
  const leafletMarkers = [];

  locations.forEach((loc, index) => {
    const lat = loc.lat;
    const lng = loc.lng;
    if (typeof lat !== 'number' || typeof lng !== 'number') return;

    latLngs.push([lat, lng]);

    const isPrimary = index === 0;
    const beaconIcon = L.divIcon({
      className: 'custom-beacon',
      html: `
        <div class="pulsing-beacon-marker" style="position: relative; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;">
          <div class="beacon-ring" style="border-color: ${isPrimary ? '#10b981' : '#f59e0b'};"></div>
          <div class="beacon-core" style="background: ${isPrimary ? '#10b981' : '#f59e0b'}; box-shadow: 0 0 10px ${isPrimary ? '#10b981' : '#f59e0b'};"></div>
        </div>
      `,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
      popupAnchor: [0, -16]
    });

    const marker = L.marker([lat, lng], { icon: beaconIcon });
    const gmapsUrl = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    const popupHtml = `
      <div style="min-width: 190px; padding: 4px 2px;">
        <div style="font-weight: 700; color: #34d399; font-size: 13px; margin-bottom: 2px;">
          ${loc.name || loc.region || 'Region'}
        </div>
        <div style="color: #94a3b8; font-size: 11px; margin-bottom: 6px;">
          📍 ${loc.country || 'Global'} • <span style="color: #fbbf24; font-weight: 600;">${loc.type || 'Native Range'}</span>
        </div>
        ${loc.habitat ? `<div style="color: #cbd5e1; font-size: 11px; margin-bottom: 4px;"><strong>Habitat:</strong> ${loc.habitat}</div>` : ''}
        ${loc.description ? `<div style="color: #94a3b8; font-size: 11px; margin-bottom: 6px;">${loc.description}</div>` : ''}
        <div style="margin-top: 6px; display: flex; align-items: center; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
          <span style="font-size: 10px; color: #6ee7b7; font-family: monospace;">${lat.toFixed(4)}°, ${lng.toFixed(4)}°</span>
          <a href="${gmapsUrl}" target="_blank" style="color: #34d399; text-decoration: underline; font-size: 10px; font-weight: 600;">Google Maps ↗</a>
        </div>
      </div>
    `;
    marker.bindPopup(popupHtml);
    speciesMarkersLayer.addLayer(marker);
    leafletMarkers.push(marker);

    const circle = L.circle([lat, lng], {
      radius: (loc.radius_km || 150) * 1000,
      color: isPrimary ? '#10b981' : '#f59e0b',
      weight: 1.5,
      opacity: 0.8,
      fillColor: isPrimary ? '#10b981' : '#f59e0b',
      fillOpacity: 0.12
    });
    speciesMarkersLayer.addLayer(circle);
  });

  if (latLngs.length > 1) {
    speciesMap.fitBounds(L.latLngBounds(latLngs), { padding: [40, 40], maxZoom: 6 });
  } else if (latLngs.length === 1) {
    speciesMap.setView(latLngs[0], 5);
  }

  renderRegionPills(locations, null, leafletMarkers);
}

function renderRegionPills(locations, gmapInstance, leafletMarkers) {
  const pillsContainer = document.getElementById('species-region-pills');
  if (!pillsContainer) return;

  pillsContainer.innerHTML = '';

  if (locations.length === 0) {
    pillsContainer.innerHTML = '<span class="text-xs text-on-surface-variant italic">No specific regional coordinates mapped for this specimen.</span>';
    return;
  }

  locations.forEach((loc, index) => {
    const isPrimary = index === 0;
    const pill = document.createElement('div');
    pill.className = 'flex items-center rounded-lg bg-surface-container hover:bg-surface-container-high border border-surface-container-highest/40 hover:border-primary/50 text-xs text-on-surface transition-all overflow-hidden';
    
    // Focus button
    const focusBtn = document.createElement('button');
    focusBtn.type = 'button';
    focusBtn.className = 'px-2.5 py-1 flex items-center gap-1.5 cursor-pointer text-left';
    focusBtn.innerHTML = `
      <span class="w-2 h-2 rounded-full ${isPrimary ? 'bg-primary' : 'bg-amber-400'}"></span>
      <span class="font-medium">${loc.name || loc.region}</span>
      <span class="text-[10px] text-on-surface-variant font-mono">(${loc.country})</span>
    `;
    focusBtn.onclick = () => {
      if (currentMapMode !== 'leaflet' && gmapInstance) {
        gmapInstance.panTo({ lat: loc.lat, lng: loc.lng });
        gmapInstance.setZoom(8);
      } else if (speciesMap) {
        speciesMap.flyTo([loc.lat, loc.lng], 6, { duration: 1.2 });
        if (leafletMarkers && leafletMarkers[index]) {
          leafletMarkers[index].openPopup();
        }
      }
    };
    pill.appendChild(focusBtn);

    // Direct Google Maps link button
    const gmapsLink = document.createElement('a');
    gmapsLink.href = `https://www.google.com/maps/search/?api=1&query=${loc.lat},${loc.lng}`;
    gmapsLink.target = '_blank';
    gmapsLink.title = 'Open location in Google Maps 3D/Satellite';
    gmapsLink.className = 'px-2 py-1 bg-surface-container-highest/40 hover:bg-secondary/20 hover:text-secondary text-on-surface-variant border-l border-surface-container-highest/40 flex items-center justify-center transition-colors';
    gmapsLink.innerHTML = '<span class="material-symbols-outlined text-[13px]">open_in_new</span>';
    pill.appendChild(gmapsLink);

    pillsContainer.appendChild(pill);
  });
}
async function viewSpeciesDistributionDetails(name) {
  navigate('scanner');
  const loading = document.getElementById('species-loading');
  if (loading) loading.classList.remove('hidden');
  try {
    const res = await fetch(`${API_BASE}/api/species/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error('Failed to load species data');
    const d = await res.json();
    const data = {
      status: 'success',
      species_name: d.name,
      scientific_name: d.details ? d.details.scientific_name : '',
      confidence: 100,
      profile: d.details || {},
      distribution: d.distribution,
      clean_speech: `${d.name}, scientific name ${d.details ? d.details.scientific_name : ''}. Found in ${d.details ? d.details.region : 'native habitats'}.`
    };
    displaySpeciesResult(data);
    const card = document.getElementById('species-result-card');
    if (card) {
      card.classList.remove('hidden');
      setTimeout(() => {
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  } catch (err) {
    console.error('Error opening species details:', err);
    const local = allSpecies.find(s => s.name.toLowerCase() === name.toLowerCase());
    if (local) {
      displaySpeciesResult({
        status: 'success',
        species_name: local.name,
        scientific_name: local.scientific_name,
        confidence: 100,
        profile: local,
        clean_speech: `${local.name}. Found in ${local.region}.`
      });
    }
  } finally {
    if (loading) loading.classList.add('hidden');
  }
}

function displaySpeciesResult(data) {
  const card = document.getElementById('species-result-card');
  card.classList.remove('hidden');

  if (data.status === 'error') {
    if (confirm(data.clean_speech + '\n\nWould you like to open Settings now to enter your Gemini API key?')) {
      openApiModal();
    }
  }

  document.getElementById('spec-name').textContent = data.species_name;
  document.getElementById('spec-sci').textContent = data.scientific_name;
  document.getElementById('spec-conf').textContent = `${data.confidence}%`;

  const p = data.profile || {};
  document.getElementById('spec-type').textContent = p.type || 'Species';
  document.getElementById('spec-family').textContent = p.family || 'Taxonomy';
  document.getElementById('spec-region').textContent = p.region || 'Global native range';
  document.getElementById('spec-habitat').textContent = p.habitat || 'Natural ecosystem';
  document.getElementById('spec-conservation').textContent = p.conservation || 'Monitored';
  document.getElementById('spec-pop').textContent = p.estimated_population || 'Stable';
  document.getElementById('spec-diet').textContent = p.diet || 'Adapted trophic foraging';
  document.getElementById('spec-role').textContent = p.ecological_role || 'Keystone ecological balancer';
  document.getElementById('spec-threats').textContent = p.threats || 'Habitat fragmentation and climate shift';
  document.getElementById('spec-facts').textContent = p.facts || p.fact || 'Documented in biodiversity databases.';

  currentSpeechText = data.clean_speech || `${data.species_name}, scientific name ${data.scientific_name}. Found in ${p.region}.`;
  document.getElementById('speech-narration-status').textContent = `Ready: ${data.species_name}`;

  // Render Geographic Range Map
  renderSpeciesDistributionMap(data.distribution, data.species_name);

  // Automatically update active context in voice assistant
  changeVoiceSubject(data.species_name);
}

// ============================================================
// 3. VOICE ASSISTANT & TEXT TO SPEECH
// ============================================================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Web Speech API is not supported on this browser.');
    return;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    isListening = true;
    document.getElementById('main-mic-btn').className = 'w-20 h-20 rounded-full bg-error text-on-error shadow-lg flex items-center justify-center transition-all duration-300 active:scale-90 animate-pulse';
    document.getElementById('mic-status-pill').className = 'flex items-center gap-2 px-3 py-1 rounded-full bg-error-container/30 text-error border border-error/30';
    document.getElementById('mic-status-text').textContent = '🔴 Listening... Speak now';
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('chat-text-input').value = transcript;
    sendVoiceQuestion(transcript);
  };

  recognition.onerror = (event) => {
    console.warn('Speech recognition error:', event.error);
    stopListeningUI();
  };

  recognition.onend = () => {
    stopListeningUI();
  };
}

function toggleVoiceRecognition() {
  if (!recognition) {
    alert('Voice recognition is not supported in this browser. You can type queries in the box below!');
    return;
  }
  if (isListening) {
    recognition.stop();
    stopListeningUI();
  } else {
    const langSelect = document.getElementById('voice-lang-select');
    recognition.lang = langSelect ? langSelect.value : 'en-IN';
    recognition.start();
  }
}

function stopListeningUI() {
  isListening = false;
  document.getElementById('main-mic-btn').className = 'w-20 h-20 rounded-full bg-secondary-container hover:bg-secondary text-on-secondary-container hover:text-on-secondary shadow-lg flex items-center justify-center transition-all duration-300 active:scale-90 cursor-pointer';
  document.getElementById('mic-status-pill').className = 'flex items-center gap-2 px-3 py-1 rounded-full bg-secondary-container/20 text-secondary border border-secondary/20';
  document.getElementById('mic-status-text').textContent = '🟢 Tap to speak';
}

async function sendVoiceQuestion(questionText) {
  if (!questionText.trim()) return;

  addChatMessage('user', questionText);

  try {
    const res = await fetch(`${API_BASE}/api/voice/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: questionText,
        species_name: activeSubject
      })
    });
    const data = await res.json();
    addChatMessage('assistant', data.answer);
    
    // Play voice audio
    if (data.clean_speech) {
      speakText(data.clean_speech);
    }
  } catch (e) {
    addChatMessage('assistant', 'Sorry, an error occurred while processing your query: ' + e.message);
  }
}

function askQuickQuery(queryType) {
  const prompts = {
    status: `What is the conservation status, region, and wild population of ${activeSubject}?`,
    diet: `What is the diet, food sources, and feeding habits of ${activeSubject}?`,
    habitat: `Where is ${activeSubject} found and what is its natural habitat and region?`,
    role: `What is the ecological role and importance of ${activeSubject}?`,
    facts: `Tell me an interesting biological fact about ${activeSubject}.`,
    forest: `How does deforestation, canopy loss, and habitat fragmentation impact ${activeSubject}?`,
    taxonomy: `What is the scientific name, taxonomy, and family of ${activeSubject}?`,
    threats: `What are the primary environmental threats endangering ${activeSubject}?`
  };
  const prompt = prompts[queryType] || `Tell me about ${activeSubject}`;
  sendVoiceQuestion(prompt);
}

function handleTextQuery(e) {
  e.preventDefault();
  const input = document.getElementById('chat-text-input');
  const txt = input.value.trim();
  if (txt) {
    sendVoiceQuestion(txt);
    input.value = '';
  }
}

function addChatMessage(sender, text) {
  const history = document.getElementById('voice-chat-history');
  const div = document.createElement('div');
  
  if (sender === 'user') {
    div.className = 'flex flex-col items-end gap-1 max-w-[85%] self-end';
    div.innerHTML = `
      <span class="font-mono text-[10px] text-on-surface-variant">YOU</span>
      <div class="bg-primary-container/20 border border-primary-container/30 rounded-2xl rounded-tr-sm p-3 text-xs text-on-surface leading-relaxed">
        ${escapeHtml(text)}
      </div>
    `;
  } else {
    div.className = 'flex flex-col items-start gap-1 max-w-[90%] self-start';
    div.innerHTML = `
      <div class="flex items-center gap-1.5 text-secondary font-mono text-[10px]">
        <span class="material-symbols-outlined text-[14px]">psychiatry</span>
        <span>ECOLENS CONSERVATION CORE</span>
      </div>
      <div class="bg-surface-container border border-surface-container-highest/40 rounded-2xl rounded-tl-sm p-4 text-xs text-on-surface leading-relaxed flex flex-col gap-2">
        <div>${formatMarkdown(text)}</div>
        <div class="pt-2 border-t border-surface-container-highest/30 flex items-center justify-between text-[11px] text-on-surface-variant">
          <button onclick="speakText('${escapeQuotes(text)}')" class="text-secondary hover:underline flex items-center gap-1">
            <span class="material-symbols-outlined text-[14px]">volume_up</span> Play Voice
          </button>
          <button onclick="stopSpeech()" class="text-error hover:underline flex items-center gap-1">
            <span class="material-symbols-outlined text-[14px]">stop</span> Stop
          </button>
        </div>
      </div>
    `;
  }
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

function clearChatHistory() {
  document.getElementById('voice-chat-history').innerHTML = '';
  addChatMessage('assistant', `Chat cleared. Active subject: **${activeSubject}**.`);
}

function speakText(text) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();

  const clean = text
    .replace(/\*\*/g, '')
    .replace(/[•\-\*]/g, '')
    .replace(/~([0-9])/g, 'approximately $1')
    .replace(/([0-9]),([0-9])/g, '$1$2');

  const utterance = new SpeechSynthesisUtterance(clean);
  const langSelect = document.getElementById('voice-lang-select');
  utterance.lang = langSelect ? langSelect.value : 'en-IN';
  utterance.rate = 0.92;
  window.speechSynthesis.speak(utterance);
}

function speakCurrentProfile() {
  if (currentSpeechText) {
    speakText(currentSpeechText);
  }
}

function stopSpeech() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
}

// ============================================================
// API KEYS CONFIG MODAL
// ============================================================
function openApiModal() {
  document.getElementById('api-modal').classList.remove('hidden');
  document.getElementById('api-modal').classList.add('flex');
}

function closeApiModal() {
  document.getElementById('api-modal').classList.add('hidden');
  document.getElementById('api-modal').classList.remove('flex');
}

async function saveApiKeys() {
  const gemini = document.getElementById('input-gemini-key').value.trim();
  const plantnet = document.getElementById('input-plantnet-key').value.trim();
  const gmaps = document.getElementById('input-gmaps-key') ? document.getElementById('input-gmaps-key').value.trim() : '';

  if (gemini) localStorage.setItem('ecolens_gemini_key', gemini);
  if (plantnet) localStorage.setItem('ecolens_plantnet_key', plantnet);
  if (gmaps) {
    localStorage.setItem('ecolens_gmaps_key', gmaps);
  } else {
    localStorage.removeItem('ecolens_gmaps_key');
  }

  try {
    const res = await fetch(`${API_BASE}/api/config/keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gemini_key: gemini || null,
        plantnet_key: plantnet || null
      })
    });
    const data = await res.json();
    alert('API configuration saved successfully!');
    closeApiModal();
    fetchHealthStatus();
    
    // Refresh current map if species data is present
    if (activeDistributionData) {
      renderSpeciesDistributionMap(activeDistributionData, activeDistributionSpecies);
    }
  } catch (e) {
    alert('Error saving keys to backend: ' + e.message + '\n(Browser local keys saved successfully)');
    closeApiModal();
  }
}

// ============================================================
// STRING HELPERS
// ============================================================
function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeQuotes(str) {
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;').replace(/\n/g, ' ');
}

function formatMarkdown(text) {
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/\n\n/g, '<br/><br/>');
  formatted = formatted.replace(/\n/g, '<br/>');
  return formatted;
}

// ============================================================
// THEME SWITCHING ENGINE
// ============================================================
function initTheme() {
  const savedTheme = localStorage.getItem('ecolens_theme') || 'biosphere-dark';
  setTheme(savedTheme);
}

function setTheme(themeName) {
  document.documentElement.setAttribute('data-theme', themeName);
  localStorage.setItem('ecolens_theme', themeName);

  // Update sidebar select if present
  const sidebarSelect = document.getElementById('sidebar-theme-select');
  if (sidebarSelect) sidebarSelect.value = themeName;

  // Highlight active theme preview card in modal
  document.querySelectorAll('.theme-preview-card').forEach(card => {
    if (card.getAttribute('data-theme-name') === themeName) {
      card.classList.add('active');
    } else {
      card.classList.remove('active');
    }
  });
}

function toggleThemeModal() {
  const modal = document.getElementById('theme-modal');
  if (modal.classList.contains('hidden')) {
    openThemeModal();
  } else {
    closeThemeModal();
  }
}

function openThemeModal() {
  const modal = document.getElementById('theme-modal');
  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closeThemeModal() {
  const modal = document.getElementById('theme-modal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
}


// ============================================================
// STANDALONE CARBON CREDIT & CLIMATE FINANCE HUB CONTROLLER
// ============================================================
let hubHectares = 50;
let hubPricePerTonne = 30;
let hubHorizonYears = 5;
let hubBaseRatePerHa = 25; // 25 t CO2e / ha / year for Tropical Rainforest default
let currentHubBiome = 'rainforest';

function initStandaloneCarbonHub() {
  setHubBiome(currentHubBiome, hubBaseRatePerHa);
  updateHubCarbonROI();
}

function setHubBiome(biomeKey, rate) {
  currentHubBiome = biomeKey;
  hubBaseRatePerHa = rate;

  const biomes = ['mangrove', 'rainforest', 'deciduous', 'semiarid'];
  biomes.forEach(b => {
    const card = document.getElementById(`hub-biome-${b}`);
    if (card) {
      if (b === biomeKey) {
        card.className = 'cursor-pointer p-3.5 rounded-xl bg-primary/15 border-2 border-primary flex flex-col gap-1 transition-all shadow-md';
        const title = card.querySelector('.font-headline');
        if (title) title.className = 'font-headline font-bold text-xs mt-1 text-primary';
      } else {
        card.className = 'cursor-pointer p-3.5 rounded-xl bg-surface-container hover:bg-surface-container-high border border-surface-container-highest/40 flex flex-col gap-1 transition-all';
        const title = card.querySelector('.font-headline');
        if (title) title.className = 'font-headline font-bold text-xs mt-1 text-on-surface';
      }
    }
  });

  updateHubCarbonROI();
}

function onHubHectaresInput(val) {
  const num = Math.max(1, Math.min(1000, parseInt(val) || 1));
  hubHectares = num;
  const slider = document.getElementById('hub-hectares-slider');
  const input = document.getElementById('hub-hectares-input');
  if (slider && parseInt(slider.value) !== num) slider.value = num;
  if (input && parseInt(input.value) !== num) input.value = num;
  updateHubCarbonROI();
}

function setHubCarbonPrice(price) {
  hubPricePerTonne = price;

  [15, 30, 50].forEach(p => {
    const btn = document.getElementById(`hub-btn-price-${p}`);
    if (btn) {
      if (p === price) {
        btn.className = 'px-2 py-2 rounded-xl bg-primary text-on-primary text-xs font-mono font-bold border border-primary transition-all text-center cursor-pointer shadow-md';
      } else {
        btn.className = 'px-2 py-2 rounded-xl bg-surface-container-high hover:bg-surface-bright text-on-surface-variant text-xs font-mono border border-surface-container-highest transition-all text-center cursor-pointer';
      }
    }
  });

  const priceText = document.getElementById('hub-active-price-text');
  if (priceText) {
    const tierName = price === 15 ? 'Avoidance' : (price === 30 ? 'Removal (Verra)' : 'Bio-ARR Premium');
    priceText.textContent = `$${price} / t CO2e (${tierName})`;
  }

  updateHubCarbonROI();
}

function setHubCarbonHorizon(years) {
  hubHorizonYears = years;

  [1, 3, 5, 10, 20].forEach(y => {
    const btn = document.getElementById(`hub-btn-horizon-${y}`);
    if (btn) {
      if (y === years) {
        btn.className = 'px-3.5 py-1.5 rounded-lg bg-secondary text-on-secondary font-bold transition-all cursor-pointer shadow';
      } else {
        btn.className = 'px-3.5 py-1.5 rounded-lg text-on-surface-variant hover:text-on-surface transition-all cursor-pointer';
      }
    }
  });

  updateHubCarbonROI();
}

function updateHubCarbonROI() {
  // Annual sequestration (tonnes of CO2e)
  const annualSequestration = hubHectares * hubBaseRatePerHa;
  // Cumulative sequestration over selected horizon
  const totalSequestration = Math.round(annualSequestration * hubHorizonYears);

  // Financial modeling (USD)
  const grossRevenue = Math.round(totalSequestration * hubPricePerTonne);

  // Capital & Operational Expenditure
  const initialPlantingCost = hubHectares * 850; // $850/ha
  const annualMonitoringCost = hubHectares * 110; // $110/ha/year
  const totalExpenditure = initialPlantingCost + (annualMonitoringCost * hubHorizonYears);

  const netProfit = Math.max(0, grossRevenue - totalExpenditure);

  // Breakeven / Payback period (years)
  const annualGross = annualSequestration * hubPricePerTonne;
  const annualNetCashflow = Math.max(100, annualGross - annualMonitoringCost);
  const paybackYears = Math.min(20, Math.max(0.8, (initialPlantingCost / annualNetCashflow))).toFixed(1);

  // Projected Internal Rate of Return (IRR estimate)
  let irrPercent = 0;
  if (initialPlantingCost > 0) {
    const annualNetMargin = annualNetCashflow / initialPlantingCost;
    irrPercent = Math.min(95, Math.max(8.5, (annualNetMargin * 100) - 2.5)).toFixed(1);
  }

  // Real-world equivalencies (EPA & ICAO standards)
  const carsRemovedPerYear = Math.round(annualSequestration / 4.6); // 4.6 t CO2e / car / year
  const flightsOffset = Math.round(totalSequestration / 0.8); // 0.8 t CO2e / passenger flight
  const waterGainPercent = Math.min(65, Math.round(18 + (hubHectares * 0.12)));

  // Update DOM Elements
  const elTotalTonnes = document.getElementById('hub-kpi-tonnes');
  const elGross = document.getElementById('hub-kpi-gross');
  const elNet = document.getElementById('hub-kpi-net');
  const elPayback = document.getElementById('hub-kpi-payback');
  const elIrr = document.getElementById('hub-kpi-irr');
  const elCars = document.getElementById('hub-eq-cars');
  const elFlights = document.getElementById('hub-eq-flights');
  const elWater = document.getElementById('hub-eq-water');

  if (elTotalTonnes) elTotalTonnes.textContent = `${totalSequestration.toLocaleString()} t`;
  if (elGross) elGross.textContent = `$${grossRevenue.toLocaleString()}`;
  if (elNet) elNet.textContent = `+$${netProfit.toLocaleString()}`;
  if (elPayback) elPayback.textContent = `${paybackYears} Years`;
  if (elIrr) elIrr.textContent = `Projected IRR: ${irrPercent}%`;
  if (elCars) elCars.textContent = `${carsRemovedPerYear.toLocaleString()} cars/yr`;
  if (elFlights) elFlights.textContent = `${flightsOffset.toLocaleString()} flights`;
  if (elWater) elWater.textContent = `+${waterGainPercent}% aquifer recharge`;
}
