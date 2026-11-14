(function () {
  "use strict";

  var meta = null;
  var baseline = {};      // slider values at page load / reset
  var current = {};       // live slider values
  var baselineTemp = null;
  var timer = null;
  var requestId = 0;

  var $ = function (id) { return document.getElementById(id); };

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) { throw new Error(data.error || "Request failed"); }
        return data;
      });
    });
  }

  function showError(msg) {
    var el = $("errorMsg");
    if (msg) { el.textContent = msg; el.hidden = false; } else { el.hidden = true; }
  }

  /* ---------- build the sliders ---------- */
  function buildControls() {
    var box = $("controls");
    box.innerHTML = "";
    meta.features.forEach(function (name) {
      var info = meta.feature_info[name];
      var wrap = document.createElement("div");
      wrap.className = "ctrl";
      wrap.id = "ctrl-" + name;
      wrap.innerHTML =
        '<div class="ctrl-top"><label for="in-' + name + '">' + info.label + '</label>' +
        '<span class="num"><span id="val-' + name + '"></span> ' + info.unit + '</span></div>' +
        '<input type="range" id="in-' + name + '" min="' + info.min + '" max="' + info.max +
        '" step="' + info.step + '" value="' + info.default + '" />' +
        '<div class="range-labels"><span>' + info.min + '</span><span>' + info.max + '</span></div>';
      box.appendChild(wrap);

      var input = wrap.querySelector("input");
      input.addEventListener("input", function () {
        current[name] = Number(input.value);
        refreshLabels();
        scheduleUpdate();
      });
    });
  }

  function refreshLabels() {
    meta.features.forEach(function (name) {
      $("val-" + name).textContent = current[name];
      $("ctrl-" + name).classList.toggle("changed", current[name] !== baseline[name]);
    });
  }

  function buildSweepSelect() {
    var sel = $("sweepFeature");
    sel.innerHTML = "";
    meta.features.forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = meta.feature_info[name].label;
      sel.appendChild(opt);
    });
    sel.value = "humidity";
    sel.addEventListener("change", updateChart);
  }

  /* ---------- predictions ---------- */
  function scheduleUpdate() {
    clearTimeout(timer);
    timer = setTimeout(updatePrediction, 120);
  }

  function updatePrediction() {
    var myId = ++requestId;
    post("/api/predict", { features: current })
      .then(function (data) {
        if (myId !== requestId) { return; }
        showError(null);
        renderPrediction(data.temperature);
        updateChart();
      })
      .catch(function (err) { showError("Could not get prediction: " + err.message); });
  }

  function renderPrediction(temp) {
    $("predValue").textContent = temp.toFixed(1);
    $("newValue").textContent = temp.toFixed(1) + " °C";
    if (baselineTemp === null) { baselineTemp = temp; }
    $("baseValue").textContent = baselineTemp.toFixed(1) + " °C";

    var diff = temp - baselineTemp;
    var box = $("deltaBox");
    if (Math.abs(diff) < 0.05) {
      box.className = "delta neutral";
      box.textContent = "Same as baseline";
    } else if (diff > 0) {
      box.className = "delta up";
      box.textContent = "▲ " + diff.toFixed(1) + " °C warmer than baseline";
    } else {
      box.className = "delta down";
      box.textContent = "▼ " + Math.abs(diff).toFixed(1) + " °C cooler than baseline";
    }
  }

  /* ---------- sweep chart (plain SVG, no external libraries) ---------- */
  function updateChart() {
    var feature = $("sweepFeature").value;
    post("/api/sweep", { feature: feature, features: current })
      .then(function (data) { drawChart(data, feature); })
      .catch(function (err) { showError("Could not draw chart: " + err.message); });
  }

  function drawChart(data, feature) {
    var info = meta.feature_info[feature];
    var W = 760, H = 340, pl = 56, pr = 20, pt = 20, pb = 48;
    var xs = data.x, ys = data.y;
    var xmin = xs[0], xmax = xs[xs.length - 1];
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    var pad = Math.max((ymax - ymin) * 0.15, 0.5);
    ymin -= pad; ymax += pad;

    function sx(x) { return pl + (x - xmin) / (xmax - xmin) * (W - pl - pr); }
    function sy(y) { return H - pb - (y - ymin) / (ymax - ymin) * (H - pt - pb); }

    var svg = '<svg viewBox="0 0 ' + W + " " + H + '" xmlns="http://www.w3.org/2000/svg">';

    // horizontal grid + y labels
    for (var i = 0; i <= 4; i++) {
      var yv = ymin + (ymax - ymin) * i / 4;
      var py = sy(yv);
      svg += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + py + '" y2="' + py + '" stroke="#e2e8f1" />';
      svg += '<text x="' + (pl - 8) + '" y="' + (py + 4) + '" text-anchor="end" font-size="12" fill="#5d6b82">' + yv.toFixed(1) + "</text>";
    }
    // x labels
    for (var j = 0; j <= 4; j++) {
      var xv = xmin + (xmax - xmin) * j / 4;
      var px = sx(xv);
      svg += '<text x="' + px + '" y="' + (H - pb + 20) + '" text-anchor="middle" font-size="12" fill="#5d6b82">' + Math.round(xv) + "</text>";
    }
    // axis titles
    svg += '<text x="' + (pl + (W - pl - pr) / 2) + '" y="' + (H - 8) + '" text-anchor="middle" font-size="13" fill="#14233a">' +
           info.label + (info.unit ? " (" + info.unit + ")" : "") + "</text>";
    svg += '<text transform="translate(14 ' + (pt + (H - pt - pb) / 2) + ') rotate(-90)" text-anchor="middle" font-size="13" fill="#14233a">Predicted temperature (°C)</text>';

    // line
    var path = "";
    for (var k = 0; k < xs.length; k++) {
      path += (k === 0 ? "M" : "L") + sx(xs[k]).toFixed(1) + " " + sy(ys[k]).toFixed(1) + " ";
    }
    svg += '<path d="' + path + '" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />';

    // current point: interpolate along the curve
    var cx = Math.min(Math.max(current[feature], xmin), xmax);
    var cy = interpolate(xs, ys, cx);
    svg += '<line x1="' + sx(cx) + '" x2="' + sx(cx) + '" y1="' + pt + '" y2="' + (H - pb) + '" stroke="#2563eb" stroke-dasharray="4 4" opacity=".4" />';
    svg += '<circle cx="' + sx(cx) + '" cy="' + sy(cy) + '" r="7" fill="#fff" stroke="#dc2626" stroke-width="3" />';
    svg += '<text x="' + sx(cx) + '" y="' + (sy(cy) - 14) + '" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">' + cy.toFixed(1) + " °C</text>";

    svg += "</svg>";
    $("chart").innerHTML = svg;
  }

  function interpolate(xs, ys, x) {
    for (var i = 0; i < xs.length - 1; i++) {
      if (x >= xs[i] && x <= xs[i + 1]) {
        var t = (x - xs[i]) / (xs[i + 1] - xs[i] || 1);
        return ys[i] + t * (ys[i + 1] - ys[i]);
      }
    }
    return ys[ys.length - 1];
  }

  /* ---------- model info panels ---------- */
  function renderModelInfo() {
    $("bestModelLine").textContent = "Selected model: " + meta.best_model + " (lowest average error).";
    var tbody = document.querySelector("#metricsTable tbody");
    tbody.innerHTML = "";
    Object.keys(meta.metrics).forEach(function (name) {
      var m = meta.metrics[name];
      var tr = document.createElement("tr");
      if (name === meta.best_model) { tr.className = "best"; }
      tr.innerHTML = "<td>" + name + "</td><td>" + m.r2.toFixed(3) + "</td><td>" + m.mae.toFixed(2) + "</td><td>" + m.rmse.toFixed(2) + "</td>";
      tbody.appendChild(tr);
    });

    var box = $("importance");
    box.innerHTML = "";
    var items = Object.keys(meta.feature_importance).map(function (k) { return [k, meta.feature_importance[k]]; });
    items.sort(function (a, b) { return b[1] - a[1]; });
    items.forEach(function (it) {
      var pct = Math.round(it[1] * 100);
      var row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = "<span>" + meta.feature_info[it[0]].label + '</span><div class="bar-track"><div class="bar-fill" style="width:' +
        Math.max(pct, 1) + '%"></div></div><span class="bar-pct">' + pct + "%</span>";
      box.appendChild(row);
    });

    $("dataNote").textContent = meta.data_note + " (" + meta.train_rows + " training rows, " + meta.test_rows + " test rows)";
  }

  /* ---------- start ---------- */
  function init() {
    fetch("/api/meta")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        meta = data;
        meta.features.forEach(function (name) {
          baseline[name] = meta.feature_info[name].default;
          current[name] = baseline[name];
        });
        buildControls();
        buildSweepSelect();
        refreshLabels();
        renderModelInfo();
        updatePrediction();

        $("resetBtn").addEventListener("click", function () {
          meta.features.forEach(function (name) {
            current[name] = baseline[name];
            $("in-" + name).value = baseline[name];
          });
          refreshLabels();
          updatePrediction();
        });
      })
      .catch(function (err) { showError("Could not load the model information: " + err.message); });
  }

  init();
})();
