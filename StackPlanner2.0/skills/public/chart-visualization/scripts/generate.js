#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

// Chart type mapping, consistent with src/utils/callTool.ts
const CHART_TYPE_MAP = {
  generate_area_chart: "area",
  generate_bar_chart: "bar",
  generate_boxplot_chart: "boxplot",
  generate_column_chart: "column",
  generate_district_map: "district-map",
  generate_dual_axes_chart: "dual-axes",
  generate_fishbone_diagram: "fishbone-diagram",
  generate_flow_diagram: "flow-diagram",
  generate_funnel_chart: "funnel",
  generate_histogram_chart: "histogram",
  generate_line_chart: "line",
  generate_liquid_chart: "liquid",
  generate_mind_map: "mind-map",
  generate_network_graph: "network-graph",
  generate_organization_chart: "organization-chart",
  generate_path_map: "path-map",
  generate_pie_chart: "pie",
  generate_pin_map: "pin-map",
  generate_radar_chart: "radar",
  generate_sankey_chart: "sankey",
  generate_scatter_chart: "scatter",
  generate_treemap_chart: "treemap",
  generate_venn_chart: "venn",
  generate_violin_chart: "violin",
  generate_word_cloud_chart: "word-cloud",
};

function getVisRequestServer() {
  return (
    process.env.VIS_REQUEST_SERVER ||
    "https://antv-studio.alipay.com/api/gpt-vis"
  );
}

function getServiceIdentifier() {
  return process.env.SERVICE_ID;
}

async function httpPost(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  return response.json();
}

async function generateChartUrl(chartType, options) {
  const url = getVisRequestServer();
  const payload = {
    type: chartType,
    source: "chart-visualization-creator",
    ...options,
  };

  const data = await httpPost(url, payload);

  if (!data.success) {
    throw new Error(data.errorMessage || "Unknown error");
  }

  return data.resultObj;
}

async function generateMap(tool, inputData) {
  const url = getVisRequestServer();
  const payload = {
    serviceId: getServiceIdentifier(),
    tool,
    input: inputData,
    source: "chart-visualization-creator",
  };

  const data = await httpPost(url, payload);

  if (!data.success) {
    throw new Error(data.errorMessage || "Unknown error");
  }

  return data.resultObj;
}

function parseOutputFile(argv) {
  const flagIndex = argv.indexOf("--output-file");
  if (flagIndex === -1) {
    return null;
  }
  const outputFile = argv[flagIndex + 1];
  if (!outputFile || outputFile.startsWith("--")) {
    throw new Error("--output-file requires a path");
  }
  return outputFile;
}

function extractHttpUrl(value) {
  if (typeof value === "string") {
    const match = value.match(/https?:\/\/[^\s\"'<>]+/);
    return match ? match[0] : null;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const url = extractHttpUrl(item);
      if (url) return url;
    }
    return null;
  }
  if (value && typeof value === "object") {
    for (const item of Object.values(value)) {
      const url = extractHttpUrl(item);
      if (url) return url;
    }
  }
  return null;
}

async function downloadChart(url, outputFile) {
  const parsed = new URL(url);
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`Unsupported chart URL protocol: ${parsed.protocol}`);
  }

  const response = await fetch(parsed, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`Unable to download chart: HTTP ${response.status}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length === 0) {
    throw new Error("Unable to download chart: empty response");
  }

  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, bytes);
  return bytes.length;
}

async function main() {
  if (process.argv.length < 3) {
    console.error("Usage: node generate.js <spec_json_or_file> [--output-file <path>]");
    process.exit(1);
  }

  const specArg = process.argv[2];
  let outputFile;
  try {
    outputFile = parseOutputFile(process.argv.slice(3));
  } catch (e) {
    console.error(e.message);
    process.exit(1);
  }
  let spec;

  try {
    if (fs.existsSync(specArg)) {
      const fileContent = fs.readFileSync(specArg, "utf-8");
      spec = JSON.parse(fileContent);
    } else {
      spec = JSON.parse(specArg);
    }
  } catch (e) {
    console.error(`Error parsing spec: ${e.message}`);
    process.exit(1);
  }

  const specs = Array.isArray(spec) ? spec : [spec];
  if (outputFile && specs.length !== 1) {
    console.error("Error: --output-file accepts exactly one chart spec; run once per chart");
    process.exit(1);
  }

  for (const item of specs) {
    const tool = item.tool;
    const args = item.args || {};

    if (!tool) {
      console.error(
        `Error: 'tool' field missing in spec: ${JSON.stringify(item)}`,
      );
      continue;
    }

    const chartType = CHART_TYPE_MAP[tool];
    if (!chartType) {
      console.error(`Error: Unknown tool '${tool}'`);
      continue;
    }

    const isMapChartTool = [
      "generate_district_map",
      "generate_path_map",
      "generate_pin_map",
    ].includes(tool);

    try {
      let result;
      if (isMapChartTool) {
        result = await generateMap(tool, args);
        if (result && result.content) {
          for (const contentItem of result.content) {
            if (contentItem.type === "text") {
              console.log(contentItem.text);
            }
          }
        } else {
          console.log(JSON.stringify(result));
        }
      } else {
        result = await generateChartUrl(chartType, args);
        console.log(result);
      }

      if (outputFile) {
        const chartUrl = extractHttpUrl(result);
        if (!chartUrl) {
          throw new Error("Generator response did not contain a downloadable HTTP(S) image URL");
        }
        const byteCount = await downloadChart(chartUrl, outputFile);
        console.log(`Saved chart: ${outputFile} (${byteCount} bytes)`);
        console.log(`Source URL: ${chartUrl}`);
      }
    } catch (e) {
      console.error(`Error generating chart for ${tool}: ${e.message}`);
      process.exitCode = 1;
    }
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err.message);
    process.exit(1);
  });
}

// Export functions for testing
module.exports = {
  generateChartUrl,
  generateMap,
  httpPost,
  parseOutputFile,
  extractHttpUrl,
  downloadChart,
  CHART_TYPE_MAP,
};
